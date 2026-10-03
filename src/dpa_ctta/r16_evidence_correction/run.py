"""Finite pipeline: source qualification/selection, frozen target matrix, CPU scoring."""
import concurrent.futures
import contextlib
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import numpy as np
import torch
from ..r10_12h_core.run import read,save,sha
from ..r9_current_first.storage import lease
from ..r9_current_first.physical import Meter
from ..r9_current_first.assets import gpu_policy,available_memory
from ..r8_ba.journal import _digest,TargetJournal,verify_online_complete
from ..r8_ba.streams import rows_sha
from ..r10_use_write_rl.runtime import classification
from .methods import STATIC
from .zero_order import KINDS,SEED

ID='R16_EVIDENCE_CORRECTION_V1'
OPS=('model_forwards','backward_calls','optimizer_steps','vjp_calls')

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or sha(c)!=os.environ['RUN_CONFIG_SHA']:raise ValueError('config identity changed')
    return c

class Guard:
    def __init__(self,c,phase):
        self.c=c;self.root=Path(c['output_root']);self.phase=phase;self.deadline=float(os.environ['RUN_DEADLINE']);self.last=0;self.meter=None
        self.extra=dict(head_forwards=0,zero_order_updates=0,image_accesses=0,partial_forwards=0);self.last_disk=0
    def observe(self,cost=None):
        if time.time()>self.deadline-10:raise TimeoutError('task deadline; close reserve preserved')
        if time.monotonic()-self.last>10:
            if self.phase!='score':
                totals=live_charges(self.c)
                cap=43200 if int(os.environ.get('RUN_ATTEMPT','0')) else 36000
                stage='preflight' if self.phase=='preflight' else 'source' if self.phase.startswith('source_') else 'target'
                if totals['all']>cap-30 or totals[stage]>{'preflight':3600,'source':10800,'target':21600}[stage]-30:raise TimeoutError('aggregate GPU-worker/stage cap; closing margin')
            size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file() and not p.is_symlink())
            if size>self.c['origin']['cache_peak_cap_bytes']:raise RuntimeError('8GiB private artifact/cache cap')
            save(self.root/'live'/(self.phase+'.json'),dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().split()[21],phase=self.phase,at=time.time(),cost=cost or {},extra=self.extra,disk_bytes=size))
            self.last=time.monotonic()
    def __call__(self):self.observe(self.meter.cost if self.meter else None)

def forbid_target_labels(c):
    binding=c['bindings']['refs']['target'];reg=read(binding['path']);forbidden={str(Path(r['mask_path']).resolve()) for r in reg['target']}
    def audit(event,args):
        if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
            if str(Path(os.fsdecode(args[0])).resolve()) in forbidden:raise PermissionError('target label access forbidden in algorithm worker')
    sys.addaudithook(audit)

def prepare():
    root=Path(os.environ['RUN_ROOT']);origin=read(root/'BUDGET_ORIGIN.json');parent=read(root/'private/parent-config.json');bindings=read(root/'private/bindings.json')
    from ..r9_current_first.assets import validate_metadata
    from ..r7_source_prep.registry import verified
    from ..r10_use_write_rl.streams import sequence
    validate_metadata(bindings);ref=bindings['refs']['target'];reg=json.loads(verified(ref['path'],ref['sha256'],16*1024**2))
    short=read(Path(parent['previous_root'])/'RESOLVED_CONFIG.json');manifests={}
    for tier,arr in (('FULL',[sequence(reg,0),sequence(reg,1)]),('SHORT',[read(x['path']) for x in short['manifests']])):
        expected=(1951,1695) if tier=='FULL' else (1024,888)
        if any((len(a),sum(r['subset']=='remaining_dev' for r in a))!=expected for a in arr):raise ValueError('arrival/principal protocol counts')
        if any(len({r['group_id'] for r in a})!=len(a) for a in arr) or {r['group_id'] for r in arr[0]}!={r['group_id'] for r in arr[1]}:raise ValueError('static unique same-content order qualification')
        manifests[tier]=[]
        for o,rows in enumerate(arr):
            p=root/'private'/f'{tier}_o{o}.json';save(p,rows);manifests[tier].append(dict(path=str(p),sha256=rows_sha(rows),visits=expected[0],principal=expected[1]))
    c=dict(experiment_id=ID,code_sha=os.environ['RUN_SHA'],base_sha=origin['base_sha'],output_root=str(root),origin=origin,bindings=bindings,
           execution_version=int(os.environ.get('RUN_VERSION','1')),preflight_attempt=int(os.environ.get('RUN_PREFLIGHT_ATTEMPT','0')),
           gpu_assignments=read(root/'private/gpus.json'),manifests=manifests,source_indices=[16*m+i for m in range(4) for i in (0,4,10,13)],
           source_seed=SEED,target_Z_seeds=[SEED,SEED+1],conditions=list(STATIC)+list(KINDS)+['G'],ROI='full_grid',postprocessing='none',
           scientific_parameters=dict(feature='up3',channels=256,grid=[128,128],prototype_temperature=.1,seed_erosion_radius=2,boundary_radius=6,
           seed_minimum=16,prototype_amplitude=.25,thresholds=[.4,.45,.5,.55,.6],matching_IoU=.5,minimum_levels=3,semantic_margin=.05,
           edit_budget_fraction=.02,interpolation='bilinear align_corners=False, seeds nearest',head_input_channels=269,head_hidden_channels=32,
           head_input_rounding='float16 -> float32 both source/target',head_updates=2000,source_augmentations='fit identity; val existing frozen simulator',
           mu_eta=[[.001,.0001],[.001,.0005],[.003,.0001],[.003,.0005]],Z_loss='channel RMS scale, stopped shared FG/BG groups, equal groups/channels'),
           probabilities='source real soft retained; target probability metrics NA, packed hard masks only',patient_dependence='unknown image grouping; exposed DEVELOPMENT',
           historical_config_sha256=sha(parent),short_config_sha256=sha(short),max_concurrent=2,recovery='one evidenced equivalent infrastructure recovery per physical task')
    save(root/'RESOLVED_CONFIG.json',c);save(root/'RUN_STATE.json',dict(status='PREPARING',experiment_id=ID,jobs={},target_scores_embargoed=True));return c

def profile_admit(c):
    root=Path(c['output_root']);p=read(root/'SOURCE_PRECHECK.json')['profile']
    static=max(p['static_seconds']);z=max(p['z_pair_seconds']);io=max(p['IO_seconds'])
    source=1.3*(4096*z+1167*static+2000*(max(p['train_pair_seconds'])+p['cache_batch_read_seconds'])+600)
    projections=[];chosen=None
    for tier,n in (('FULL',1951),('SHORT',1024)):
        target=1.3*(n*(static+io+.13)+12*n*(z+.13)+14*60)
        projected=source+target+charged(c)
        ok=source<=10800 and target<=21600 and projected<=36000 and time.time()+(source+target)/2<c['origin']['normal_compute_deadline_epoch']
        projections.append(dict(tier=tier,source_seconds=source,target_seconds=target,total_gpu_seconds=projected,admitted=ok))
        if ok and chosen is None:chosen=tier
    result=dict(admitted=chosen is not None,tier=chosen,candidates=projections,safety_factor=1.3,measurements=p,charged_before=charged(c),target_access=0)
    save(root/'PROFILE_ADMISSION.json',result);return result

def charged(c):
    root=Path(c['output_root']);total=0.
    for p in (root/'attempts').glob('*.json'):total+=read(p).get('cost',{}).get('gpu_seconds',0.)
    return total

def live_charges(c):
    root=Path(c['output_root']);totals=dict(all=0.,preflight=0.,source=0.,target=0.);completed=set()
    def add(phase,amount):
        totals['all']+=amount
        key='preflight' if phase=='preflight' else 'source' if phase.startswith('source_') else 'target' if phase.startswith('online_') else None
        if key:totals[key]+=amount
    for p in (root/'attempts').glob('*.json'):
        r=read(p);completed.add((r['phase'],r['attempt']));add(r['phase'],r['cost'].get('gpu_seconds',0))
    for p in (root/'processes').glob('*.json'):
        r=read(p)
        if r['active'] and (r['phase'],r['attempt']) not in completed and r['phase']!='score':add(r['phase'],max(0,time.time()-r['started']))
    return totals

def reuse(c,tier):
    from ..r13_full_coverage.coverage import scalar_rows,compose
    root=Path(c['output_root']);parent=read(root/'private/parent-config.json');short=read(Path(parent['previous_root'])/'RESOLVED_CONFIG.json');refs=[]
    def sealed(p):
        on=read(p/'online_complete.json');sc=read(p/'score_complete.json')
        if sc['scalar_sha256']!=_digest(p/'scalars.private.jsonl') or sc['visits']!=on['visits']:raise ValueError('historical scalar seal')
        if sc['schema']=='R10_SCORE_COMPLETE_V1' and sc['online_sha256']!=sha(on):raise ValueError('historical probability identity')
        if sc['schema']=='R8_SCORE_COMPLETE_V1' and (sc['online_prediction_sha256']!=on['prediction_sha256'] or sc['rows_sha256']!=on['identity']['rows_sha256']):raise ValueError('historical bitmask identity')
        return scalar_rows(p/'scalars.private.jsonl'),dict(online=sha(on),score=sha(sc),scalar=sc['scalar_sha256'])
    for o in (0,1):
        rows=read(c['manifests'][tier][o]['path']);prepared={}
        if tier=='FULL':
            for arm in ('C0','G'):
                r=next(r for r in parent['reused_results'] if r['condition']==arm and r['order']==o and r['role']=='FULL_HISTORICAL');p=Path(r['path'])
                if read(p/'online_complete.json')['identity']['rows_sha256']!=rows_sha(rows):raise ValueError('full reference arrival identity')
                values,seal=sealed(p);prepared[arm]=(values,[seal])
            p=Path(parent['previous_root'])/'target'/f'CV_H025_o{o}';q=Path(parent['output_root'])/'target'/f'CV_H025_o{o}'
            a,sa=sealed(p);b,sb=sealed(q);prepared['H025']=(compose(rows,[a,b]),[sa,sb])
        else:
            for arm in ('C0','G'):
                r=next(r for r in short['reused_results'] if r['job']['order']==o and r['job']['arm']==('C0_CURRENT_STATS' if arm=='C0' else 'G_CTTA'))
                p=Path(r['root'])/'target'/r['job']['id'];values,seal=sealed(p)
                if read(p/'online_complete.json')['identity']['rows_sha256']!=rows_sha(rows):raise ValueError('short reference arrival identity')
                prepared[arm]=(values,[seal])
            p=Path(parent['previous_root'])/'target'/f'CV_H025_o{o}';prepared['H025']=sealed(p)
            prepared['H025']=(prepared['H025'][0],[prepared['H025'][1]])
        for arm,(values,seals) in prepared.items():
            if len(values)!=len(rows) or any((r['content'],r['domain'],r['subset'])!=(m['group_id'],m['domain'],m['subset']) for r,m in zip(values,rows)):raise ValueError('reference pair/order/content')
            p=root/'private'/f'reference_{arm}_o{o}.json';save(p,values);refs.append(dict(condition=arm,order=o,values_path=str(p),values_sha256=sha(values),seals=seals,origin='HISTORICAL_MATCHED' if arm!='H025' or tier=='SHORT' else 'STATIC_DISJOINT_SCALAR_COMPOSITION'))
    return refs

def freeze_target(c,admission):
    root=Path(c['output_root']);z=read(root/'SOURCE_Z.json');d=read(root/'SOURCE_D.json')
    if z['status']!='COMPLETE' or d['status']!='COMPLETE':raise ValueError('source selection incomplete')
    target=next(r['target_seconds'] for r in admission['candidates'] if r['tier']==admission['tier'])
    if charged(c)+target>36000 or time.time()+target/2>c['origin']['normal_compute_deadline_epoch']:raise RuntimeError('NOT_RUN_BUDGET after source costs')
    chosen={k:dict(v,sha256=_digest(Path(v['file']))) for k,v in d['selected'].items()}
    p=dict(experiment_id=ID,config_sha256=sha(c),code_sha=c['code_sha'],checkpoint_sha256=c['bindings']['checkpoint_sha256'],
           source_manifest=c['bindings']['refs']['manifest']['sha256'],source_split=c['bindings']['refs']['split']['sha256'],registration=c['bindings']['refs']['target']['sha256'],
           tier=admission['tier'],manifests=c['manifests'][admission['tier']],Z_selected=z['selected'],D_selected=chosen,
           statistics_sha256=_digest(root/'private/head-statistics.pt'),source_results=dict(Z=sha(z),D=sha(d),STATIC=sha(read(root/'SOURCE_STATIC.json'))),
           reused=reuse(c,admission['tier']),target_seeds=c['target_Z_seeds'],scope='one finite registered matrix, DEVELOPMENT, no followon')
    lock=dict(payload=p,sha256=sha(p));save(root/'EXPERIMENT_LOCK.json',lock);return lock

def native_segmenter(c):
    from ..r10_use_write_rl.factory import construct
    h,close=construct(dict(arm='C0',seed=SEED,source_job=None,artifact=None),c,c['output_root'],dict(sha256=sha(c)))
    return h.segmenter,close

class OnlineHost:
    def __init__(self,c,lock,segmenter,arm,order,seed):
        from .methods import Engine,ResidualHead
        from .zero_order import Adapter
        from ..r10_carrier.source import bn_record,parameter_stamp
        self.c=c;self.lock=lock;self.seg=segmenter;self.arm=arm;self.order=order;self.seed=seed;self.visits=0
        self.context=dict(sha256=sha(dict(lock=lock['sha256'],arm=arm,order=order,seed=seed)))
        self.before=dict(parameters=parameter_stamp(segmenter),BN=bn_record(segmenter));self.engine=None;self.adapter=None
        if arm=='STATIC':
            self.engine=Engine(segmenter);self.heads={}
            for kind,ref in lock['payload']['D_selected'].items():
                if _digest(Path(ref['file']))!=ref['sha256']:raise ValueError('source head changed')
                h=ResidualHead().cuda();h.load_state_dict(torch.load(ref['file'],weights_only=True));h.eval().requires_grad_(False);self.heads[kind]=h
            self.statistics=torch.load(Path(c['output_root'])/'private/head-statistics.pt',weights_only=True)
        else:
            z=lock['payload']['Z_selected'];self.adapter=Adapter(segmenter,arm,z['mu'],z['eta'],seed,order)
    def check_frozen(self,boundary=False):
        if boundary:
            from ..r10_carrier.source import bn_record,parameter_stamp
            if self.before!=dict(parameters=parameter_stamp(self.seg),BN=bn_record(self.seg)):raise ValueError('online backbone/BN changed')
    def snapshot(self):return dict(visits=self.visits,adapter=None if self.adapter is None else self.adapter.snapshot())
    def restore(self,x):
        self.visits=x['visits']
        if self.adapter:self.adapter.restore(x['adapter'])
        elif x['adapter'] is not None:raise ValueError('static state contamination')
    def close(self):
        if self.engine:self.engine.close()
        if self.adapter:self.adapter.close()

def online(c,guard,job,failure=None):
    from ..r7_target_screen.runner import TargetReader,image_records
    root=Path(c['output_root']);lock=read(root/'EXPERIMENT_LOCK.json')
    def reject_optimizer(*args,**kwargs):raise PermissionError('target algorithm cannot construct an optimizer')
    torch.optim.Optimizer.__init__=reject_optimizer
    if sha(lock['payload'])!=lock['sha256'] or lock['payload']['config_sha256']!=sha(c):raise ValueError('target lock changed')
    m=lock['payload']['manifests'][job['order']];rows=read(m['path'])
    if rows_sha(rows)!=m['sha256']:raise ValueError('target manifest changed')
    seg,close=native_segmenter(c);guard.meter.attach(seg.model);host=OnlineHost(c,lock,seg,job['arm'],job['order'],job['seed'])
    p=root/'target'/job['id'];width=65536*(len(STATIC) if job['arm']=='STATIC' else 1)
    journal=TargetJournal(p,host,job['id'],rows_sha(rows),width);reader=TargetReader(c['bindings']['target_root'],256*1024**2,'image')
    try:
        if failure:journal.recover_once(failure)
        else:journal.create()
        for i in range(host.visits,len(rows)):
            guard();image=reader.read(image_records((rows[i],))[0]);guard.extra['image_accesses']+=1;start=time.perf_counter()
            if job['arm']=='STATIC':
                outputs,diag,_=host.engine.outputs(image,host.heads,host.statistics);guard.extra['head_forwards']+=2
                # Keep only explicit fixed arm order; dictionaries are not a scientific ordering guarantee.
                from .structure import hard as mask_from_logits
                masks=np.stack([mask_from_logits(outputs[k]) for k in STATIC]);trace=dict(diagnostics=diag)
            else:
                z,trace=host.adapter.step(image);masks=(z.float().sigmoid()>=.5).numpy()[0]
                guard.extra['zero_order_updates']+=trace['updates']
            host.visits+=1;trace.update(visit=host.visits,state_committed=True,seconds=time.perf_counter()-start)
            journal.append(np.packbits(masks).tobytes(),trace)
        receipt=journal.complete(len(rows));save(p/'deployment_context.json',host.context);return receipt
    finally:reader.after_check();host.close();close()

def worker():
    c=config();phase=os.environ['RUN_PHASE'];root=Path(c['output_root']);assignment=json.loads(os.environ['RUN_ASSIGNMENT']);guard=Guard(c,phase);start=float(os.environ['RUN_STARTED']);failure=None;result=None;meter=None
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    # The selected physical device is execution metadata, not a source-choice variable.
    local=dict(c,gpu_assignments=[assignment]);cpu=phase=='score'
    def stop_group():os.killpg(os.getpgrp(),signal.SIGTERM)
    timer=threading.Timer(max(.1,guard.deadline-time.time()),stop_group);timer.daemon=True;timer.start()
    try:
        if cpu:
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('scorer must be independent CPU')
            from .score import score
            result=score(c,guard)
        else:
            gpu_policy(assignment);available_memory(assignment,3*1024**3);forbid_target_labels(c)
            with lease(root/'leases'/f'gpu{assignment["physical_id"]}',dict(experiment_id=ID,GPU=assignment)):
                with Meter(None,guard.observe) as meter:
                    guard.meter=meter
                    from . import source
                    if phase=='preflight':result=source.preflight(local,guard)
                    elif phase=='source_Z':result=source.calibrate(local,guard)
                    elif phase=='source_D':result=source.train_heads(local,guard)
                    elif phase.startswith('online_'):result=online(c,guard,json.loads(os.environ['RUN_JOB']),json.loads(os.environ['RUN_FAILURE']) if os.environ.get('RUN_FAILURE') else None)
                    else:raise ValueError('unknown phase')
                    if phase.startswith('online_') and any(meter.cost[k] for k in ('backward_calls','optimizer_steps','vjp_calls')):raise ValueError('target uses only frozen or zero-order updates')
    except BaseException as e:
        failure=dict(reason=str(e),error_type=type(e).__name__,classification=classification(e),phase=phase);traceback.print_exc()
    finally:timer.cancel()
    cost=meter.cost.copy() if meter else dict.fromkeys((*OPS,'gpu_seconds'),0)
    cost['gpu_seconds']=0 if cpu else time.time()-start;cost.update(guard.extra)
    r=dict(status='FAILED' if failure else 'COMPLETE',phase=phase,attempt=int(os.environ.get('RUN_ATTEMPT','0')),started=start,ended=time.time(),wall_seconds=time.time()-start,cost=cost,
           assignment=None if cpu else assignment,failure=failure,result=result,code_sha=c['code_sha'],config_sha256=sha(c))
    save(root/'attempts'/f'{phase}.{r["attempt"]}.json',r);return r

def process_identity(p):return dict(pid=p.pid,pgid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],active=True,argv=Path(f'/proc/{p.pid}/cmdline').read_bytes().replace(b'\0',b' ').decode().strip())

def run_task(c,phase,assignment,seconds,job=None,attempt=0,failure=None):
    root=Path(c['output_root']);start=time.time();deadline=min(start+seconds,c['origin']['compute_deadline_epoch'] if attempt else c['origin']['normal_compute_deadline_epoch'])
    env=dict(os.environ,RUN_MODE='worker',RUN_PHASE=phase,RUN_CONFIG_SHA=sha(c),RUN_STARTED=str(start),RUN_DEADLINE=str(deadline),RUN_ASSIGNMENT=json.dumps(assignment),RUN_ATTEMPT=str(attempt),CUDA_VISIBLE_DEVICES='' if phase=='score' else str(assignment['physical_id']))
    if job:env['RUN_JOB']=json.dumps(job)
    if failure:env['RUN_FAILURE']=json.dumps(failure)
    log=root/'logs'/f'{phase}.{attempt}.log'
    with log.open('ab') as f:
        p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,start_new_session=True,stdout=f,stderr=subprocess.STDOUT)
        ident=process_identity(p);ident.update(phase=phase,assignment=assignment,started=start,deadline=deadline,attempt=attempt);save(root/'processes'/f'{phase}.json',ident)
        try:p.wait(timeout=max(.1,deadline-time.time())+5)
        except subprocess.TimeoutExpired:
            try:os.killpg(p.pid,signal.SIGTERM);p.wait(timeout=5)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        ident.update(active=False,ended=time.time(),exit_code=p.returncode);save(root/'processes'/f'{phase}.json',ident)
    path=root/'attempts'/f'{phase}.{attempt}.json'
    if path.exists():return read(path)
    result=dict(status='FAILED',phase=phase,attempt=attempt,started=start,ended=time.time(),cost=dict(gpu_seconds=0 if phase=='score' else time.time()-start),
                failure=dict(reason='worker exited without receipt',error_type='MissingReceipt',classification='RESOURCE',exit_code=p.returncode),code_sha=c['code_sha'])
    save(path,result);return result

def update_ledger(c,state):
    attempts=[read(p) for p in sorted((Path(c['output_root'])/'attempts').glob('*.json'))]
    ledger=dict(experiment_id=ID,T0=c['origin']['T0'],gpu_seconds=sum(r['cost'].get('gpu_seconds',0) for r in attempts),wall_seconds=time.time()-c['origin']['T0_epoch'],
                operations={k:sum(r['cost'].get(k,0) for r in attempts) for k in (*OPS,'head_forwards','zero_order_updates','image_accesses','partial_forwards')},attempts=attempts,status=state['status'],gpu_cap=43200)
    save(Path(c['output_root'])/'RESOURCE_LEDGER.json',ledger);return ledger

def supervise():
    c=config();root=Path(c['output_root']);state=read(root/'RUN_STATE.json')
    with lease(root/'supervisor_control',dict(experiment_id=ID,config=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('execution exists; reconcile, do not restart')
        save(root/'execution_started.json',dict(at=time.time(),pid=os.getpid(),config_sha256=sha(c)));state.update(status='SOURCE_PREFLIGHT_RUNNING');save(root/'RUN_STATE.json',state)
        try:
            a=c['gpu_assignments'][0];r=run_task(c,'preflight',a,max(30,3600-live_charges(c)['preflight']),attempt=c.get('preflight_attempt',0));state['jobs']['preflight']=r['status'];update_ledger(c,state)
            if r['status']!='COMPLETE':raise RuntimeError('qualification failed; no target access')
            old=root/'history/v1/SOURCE_PRECHECK.json'
            if old.exists():
                before=read(old)['rows'];after=read(root/'SOURCE_PRECHECK.json')['rows'];maximum=0.
                if len(before)!=len(after):raise ValueError('equivalent qualification coverage')
                for a,b in zip(before,after):
                    if (a['episode'],a['visit'],a['mode'])!=(b['episode'],b['visit'],b['mode']):raise ValueError('equivalent source role')
                    for condition in a['metrics']:
                        for metric,value in a['metrics'][condition].items():maximum=max(maximum,abs(value-b['metrics'][condition][metric]))
                    if a['prototype']!=b['prototype']:raise ValueError('equivalent prototype evidence changed')
                save(root/'SOURCE_OPTIMIZATION_PARITY.json',dict(passed=maximum==0,images=32,all_conditions_hard_soft_max_delta=maximum,scientific_rules_changed=False))
                if maximum!=0:raise ValueError('candidate optimization changed source metrics')
            admission=profile_admit(c)
            if not admission['admitted']:raise RuntimeError('NOT_RUN_BUDGET even short full matrix')
            state.update(status='SOURCE_SELECTION_RUNNING',tier=admission['tier']);save(root/'RUN_STATE.json',state)
            source_estimate=next(x['source_seconds'] for x in admission['candidates'] if x['tier']==admission['tier'])
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                futures={executor.submit(run_task,c,phase,gpu,min(5400,source_estimate)):phase for phase,gpu in zip(('source_Z','source_D'),c['gpu_assignments'][:2])}
                for f in concurrent.futures.as_completed(futures):
                    r=f.result();state['jobs'][futures[f]]=r['status'];save(root/'RUN_STATE.json',state);update_ledger(c,state)
            if any(state['jobs'].get(x)!='COMPLETE' for x in ('source_Z','source_D')):raise RuntimeError('source route failed; preserve all evidence before target')
            # Retire only this run's reproducible source feature cache before target journals.
            cache=root/'source_cache';files=list(cache.glob('*.pt'));size=sum(p.stat().st_size for p in files)
            save(root/'SOURCE_CACHE_RETIRED.json',dict(bytes=size,files=len(files),index_sha256=sha(read(root/'private/source-cache-index.json')),reason='selected source heads and fit statistics retained; 8GiB peak cap'))
            for p in files:p.unlink()
            lock=freeze_target(c,admission);state.update(status='TARGET_RUNNING',target_lock_sha256=lock['sha256']);save(root/'RUN_STATE.json',state)
            jobs=[dict(id='STATIC_o0',arm='STATIC',order=0,seed=SEED)]+[dict(id=f'{k}_o{o}_s{s}',arm=k,order=o,seed=s) for k in KINDS for o in (0,1) for s in c['target_Z_seeds']]
            state['jobs'].update({j['id']:'NOT_RUN' for j in jobs});save(root/'RUN_STATE.json',state)
            per=next(x['target_seconds'] for x in admission['candidates'] if x['tier']==admission['tier']);deadline=time.time()+min(per,21600)/2+600
            for offset in range(0,len(jobs),2):
                batch=jobs[offset:offset+2]
                if charged(c)>36000-120 or time.time()>deadline:raise RuntimeError('target normal budget stop')
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                    futures={executor.submit(run_task,c,'online_'+j['id'],gpu,max(60,min(3600,deadline-time.time())),j):j for j,gpu in zip(batch,c['gpu_assignments'][:2])}
                    for f in concurrent.futures.as_completed(futures):
                        j=futures[f];r=f.result()
                        if r['status']=='FAILED' and r['failure']['classification']=='INFRASTRUCTURE' and charged(c)+900<=43200:
                            failure=dict(**r['failure'],**{'class':'INFRASTRUCTURE','evidence':dict(receipt=sha(r),phase=r['phase'])})
                            r=run_task(c,'online_'+j['id'],c['gpu_assignments'][batch.index(j)],900,j,1,failure)
                        state['jobs'][j['id']]=r['status'];save(root/'RUN_STATE.json',state);update_ledger(c,state)
            state.update(status='TARGET_MATRIX_TERMINAL',target_scores_embargoed=True);save(root/'RUN_STATE.json',state)
            r=run_task(c,'score',None,max(60,min(7200,c['origin']['absolute_deadline_epoch']-time.time()-300)))
            state['jobs']['score']=r['status'];state.update(status='COMPLETE' if r['status']=='COMPLETE' and all(state['jobs'][j['id']]=='COMPLETE' for j in jobs) else 'PARTIAL',target_scores_embargoed=False)
        except BaseException as e:
            traceback.print_exc();state.update(status='STOPPED',stop_reason=str(e));state['jobs']={k:('NOT_RUN_STOPPED' if v in ('RUNNING','NOT_RUN') else v) for k,v in state['jobs'].items()}
            if (root/'EXPERIMENT_LOCK.json').exists() and any(k.startswith(('STATIC','Z_')) and v=='COMPLETE' for k,v in state['jobs'].items()):
                state.update(status='TARGET_MATRIX_TERMINAL');save(root/'RUN_STATE.json',state)
                r=run_task(c,'score',None,max(60,min(7200,c['origin']['absolute_deadline_epoch']-time.time()-300)))
                state['jobs']['score']=r['status'];state.update(status='PARTIAL',target_scores_embargoed=False)
        state.update(ended=time.time(),delivery='PENDING_LOCAL_GITHUB');save(root/'RUN_STATE.json',state);update_ledger(c,state)
        from .report import report
        report(c,state);return state

def watch():
    c=config();root=Path(c['output_root']);p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=dict(os.environ,RUN_MODE='supervise'),start_new_session=True)
    save(root/'watchdog.json',dict(pid=os.getpid(),supervisor_pid=p.pid,absolute_deadline=c['origin']['absolute_deadline_epoch']))
    try:p.wait(timeout=max(.1,c['origin']['absolute_deadline_epoch']-time.time()))
    finally:
        for f in (root/'processes').glob('*.json'):
            a=read(f);stat=Path(f'/proc/{a["pid"]}/stat')
            if a['active'] and stat.exists() and stat.read_text().split()[21]==a['start_ticks']:
                try:os.killpg(a['pgid'],signal.SIGKILL)
                except ProcessLookupError:pass
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGKILL)
            except ProcessLookupError:pass

def main():
    mode=os.environ['RUN_MODE']
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',sha256=sha(prepare()))))
    elif mode=='worker':r=worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='supervise':supervise()
    elif mode=='watch':watch()
    else:raise ValueError('unknown execution mode')

if __name__=='__main__':main()
