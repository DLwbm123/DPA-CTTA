"""R17 uses the proven finite worker/watch/lease/ledger/journal machinery."""
import concurrent.futures,json,os,signal,sys,threading,time,traceback
from pathlib import Path
import numpy as np
import torch
from ..r16_evidence_correction import run as base
from ..r10_12h_core.run import read,save,sha
from ..r9_current_first.storage import lease
from ..r9_current_first.physical import Meter
from ..r9_current_first.assets import gpu_policy,available_memory
from ..r8_ba.journal import TargetJournal,_digest
from ..r8_ba.streams import rows_sha
from ..r10_use_write_rl.runtime import classification
from .method import ARMS,SEED,TARGET_SEED,LAMBDA,construct,load_heads

ID='R17_GRATA_INTEGRATED_V1'

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or sha(c)!=os.environ['RUN_CONFIG_SHA']:raise ValueError('frozen configuration identity')
    return c

def install():
    # Explicit callbacks in this process only; sealed R16 source files remain untouched.
    base.ID=ID;base.config=config

def prepare():
    r=Path(os.environ['RUN_ROOT']);origin=read(r/'BUDGET_ORIGIN.json');parent=read(r/'private/parent-config.json')
    manifests=[]
    for o,m in enumerate(parent['manifests']['FULL']):
        rows=read(m['path'])
        if (len(rows),sum(x['subset']=='remaining_dev' for x in rows))!=(1951,1695) or rows_sha(rows)!=m['sha256']:raise ValueError('full manifest identity')
        f=r/'private'/f'FULL_o{o}.json';save(f,rows);manifests.append(dict(m,path=str(f)))
    c=dict(experiment_id=ID,code_sha=os.environ['RUN_SHA'],base_sha=origin['base_sha'],output_root=str(r),origin=origin,bindings=parent['bindings'],
       gpu_assignments=read(r/'private/gpus.json'),parent_root=parent['output_root'],parent_config_sha256=sha(parent),manifests=manifests,
       source_seed=SEED,target_seed=TARGET_SEED,conditions=['C0','DS','M_STATIC','G',*ARMS],target_arms=list(ARMS),lambda_correction=LAMBDA,
       formula='BCE(student,multiview teacher) + .1 * mean_M(BCE(student,q_head)-BCE(student,q_DS)); teacher stopped',
       source=dict(folds=['fit','val'],updates=['C','G'],episodes=[0,16,32,48],visits_per_episode=32,head_steps=2000,head_checkpoints=[500,1000,1500,2000],batch=8,lr=.001,wd=.0001,source_selection_only=True,selected_head_closed_loop_visits=96),
       budget_rule='FULL only; if 1.3 projection fails stop before target; no arm deletion or target tuning',patient_dependence='unknown; same exposed DEVELOPMENT images across orders',max_concurrent=2)
    save(r/'RESOLVED_CONFIG.json',c);save(r/'RUN_STATE.json',dict(experiment_id=ID,status='PREPARING',jobs={},target_scores_embargoed=True));return c

def admission(c):
    r=Path(c['output_root']);p=read(r/'SOURCE_PRECHECK.json')['profile'];per=max(p['native_seconds']+p['active_seconds'])
    source=1.3*(608*(per+.12)+2000*(max(p['train_pair_seconds'])+.08)+600)
    target=1.3*(6*1951*(per+.15)+6*60)
    total=base.charged(c)+source+target
    result=dict(admitted=source<=10800 and target<=21600 and total<=36000 and time.time()+source+target<c['origin']['normal_compute_deadline_epoch'],tier='FULL',source_seconds=source,target_seconds=target,total_gpu_seconds=total,safety_factor=1.3,profile=p,target_accesses=0)
    save(r/'PROFILE_ADMISSION.json',result);return result

def freeze(c,a):
    r=Path(c['output_root']);d=read(r/'SOURCE_D.json');old=Path(c['parent_root']);receipt=read(old/'SCORER_RECEIPT.json');p=old/'score/all-scalars.private.jsonl'
    if d['status']!='COMPLETE' or receipt['status']!='COMPLETE' or receipt['scalar_sha256']!=_digest(p):raise ValueError('source/reference not sealed')
    if base.charged(c)+a['target_seconds']>36000 or time.time()+a['target_seconds']>c['origin']['normal_compute_deadline_epoch']:raise RuntimeError('post-source target budget admission failed')
    refs=[];raw={}
    for line in p.open():
        row=json.loads(line)
        if row['condition'] in ('C0','DS','D_CONTEXT','G'):raw.setdefault((row['condition'],row['order']),[]).append(row)
    if set(raw)!={(a,o) for a in ('C0','DS','D_CONTEXT','G') for o in (0,1)}:raise ValueError('all eight reference streams required')
    for (arm,o),values in raw.items():
        values.sort(key=lambda x:x['visit']);manifest=read(c['manifests'][o]['path'])
        if len(values)!=1951 or any((x['content'],x['domain'],x['subset'])!=(m['group_id'],m['domain'],m['subset']) for x,m in zip(values,manifest)):raise ValueError('reference content/order/role pairing')
        name='M_STATIC' if arm=='D_CONTEXT' else arm;f=r/'private'/f'ref_{name}_o{o}.json';save(f,[dict(v,condition=name) for v in values]);refs.append(dict(condition=name,order=o,values_path=str(f),values_sha256=sha(read(f)),origin='R16 sealed exact per-image scalar reference',parent_scalar_sha256=receipt['scalar_sha256']))
    selected={k:dict(v,sha256=_digest(Path(v['file']))) for k,v in d['selected'].items()}
    p=dict(experiment_id=ID,config_sha256=sha(c),code_sha=c['code_sha'],checkpoint_sha256=c['bindings']['checkpoint_sha256'],registration_sha256=c['bindings']['refs']['target']['sha256'],manifests=c['manifests'],D_selected=selected,statistics_sha256=_digest(r/'private/head-statistics.pt'),source_results_sha256=sha(d),reused=refs,arms=list(ARMS),target_seed=TARGET_SEED,lambda_correction=LAMBDA)
    lock=dict(payload=p,sha256=sha(p));save(r/'EXPERIMENT_LOCK.json',lock);return lock

def online(c,guard,job,failure=None):
    from ..r7_target_screen.runner import TargetReader,image_records
    r=Path(c['output_root']);lock=read(r/'EXPERIMENT_LOCK.json');p=lock['payload']
    if sha(p)!=lock['sha256'] or p['config_sha256']!=sha(c):raise ValueError('target lock identity')
    for ref in p['D_selected'].values():
        if _digest(Path(ref['file']))!=ref['sha256']:raise ValueError('head artifact changed')
    if _digest(r/'private/head-statistics.pt')!=p['statistics_sha256']:raise ValueError('fit statistics changed')
    heads,stats=load_heads(r,p['D_selected']);kind='D_LOGIT' if job['arm']=='G_LOGIT' else 'D_CONTEXT'
    host,corr,close=construct(c,job['arm'],TARGET_SEED,lock,heads[kind],stats)
    guard.meter.attach(host.native.model);host.context['payload']['r17']=dict(target_lock=lock['sha256'],job=job);host.context['sha256']=sha(host.context['payload'])
    m=p['manifests'][job['order']];rows=read(m['path'])
    if rows_sha(rows)!=m['sha256']:raise ValueError('target arrival mutation')
    reader=TargetReader(c['bindings']['target_root'],256*1024**2,'image');dest=r/'target'/job['id'];journal=TargetJournal(dest,host,job['id'],m['sha256'])
    before={k:v.detach().clone() for k,v in heads[kind].state_dict().items()}
    try:
        if failure:journal.recover_once(failure)
        else:journal.create()
        for i in range(host.visits,len(rows)):
            guard();image=reader.read(image_records((rows[i],))[0]);guard.extra['image_accesses']+=1;start=time.perf_counter();z,trace=host.step(image);guard.extra['head_forwards']+=trace['correction']['head_forwards'];trace['seconds']=time.perf_counter()-start
            if any(v.grad is not None for v in heads[kind].parameters()):raise ValueError('target correction head gradient')
            journal.append(np.packbits((z.float().sigmoid()>=.5).numpy()[0]).tobytes(),trace)
        if any(not torch.equal(v,before[k]) for k,v in heads[kind].state_dict().items()):raise ValueError('target head changed')
        receipt=journal.complete(len(rows));save(dest/'deployment_context.json',host.context);return receipt
    finally:reader.after_check();close()

def worker():
    c=config();phase=os.environ['RUN_PHASE'];r=Path(c['output_root']);assignment=json.loads(os.environ['RUN_ASSIGNMENT']);guard=base.Guard(c,phase);start=float(os.environ['RUN_STARTED']);cpu=phase=='score';failure=None;result=None;meter=None
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    timer=threading.Timer(max(.1,guard.deadline-time.time()),lambda:os.killpg(os.getpgrp(),signal.SIGTERM));timer.daemon=True;timer.start()
    try:
        if cpu:
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU scorer GPU visibility')
            from .score import score
            result=score(c,guard)
        else:
            gpu_policy(assignment);available_memory(assignment,5*1024**3);base.forbid_target_labels(c)
            with lease(r/'leases'/f'gpu{assignment["physical_id"]}',dict(experiment_id=ID,GPU=assignment)):
                with Meter(None,guard.observe) as meter:
                    guard.meter=meter;local=dict(c,gpu_assignments=[assignment])
                    from . import source
                    if phase=='preflight':result=source.preflight(local,guard)
                    elif phase=='source_D':result=source.train(local,guard)
                    elif phase.startswith('online_'):result=online(c,guard,json.loads(os.environ['RUN_JOB']),json.loads(os.environ['RUN_FAILURE']) if os.environ.get('RUN_FAILURE') else None)
                    else:raise ValueError('unknown phase')
    except BaseException as e:failure=dict(reason=str(e),error_type=type(e).__name__,classification=classification(e),phase=phase);traceback.print_exc()
    finally:timer.cancel()
    cost=meter.cost.copy() if meter else dict.fromkeys((*base.OPS,'gpu_seconds'),0);cost['gpu_seconds']=0 if cpu else time.time()-start;cost.update(guard.extra)
    row=dict(status='FAILED' if failure else 'COMPLETE',phase=phase,attempt=int(os.environ.get('RUN_ATTEMPT','0')),started=start,ended=time.time(),wall_seconds=time.time()-start,cost=cost,assignment=None if cpu else assignment,failure=failure,result=result,code_sha=c['code_sha'],config_sha256=sha(c));save(r/'attempts'/f'{phase}.{row["attempt"]}.json',row);return row

def supervise():
    c=config();r=Path(c['output_root']);state=read(r/'RUN_STATE.json');jobs=[]
    with lease(r/'supervisor_control',dict(experiment_id=ID,config=sha(c))):
        if (r/'execution_started.json').exists():raise ValueError('execution already started; no duplicate')
        save(r/'execution_started.json',dict(at=time.time(),pid=os.getpid(),config_sha256=sha(c)))
        try:
            state.update(status='SOURCE_PREFLIGHT_RUNNING');save(r/'RUN_STATE.json',state)
            result=base.run_task(c,'preflight',c['gpu_assignments'][0],3600);state['jobs']['preflight']=result['status'];base.update_ledger(c,state)
            if result['status']!='COMPLETE':raise RuntimeError('source qualification failed')
            a=admission(c)
            if not a['admitted']:raise RuntimeError('NOT_RUN_BUDGET; complete full matrix required')
            state.update(status='SOURCE_TRAINING_RUNNING');save(r/'RUN_STATE.json',state)
            result=base.run_task(c,'source_D',c['gpu_assignments'][0],a['source_seconds']);state['jobs']['source_D']=result['status'];base.update_ledger(c,state)
            if result['status']!='COMPLETE':raise RuntimeError('source training incomplete')
            lock=freeze(c,a);files=list((r/'source_cache').glob('*.pt'));save(r/'SOURCE_CACHE_RETIRED.json',dict(files=len(files),bytes=sum(p.stat().st_size for p in files),reason='source selected artifacts retained before target'))
            for p in files:p.unlink()
            jobs=[dict(id=f'{k}_o{o}',arm=k,order=o,seed=TARGET_SEED) for k in ARMS for o in (0,1)];state['jobs'].update({j['id']:'NOT_RUN' for j in jobs});state.update(status='TARGET_RUNNING',target_lock_sha256=lock['sha256']);save(r/'RUN_STATE.json',state)
            deadline=min(time.time()+21600,c['origin']['normal_compute_deadline_epoch'])
            for start in range(0,len(jobs),2):
                if base.charged(c)>36000-120 or time.time()>deadline:raise RuntimeError('target normal budget stop')
                batch=jobs[start:start+2]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    future={pool.submit(base.run_task,c,'online_'+j['id'],gpu,max(30,deadline-time.time()),j):j for j,gpu in zip(batch,c['gpu_assignments'][:2])}
                    for f in concurrent.futures.as_completed(future):
                        j=future[f];row=f.result()
                        if row['status']=='FAILED' and row['failure']['classification']=='INFRASTRUCTURE' and base.charged(c)+900<43200:
                            failure=dict(row['failure'],**{'class':'INFRASTRUCTURE','evidence':dict(receipt=sha(row),phase=row['phase'])});row=base.run_task(c,'online_'+j['id'],c['gpu_assignments'][batch.index(j)],900,j,1,failure)
                        state['jobs'][j['id']]=row['status'];save(r/'RUN_STATE.json',state);base.update_ledger(c,state)
            state.update(status='TARGET_MATRIX_TERMINAL');save(r/'RUN_STATE.json',state)
            row=base.run_task(c,'score',None,max(30,min(7200,c['origin']['absolute_deadline_epoch']-time.time()-300)));state['jobs']['score']=row['status'];state.update(status='COMPLETE' if row['status']=='COMPLETE' and all(state['jobs'][j['id']]=='COMPLETE' for j in jobs) else 'PARTIAL',target_scores_embargoed=False)
        except BaseException as e:
            traceback.print_exc();state.update(status='STOPPED',stop_reason=str(e));state['jobs']={k:'NOT_RUN_STOPPED' if v in ('NOT_RUN','RUNNING') else v for k,v in state['jobs'].items()}
            if (r/'EXPERIMENT_LOCK.json').exists() and any(state['jobs'].get(j['id'])=='COMPLETE' for j in jobs):
                state.update(status='TARGET_MATRIX_TERMINAL');save(r/'RUN_STATE.json',state)
                row=base.run_task(c,'score',None,max(30,min(7200,c['origin']['absolute_deadline_epoch']-time.time()-300)))
                state['jobs']['score']=row['status'];state.update(status='PARTIAL',target_scores_embargoed=False)
        state.update(ended=time.time(),delivery='PENDING_LOCAL_GITHUB');save(r/'RUN_STATE.json',state);base.update_ledger(c,state)
        from .score import report
        report(c,state)

def main():
    install();mode=os.environ['RUN_MODE']
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',sha256=sha(prepare()))))
    elif mode=='worker':sys.exit(0 if worker()['status']=='COMPLETE' else 1)
    elif mode=='supervise':supervise()
    elif mode=='watch':base.watch()
    else:raise ValueError('unknown mode')
