"""Bounded coverage successor using existing image-only journal and independent scorer."""
import json,os,time,traceback,signal,subprocess,sys
from pathlib import Path
from collections import Counter
from ..r10_12h_core.run import read,save,sha,epoch,Guard,launch,terminate_group,OPS
from ..r12_horizontal_weight.view import construct,VIEWS,WEIGHTS,AGGREGATION
from .coverage import scalar_rows,compose,hard_parity
ID='R13_FULL_COVERAGE_V1'

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or os.environ.get('RUN_CONFIG_SHA',sha(c))!=sha(c):raise ValueError('frozen config identity')
    return c

def lock(c,name):
    if name not in ('C0','CV_H025'):raise ValueError('unregistered condition')
    p=dict(experiment_id=ID,condition=name,config_sha256=sha(c),views=[list(v) for v in VIEWS[name]],weights=list(WEIGHTS[name]),checkpoint_sha256=c['bindings']['checkpoint_sha256'],aggregation=AGGREGATION)
    return dict(schema=ID+'_LOCK',payload=p,sha256=sha(p))

def check_lock(x):
    p=x.get('payload',{});name=p.get('condition')
    if x.get('schema')!=ID+'_LOCK' or x.get('sha256')!=sha(p) or p.get('experiment_id')!=ID or name not in ('C0','CV_H025') or p.get('weights')!=list(WEIGHTS[name]) or p.get('views')!=[list(v) for v in VIEWS[name]] or p.get('aggregation')!=AGGREGATION:raise ValueError('frozen view lock')

def prepare():
    from ..r8_ba.streams import rows_sha
    from ..r8_ba.journal import _digest
    from ..r7_source_prep.registry import verified
    from ..r10_use_write_rl.streams import sequence
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PREVIOUS_ROOT']);oc=read(old/'RESOLVED_CONFIG.json');origin=read(root/'BUDGET_ORIGIN.json');previous=read(old/'RESOURCE_LEDGER.json')
    if read(old/'RUN_STATE.json')['status']!='COMPLETE' or not read(old/'SOURCE_REFERENCE_PARITY.json')['passed'] or time.time()>epoch(origin['preflight_deadline']):raise ValueError('previous/source/preflight qualification')
    if abs(previous['prior_package_charged_seconds']+previous['actual_wall_seconds']-origin['prior_package_charged_seconds'])>.01:raise ValueError('cost lineage')
    if origin['prior_gpu_worker_seconds']+origin['new_cap_seconds']>86400 or epoch(origin['absolute_deadline'])>1790952587.771787-3600:raise ValueError('campaign reserve/cap')
    ref=oc['bindings']['refs']['target'];reg=json.loads(verified(ref['path'],ref['sha256'],16*1024**2));manifests=[];full_manifests=[];reuse=[];proof=[]
    def seal(path,expected,n):
        on=read(path/'online_complete.json');sc=read(path/'score_complete.json')
        if on['visits']!=n or sc['visits']!=n or on['identity']['rows_sha256']!=expected or sc['scalar_sha256']!=_digest(path/'scalars.private.jsonl'):raise ValueError('reused scalar/arrival seal')
        if sc['schema']=='R10_SCORE_COMPLETE_V1':
            if sc['online_sha256']!=sha(on):raise ValueError('probability score identity')
        elif sc['schema']=='R8_SCORE_COMPLETE_V1':
            if sc['rows_sha256']!=expected or sc['online_prediction_sha256']!=on['prediction_sha256'] or sc['context_sha256']!=on['identity']['context_sha256']:raise ValueError('legacy bit score identity')
        else:raise ValueError('score schema')
        return dict(path=str(path),online=on,score=sc)
    for o in (0,1):
        full=sequence(reg,o);short=read(oc['manifests'][o]['path']);keep={r['group_id'] for r in short};complement=[r for r in full if r['group_id'] not in keep]
        if rows_sha(short)!=oc['manifests'][o]['sha256'] or len(keep)!=1024 or len(complement)!=927 or sum(r['subset']=='remaining_dev' for r in complement)!=807:raise ValueError('disjoint complement counts')
        for rows,arr,name in ((full,full_manifests,'full'),(complement,manifests,'complement')):
            path=root/'private'/f'{name}_o{o}.json';save(path,rows);arr.append(dict(path=str(path),sha256=rows_sha(rows),visits=len(rows),principal=sum(r['subset']=='remaining_dev' for r in rows),strata=dict(Counter(f"{r['domain']}|{r['subset']}" for r in rows))))
        q=seal(old/'target'/f'CV_H025_o{o}',rows_sha(short),1024);q.update(condition='CV_H025',order=o,role='SHORT_PARTITION');reuse.append(q)
        for name,jid in (('C0',f'C0_CURRENT_STATS_order{o}'),('G',f'G_CTTA_RELEASE_TRANSFER_20260907_order{o}')):
            p=Path(os.environ['FULL_BASELINE_ROOT'])/jid;x=seal(p,rows_sha(full),1951);x.update(condition=name,order=o,role='FULL_HISTORICAL');ctx=read(p/'deployment_context.json');payload=ctx['payload']
            if ctx['sha256']!=x['online']['identity']['context_sha256']:raise ValueError('historical deployment identity')
            if name=='G':
                ident=payload['identity']
                if payload['arm']!='G_CTTA_RELEASE_TRANSFER' or ident['checkpoint_sha256']!=oc['bindings']['checkpoint_sha256'] or ident['registration_sha256']!=ref['sha256'] or ident['seed']!=20260907 or ident['code_sha']!='2329fb625ee023f461c23d1256162eb5514906c6':raise ValueError('full native baseline binding')
            elif payload['method'] is not None or payload['source']['checkpoint_sha256']!=oc['bindings']['checkpoint_sha256'] or payload['environment']['policy']['preprocessing']!='pinned_fundus_minmax_float32_batch1_512_v1':raise ValueError('full C0 binding')
            rs=scalar_rows(p/'scalars.private.jsonl')
            if any((r['visit'],r['content'],r['domain'],r['subset'])!=(i+1,m['group_id'],m['domain'],m['subset']) for i,(r,m) in enumerate(zip(rs,full))) or len(rs)!=1951:raise ValueError('full baseline row order')
            if name=='C0':
                sr=next(x for x in oc['reused_results'] if x['job']['arm']=='C0_CURRENT_STATS' and x['job']['order']==o);short_path=Path(sr['root'])/'target'/sr['job']['id'];seal(short_path,rows_sha(short),1024);proof.append(dict(order=o,**hard_parity(scalar_rows(short_path/'scalars.private.jsonl'),rs)))
            reuse.append(x)
    costs=oc['old_profile_prior_cost'].copy()
    for k in OPS:costs[k]+=previous['operations'][k]
    c=dict(experiment_id=ID,code_sha=os.environ['RUN_SHA'],base_sha=os.environ['BASE_SHA'],origin=origin,output_root=str(root),previous_root=str(old),bindings=oc['bindings'],gpu_assignments=oc['gpu_assignments'],old_profile_prior_cost=costs,manifests=manifests,full_manifests=full_manifests,reused_results=reuse,reuse_parity=proof,reused_source_sha256=sha(read(old/'SOURCE_COMPARISON.json')),jobs=[dict(id=f'CV_H025_o{o}',arm='CV_H025',order=o,seed=20260924) for o in (0,1)],patient_dependence='Unknown; exposed original development cohort, two orders share content; not independent confirmation',status='PREFLIGHT',candidate_conditions=['CV_H025'],source_conditions=['C0','CV_H025'])
    save(root/'RESOLVED_CONFIG.json',c);save(root/'RESOURCE_LEDGER.json',origin);save(root/'REUSE_QUALIFICATION.json',dict(passed=True,full_visits=1951,full_principal=1695,complement_visits=927,complement_principal=807,short_C0_hard_parity=proof,legacy_soft_metrics='NA; bit-mask hard metrics only',core_source_compatibility='Verified unchanged six core source files from historical native execution to parent commit before deployment'))
    return c

def preflight(c,guard):
    import torch
    from ..r10_carrier.source import data,bn_record,parameter_stamp
    from ..r10_carrier.diagnostic import source_item
    timings={k:0. for k in ('C0','CV_H025')};reference={};checks=[]
    with torch.no_grad(),data(c,guard) as src:
        images=[(i,v,source_item(src,i,v)[0]) for i in (0,16,32,48) for v in (0,4,8,12,16,20,24,28)]
        for order in (0,1):
            for name in timings:
                h,close=construct(name,c,lock(c,name));handle=guard.meter.attach(h.segmenter.model)
                before=dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter))
                if any(m['training'] or m['track_running_stats'] or m['running_mean'] is not None or m['running_var'] is not None for m in before['BN'].values()):raise ValueError('current statistics BN')
                try:
                    for i,v,image in images if order==0 else reversed(images):
                        guard();torch.cuda.synchronize();t=time.perf_counter();z=h.step(image)[0];torch.cuda.synchronize();timings[name]+=time.perf_counter()-t;z=z.detach().cpu()
                        if order==0:reference[name,i,v]=z
                        elif not torch.equal(reference[name,i,v],z):raise ValueError('order-dependent predictive state')
                    h.check_frozen(True)
                    if before!=dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)):raise ValueError('backbone changed')
                    checks.append(dict(condition=name,order=order,immutable=True,visits=32))
                finally:handle.remove();close()
    proof=dict(passed=True,source_images=32,logical_visits=128,model_forwards=192,reverse_order_logits_exact=True,parameters_BN_unchanged=True,checks=checks,timings=timings,config_sha256=sha(c))
    save(Path(c['output_root'])/'ZERO_PARITY.json',proof);return dict(passed=True,visits=128)

def target(c,guard,phase,failure):
    from ..r10_use_write_rl.target import online,score,retire_probabilities
    from ..r8_ba.streams import rows_sha
    j=next(j for j in c['jobs'] if j['id']==phase.split(':',1)[1]);p=Path(c['output_root'])/'target'/j['id'];m=c['manifests'][j['order']];rows=read(m['path']);lk=lock(c,j['arm'])
    if rows_sha(rows)!=m['sha256']:raise ValueError('frozen complement')
    if phase.startswith('score:'):
        on=read(p/'online_complete.json');r=score(rows,c['bindings']['target_root'],p,j['id'],on['identity']['context_sha256'],lk,guard,failure,check_lock);retire_probabilities(p);return r
    host,close=construct(j['arm'],c,lk);handle=guard.meter.attach(host.segmenter.model);step=host.step
    def counted(image):
        before=guard.meter.cost.copy();z=step(image)
        if tuple(guard.meter.cost[k]-before[k] for k in OPS)!=(2,0,0,0):raise ValueError('fixed quarter physical contract')
        return z
    host.step=counted
    try:return online(host,rows,c['bindings']['target_root'],p,j['id'],lk,guard,failure,check_lock)
    finally:handle.remove();close()

def worker():
    import torch
    from ..r9_current_first.physical import Meter
    from ..r10_use_write_rl.assets import gpu_policy
    from ..r10_use_write_rl.runtime import classification
    c=config();phase=os.environ['RUN_PHASE'];guard=Guard(c,phase);start=time.time();meter=None;failure=None;result=None;attempt=os.environ.get('RUN_ATTEMPT','0');prior=json.loads(os.environ['RUN_FAILURE']) if os.environ.get('RUN_FAILURE') else None
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    try:
        if phase.startswith('score:'):result=target(c,guard,phase,prior)
        else:
            gpu_policy(c['gpu_assignments'][0])
            with Meter(None,guard.observe) as meter:
                guard.meter=meter;result=preflight(c,guard) if phase=='preflight' else target(c,guard,phase,prior)
                if any(meter.cost[k] for k in OPS[1:]) or phase=='preflight' and meter.cost['model_forwards']!=192:raise ValueError('source/no-update count')
    except BaseException as e:
        failure=dict(reason=str(e),error_type=type(e).__name__,**{'class':'RESOURCE' if isinstance(e,TimeoutError) else classification(e)},evidence=dict(phase=phase,attempt=attempt,config_sha256=sha(c)));traceback.print_exc()
    r=dict(phase=phase,attempt=attempt,status='FAILED' if failure else 'COMPLETE',failure=failure,result=result,cost=meter.cost if meter else dict.fromkeys((*OPS,'gpu_seconds'),0),started=start,ended=time.time(),wall_seconds=time.time()-start,config_sha256=sha(c));save(Path(c['output_root'])/'attempts'/(phase.replace(':','_')+'.'+attempt+'.json'),r);return r

def admit():
    c=config();root=Path(c['output_root']);z=read(root/'ZERO_PARITY.json');old=read(Path(c['previous_root'])/'RESOURCE_LEDGER.json')
    if not z['passed'] or not read(root/'REUSE_QUALIFICATION.json')['passed'] or read(root/'attempts/preflight.0.json')['status']!='COMPLETE':raise ValueError('qualification failed')
    per=max(a['wall_seconds']/1024 for a in old['attempts'] if a['phase'].startswith('online:CV_H025'))
    score=max(a['wall_seconds']/1024 for a in old['attempts'] if a['phase'].startswith('score:'))
    seconds=1.3*(1854*(per+score+.13)+120);eligible=seconds<1800 and seconds<epoch(c['origin']['normal_compute_deadline'])-time.time() and time.time()<epoch(c['origin']['preflight_deadline'])
    c.update(status='FROZEN',admission=dict(target_seconds=seconds,safety_factor=1.3,admitted=eligible));save(root/'RESOLVED_CONFIG.json',c)
    if not eligible:raise RuntimeError('NOT_RUN_BUDGET')
    save(root/'private/authorization.json',dict(experiment_id=ID,code_sha=c['code_sha'],config_sha256=sha(c),qualification_sha256=sha(z),scope='successor two; fixed .25 quarter complement1854 target accesses, historical full hard baselines, no tuning'))
    return c['admission']

def supervise():
    from ..r9_current_first.storage import lease
    from .report import report
    c=config();root=Path(c['output_root']);auth=read(root/'private/authorization.json');normal=epoch(c['origin']['normal_compute_deadline']);hard=epoch(c['origin']['compute_deadline'])
    if auth['config_sha256']!=sha(c) or auth['code_sha']!=c['code_sha'] or time.time()>epoch(c['origin']['preflight_deadline']):raise ValueError('authorization/deadline')
    with lease(root/'supervisor_control',dict(experiment_id=ID)):
        if (root/'execution_started.json').exists():raise ValueError('existing run needs reconciliation')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)));state=dict(experiment_id=ID,status='RUNNING',source='QUALIFIED_REUSE',jobs={j['id']:'NOT_RUN' for j in c['jobs']},recovery_used=False);save(root/'RUN_STATE.json',state);deadline=min(normal,time.time()+1800)
        def run(phase):
            r=launch(c,phase,deadline)
            if r['status']=='FAILED' and r['failure']['class']=='INFRASTRUCTURE' and not state['recovery_used']:
                state['recovery_used']=True;save(root/'RUN_STATE.json',state);r=launch(c,phase,min(hard,deadline+900,time.time()+900),1,r['failure'])
            return r
        try:
            for j in c['jobs']:
                state['jobs'][j['id']]='RUNNING';save(root/'RUN_STATE.json',state);r=run('online:'+j['id'])
                if r['status']=='COMPLETE':r=run('score:'+j['id'])
                state['jobs'][j['id']]=r['status'];save(root/'RUN_STATE.json',state)
                if r['failure'] and r['failure']['class'] in ('ISOLATION','NUMERICAL','IDENTITY_OR_IMPLEMENTATION'):raise RuntimeError(r['failure']['reason'])
        except BaseException as e:state['stop_reason']=str(e);traceback.print_exc()
        for k,v in state['jobs'].items():
            if v in ('RUNNING','NOT_RUN'):state['jobs'][k]='NOT_RUN_STOPPED'
        state.update(status='COMPLETE' if all(v=='COMPLETE' for v in state['jobs'].values()) else 'PARTIAL',ended=time.time());save(root/'RUN_STATE.json',state);report(c,state);return state

def watch():
    c=config();root=Path(c['output_root']);deadline=epoch(c['origin']['absolute_deadline']);p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=dict(os.environ,RUN_MODE='supervise',RUN_CONFIG_SHA=sha(c)),start_new_session=True);save(root/'watchdog.json',dict(pid=os.getpid(),supervisor_pid=p.pid,absolute_deadline=deadline))
    try:return p.wait(timeout=max(.1,deadline-time.time()))
    finally:
        f=root/'active_process.json'
        if f.exists():
            a=read(f);stat=Path(f"/proc/{a['pid']}/stat")
            if a.get('active') and stat.exists() and stat.read_text().split()[21]==a.get('start_ticks'):
                try:os.killpg(a['pgid'],signal.SIGKILL)
                except ProcessLookupError:pass
        if p.poll() is None:terminate_group(p)

def main():
    mode=os.environ['RUN_MODE']
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',jobs=len(prepare()['jobs']))))
    elif mode=='worker':r=worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='preflight':
        c=config();root=Path(c['output_root'])
        with (root/'qualification_started.json').open('x') as f:json.dump(dict(at=time.time(),config_sha256=sha(c)),f)
        r=launch(c,'preflight',min(time.time()+600,epoch(c['origin']['preflight_deadline'])));print(json.dumps(r));sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='admit':print(json.dumps(admit()))
    elif mode=='supervise':print(json.dumps(supervise()))
    elif mode=='watch':sys.exit(watch())
    else:raise ValueError('explicit mode')
