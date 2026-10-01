"""Bounded fixed current-view pilot; shared physical counting and sealed target IO."""
import json,os,time,traceback,signal,subprocess,sys
from pathlib import Path
from ..r10_12h_core.run import read,save,sha,epoch,Guard,launch,terminate_group,OPS
from .view import construct,VIEWS,AGGREGATION
ID='R11_CURRENT_VIEW_V1'
ARMS=SOURCE=('CV_H2','CV_FLIP4')
def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or os.environ.get('RUN_CONFIG_SHA',sha(c))!=sha(c):raise ValueError('current view frozen identity')
    return c

def lock(c,name):
    if name not in VIEWS:raise ValueError('unregistered view')
    p=dict(experiment_id=ID,config_sha256=sha(c),condition=name,views=[list(v) for v in VIEWS[name]],aggregation=AGGREGATION,checkpoint_sha256=c['bindings']['checkpoint_sha256'])
    return dict(schema=ID+'_LOCK',payload=p,sha256=sha(p))
def check_lock(x):
    p=x.get('payload',{});name=p.get('condition')
    if x.get('schema')!=ID+'_LOCK' or x.get('sha256')!=sha(p) or p.get('experiment_id')!=ID or name not in VIEWS or p.get('views')!=[list(v) for v in VIEWS[name]] or p.get('aggregation')!=AGGREGATION:raise ValueError('current view lock identity')
def supervisor_lease(c):
    from ..r9_current_first.storage import lease
    # Queue exclusion is stable across repairs; authorization checks the exact config.
    return lease(Path(c['output_root'])/'supervisor_control',dict(experiment_id=ID))

def prepare():
    from ..r8_ba.streams import rows_sha
    from ..r8_ba.journal import _digest
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PREVIOUS_ROOT']);oc=read(old/'RESOLVED_CONFIG.json');ledger=read(old/'RESOURCE_LEDGER.json');origin=read(root/'BUDGET_ORIGIN.json')
    if read(old/'RUN_STATE.json')['status']!='COMPLETE' or time.time()>epoch(origin['preflight_deadline']):raise ValueError('previous completion/preflight deadline')
    prior=ledger['prior_package_charged_seconds']+ledger['actual_wall_seconds']
    if abs(prior-origin['prior_package_charged_seconds'])>.01:raise ValueError('budget lineage drift')
    manifests=[]
    for i,m in enumerate(oc['manifests']):
        rows=read(m['path'])
        if len(rows)!=1024 or rows_sha(rows)!=m['sha256'] or sum(x['subset']=='remaining_dev' for x in rows)!=888:raise ValueError('manifest identity')
        path=root/'private'/f'order{i}.json';save(path,rows);manifests.append(dict(m,path=str(path)))
    reuse=[x for x in oc['reused_results'] if x['job']['arm'] in ('C0_CURRENT_STATS','G')]
    if len(reuse)!=4:raise ValueError('four exact C0/G baselines required')
    for x in reuse:
        j=x['job'];p=Path(x['root'])/'target'/j['id'];on=read(p/'online_complete.json');sc=read(p/'score_complete.json')
        if on!=x['online'] or sc!=x['score'] or sc['online_sha256']!=sha(on) or sc['scalar_sha256']!=_digest(p/'scalars.private.jsonl') or on['identity']['rows_sha256']!=manifests[j['order']]['sha256'] or sc['principal']!=888:raise ValueError('reused seals mismatch')
    costs=oc['old_profile_prior_cost'].copy()
    for k in OPS:costs[k]+=ledger['operations'][k]
    src=read(old/'SOURCE_COMPARISON.json')
    if src['status']!='COMPLETE' or src['indices']!=oc['val_indices']:raise ValueError('source baseline incomplete')
    c=dict(experiment_id=ID,base_sha=os.environ['BASE_SHA'],code_sha=os.environ['RUN_SHA'],previous_root=str(old),output_root=str(root),origin=origin,bindings=oc['bindings'],gpu_assignments=oc['gpu_assignments'],old_profile_prior_cost=costs,manifests=manifests,val_indices=oc['val_indices'],jobs=[dict(id=f'{a}_o{o}',arm=a,order=o,seed=20260924) for a in ARMS for o in (0,1)],source_conditions=SOURCE,reused_results=reuse,reused_source_C0=[r for r in src['rows'] if r['condition']=='C0'],patient_dependence=oc['patient_dependence'],status='PREFLIGHT',aggregation=AGGREGATION)
    save(root/'RESOLVED_CONFIG.json',c);save(root/'RESOURCE_LEDGER.json',origin);return c
def target(c,guard,phase,failure):
    from ..r10_use_write_rl.target import online,score,retire_probabilities
    from ..r8_ba.streams import rows_sha
    j=next(j for j in c['jobs'] if j['id']==phase.split(':',1)[1]);p=Path(c['output_root'])/'target'/j['id'];rows=read(c['manifests'][j['order']]['path']);lk=lock(c,j['arm'])
    if rows_sha(rows)!=c['manifests'][j['order']]['sha256']:raise ValueError('frozen arrivals')
    if phase.startswith('score:'):
        on=read(p/'online_complete.json');r=score(rows,c['bindings']['target_root'],p,j['id'],on['identity']['context_sha256'],lk,guard,failure,check_lock);retire_probabilities(p);return r
    host,close=construct(j['arm'],c,lk);handle=guard.meter.attach(host.segmenter.model);step=host.step
    def checked(image):
        before=guard.meter.cost.copy();z,t=step(image)
        if tuple(guard.meter.cost[k]-before[k] for k in OPS)!=(len(VIEWS[j['arm']]),0,0,0):raise ValueError('view forward/no-update contract')
        return z,t
    host.step=checked
    try:return online(host,rows,c['bindings']['target_root'],p,j['id'],lk,guard,failure,check_lock)
    finally:handle.remove();close()

def worker():
    import torch
    from ..r9_current_first.physical import Meter
    from ..r10_use_write_rl.assets import gpu_policy
    from ..r10_use_write_rl.runtime import classification
    from .source import preflight,comparison
    c=config();phase=os.environ['RUN_PHASE'];guard=Guard(c,phase);start=time.time();meter=None;failure=None;result=None;attempt=os.environ.get('RUN_ATTEMPT','0');old_failure=json.loads(os.environ['RUN_FAILURE']) if os.environ.get('RUN_FAILURE') else None
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    try:
        if phase.startswith('score:'):result=target(c,guard,phase,old_failure)
        else:
            gpu_policy(c['gpu_assignments'][0])
            with Meter(None,guard.observe) as meter:
                guard.meter=meter
                if phase=='preflight':result=preflight(c,guard)
                elif phase=='source':result=comparison(c,guard)
                else:result=target(c,guard,phase,old_failure)
                if meter.cost['backward_calls'] or meter.cost['optimizer_steps'] or meter.cost['vjp_calls']:raise ValueError('no-training contract')
    except BaseException as e:
        failure=dict(reason=str(e),error_type=type(e).__name__,**{'class':'RESOURCE' if isinstance(e,TimeoutError) else classification(e)},evidence=dict(phase=phase,attempt=attempt,config_sha256=sha(c)));traceback.print_exc()
    record=dict(phase=phase,attempt=attempt,status='FAILED' if failure else 'COMPLETE',failure=failure,result=result,cost=meter.cost if meter else dict.fromkeys((*OPS,'gpu_seconds'),0),started=start,ended=time.time(),wall_seconds=time.time()-start,config_sha256=sha(c))
    save(Path(c['output_root'])/'attempts'/(phase.replace(':','_')+'.'+attempt+'.json'),record);return record

def admit():
    c=config();root=Path(c['output_root']);z=read(root/'ZERO_PARITY.json');old=read(Path(c['previous_root'])/'RESOURCE_LEDGER.json')
    if read(root/'attempts/preflight.0.json')['status']!='COMPLETE' or not z['passed']:raise ValueError('identity parity incomplete')
    scorer=max(a['wall_seconds'] for a in old['attempts'] if a['phase'].startswith('score:'))
    # Measured fixed-view forward time plus preserved conservative read/fsync/loading margin.
    source_seconds=1.3*(sum(z['timings'][k] for k in SOURCE)*16+120)
    target_seconds=1.3*(sum(z['timings'][k] for k in ARMS)*64+4096*.13+4*(scorer+30))
    eligible=source_seconds<=900 and target_seconds<=1800 and source_seconds+target_seconds<epoch(c['origin']['normal_compute_deadline'])-time.time()
    c.update(status='FROZEN',admission=dict(source_seconds=source_seconds,target_seconds=target_seconds,safety_factor=1.3,admitted=eligible))
    save(root/'RESOLVED_CONFIG.json',c)
    if not eligible:raise RuntimeError('NOT_RUN_BUDGET: complete fixed matrix does not fit')
    save(root/'private/authorization.json',dict(experiment_id=ID,code_sha=c['code_sha'],config_sha256=sha(c),zero_parity_sha256=sha(z),scope='two fixed views; four target jobs; no tuning or monitoring; publish anonymous completed results to project GitHub'))
    return c['admission']
def supervise():
    from .report import report
    c=config();root=Path(c['output_root']);auth=read(root/'private/authorization.json');normal=epoch(c['origin']['normal_compute_deadline']);hard=epoch(c['origin']['compute_deadline'])
    if auth['config_sha256']!=sha(c) or auth['code_sha']!=c['code_sha'] or time.time()>epoch(c['origin']['preflight_deadline']):raise ValueError('authorization/deadline')
    with supervisor_lease(c):
        if (root/'execution_started.json').exists():raise ValueError('existing run requires ledger reconciliation')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)))
        state=dict(experiment_id=ID,status='RUNNING',source='NOT_RUN',jobs={j['id']:'NOT_RUN' for j in c['jobs']},recovery_used=False,engineering_repairs=[]);save(root/'RUN_STATE.json',state)
        def run(phase,deadline):
            r=launch(c,phase,min(normal,deadline),c.get('source_attempt',0) if phase=='source' else 0)
            if r['status']=='FAILED' and r['failure']['class']=='INFRASTRUCTURE' and phase!='source' and not state['recovery_used']:
                state['recovery_used']=True;save(root/'RUN_STATE.json',state);r=launch(c,phase,min(hard,deadline+900,time.time()+900),1,r['failure'])
            return r
        try:
            state['source']='RUNNING';save(root/'RUN_STATE.json',state);r=run('source',time.time()+900);state['source']=r['status'];save(root/'RUN_STATE.json',state)
            if r['status']!='COMPLETE':raise RuntimeError('paired source comparison failed')
            deadline=time.time()+1800
            for j in c['jobs']:
                state['jobs'][j['id']]='RUNNING';save(root/'RUN_STATE.json',state);r=run('online:'+j['id'],deadline)
                if r['status']=='COMPLETE':
                    r=run('score:'+j['id'],deadline)
                state['jobs'][j['id']]=r['status'];save(root/'RUN_STATE.json',state)
                if r['failure'] and r['failure']['class'] in ('ISOLATION','NUMERICAL','IDENTITY_OR_IMPLEMENTATION'):raise RuntimeError(r['failure']['reason'])
        except BaseException as e:state['stop_reason']=str(e);traceback.print_exc()
        for k,v in state['jobs'].items():
            if v in ('NOT_RUN','RUNNING'):state['jobs'][k]='NOT_RUN_STOPPED'
        state['status']='COMPLETE' if state['source']=='COMPLETE' and all(v=='COMPLETE' for v in state['jobs'].values()) else 'PARTIAL';state['ended']=time.time();save(root/'RUN_STATE.json',state);report(c,state);return state

def watch():
    c=config();root=Path(c['output_root']);deadline=epoch(c['origin']['absolute_deadline'])
    env=dict(os.environ,RUN_MODE='supervise',RUN_CONFIG_SHA=sha(c))
    p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,start_new_session=True)
    save(root/'watchdog.json',dict(pid=os.getpid(),supervisor_pid=p.pid,absolute_deadline=deadline))
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
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',reused=len(prepare()['reused_results']))))
    elif mode=='worker':r=worker();__import__('sys').exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='preflight':c=config();r=launch(c,'preflight',epoch(c['origin']['preflight_deadline']),c.get('preflight_attempt',0));print(json.dumps(r));__import__('sys').exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='admit':print(json.dumps(admit()))
    elif mode=='supervise':print(json.dumps(supervise()))
    elif mode=='watch':__import__('sys').exit(watch())
    else:raise ValueError('explicit mode required')
