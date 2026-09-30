"""Independent bounded no-training queue, using existing seals and process-group cutoff."""
import json,os,time,traceback,signal,subprocess,sys
from pathlib import Path
from ..r10_12h_core.run import read,save,sha,epoch,Guard,launch,terminate_group,OPS
from .diagnostic import resolved,construct
ID='R10_CARRIER_DECISION_3H_V1'
ARMS=('B_ZERO','B_RESET_1','B_FULL_READOUT_025','B_RESET_READOUT_025')
SOURCE=('C0','B_FULL_1','B_RESET_1','B_FULL_READOUT_025','B_RESET_READOUT_025')

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or os.environ.get('RUN_CONFIG_SHA',sha(c))!=sha(c):raise ValueError('carrier frozen identity')
    return c

def lock(c,name):
    p=dict(experiment_id=ID,config_sha256=sha(c),resolved=resolved(name),checkpoint_sha256=c['bindings']['checkpoint_sha256'],parent=c['parent'])
    return dict(schema=ID+'_LOCK',payload=p,sha256=sha(p))

def check_lock(x):
    if x.get('schema')!=ID+'_LOCK' or x.get('sha256')!=sha(x['payload']) or x['payload']['experiment_id']!=ID:raise ValueError('carrier lock identity')
    r=x['payload']['resolved']
    if r!=resolved(r['diagnostic_condition']):raise ValueError('unregistered intervention')

def prepare():
    from ..r8_ba.streams import rows_sha
    from ..r8_ba.journal import _digest
    from ..r9_current_first.storage import load_torch
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PREVIOUS_ROOT']);oc=read(old/'RESOLVED_CONFIG.json');ledger=read(old/'RESOURCE_LEDGER.json');origin=read(root/'BUDGET_ORIGIN.json')
    if read(old/'RUN_STATE.json')['status']!='COMPLETE' or time.time()>epoch(origin['preflight_deadline']):raise ValueError('previous completion or preflight deadline')
    prior=ledger['prior_package_charged_seconds']+ledger['actual_wall_seconds']
    cap=min(10800,43200-prior)
    if cap<=1800:raise ValueError('insufficient original budget balance')
    origin.update(prior_package_charged_seconds=prior,prior_attribution_actual_wall_seconds=ledger['actual_wall_seconds'],prior_attribution_gpu_worker_seconds=ledger['gpu_worker_seconds'],new_cap_seconds=cap)
    for k,offset in [('preflight_deadline',min(1800,cap/6)),('normal_compute_deadline',max(0,cap-3600)),('compute_deadline',max(0,cap-1800)),('absolute_deadline',cap)]:origin[k]=__import__('datetime').datetime.fromtimestamp(epoch(origin['T0'])+offset,__import__('datetime').timezone.utc).isoformat()
    save(root/'BUDGET_ORIGIN.json',origin)
    (root/'private').mkdir(exist_ok=True);manifests=[]
    for i,m in enumerate(oc['manifests']):
        rows=read(m['path'])
        if len(rows)!=1024 or rows_sha(rows)!=m['sha256'] or sum(x['subset']=='remaining_dev' for x in rows)!=m['principal']:raise ValueError('manifest identity/eligibility')
        p=root/'private'/f'order{i}.json';save(p,rows);manifests.append(dict(m,path=str(p)))
    reuse=[]
    for a,base,jobs in [('NEW_PREVIOUS',old,oc['jobs']),('CORE',Path(oc['previous_root']),oc['old_jobs'])]:
        for j in jobs:
            if j['arm'] not in ('C0_CURRENT_STATS','B_PARENT_FULL','G','N'):continue
            p=base/'target'/j['id'];on=read(p/'online_complete.json');sc=read(p/'score_complete.json');rs=[json.loads(s) for s in (p/'scalars.private.jsonl').read_text().splitlines()];rows=read(manifests[j['order']]['path'])
            if on['identity']['rows_sha256']!=manifests[j['order']]['sha256'] or sc['online_sha256']!=sha(on) or sc['scalar_sha256']!=_digest(p/'scalars.private.jsonl') or sc['principal']!=manifests[j['order']]['principal']:raise ValueError('old sealed result identity')
            if len(rs)!=len(rows) or any(s['visit']!=i+1 or (s['content'],s['domain'],s['subset'])!=(r['group_id'],r['domain'],r['subset']) for i,(s,r) in enumerate(zip(rs,rows))):raise ValueError('old scalar pairing')
            meta=read(p/'checkpoint.0.json');snap=load_torch(p/'checkpoint.0.pt',meta['sha256'])
            if snap['identity']['context_sha256']!=on['identity']['context_sha256']:raise ValueError('old actual deployment')
            reuse.append(dict(job=j,root=str(base),online=on,score=sc,origin=a))
    if len(reuse)!=8:raise ValueError('eight paired baselines required')
    costs=oc['old_profile_prior_cost'].copy()
    for k in OPS:costs[k]+=ledger['operations'][k]
    parent=oc['bindings']['screen24_index']['source_jobs']['B_FULL_20260924']
    c=dict(experiment_id=ID,base_sha=os.environ['BASE_SHA'],code_sha=os.environ['RUN_SHA'],previous_root=str(old),previous_config_sha256=sha(oc),previous_code_sha=oc['code_sha'],output_root=str(root),origin=origin,bindings=oc['bindings'],gpu_assignments=oc['gpu_assignments'],parent=parent,old_profile_prior_cost=costs,manifests=manifests,val_indices=oc['val_indices'],jobs=[dict(id=f'{a}_o{o}',arm=a,order=o,seed=20260924) for a in ARMS for o in (0,1)],source_conditions=SOURCE,reused_results=reuse,patient_dependence=oc['patient_dependence'],recovery_policy=dict(max_jobs=1,max_extra_attempts_per_job=1,reserve_seconds=1800),target_unblind='all new jobs terminal',status='PREFLIGHT')
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
        if tuple(guard.meter.cost[k]-before[k] for k in OPS)!=(2,0,0,0):raise ValueError('B two-forward/no-update contract')
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
    if read(root/'attempts'/f"preflight.{c.get('preflight_attempt',0)}.json")['status']!='COMPLETE' or not z['passed']:raise ValueError('zero parity not qualified')
    native=max(a['wall_seconds'] for a in old['attempts'] if a['phase'].startswith('online:B_PARENT_FULL'))
    scorer=max(a['wall_seconds'] for a in old['attempts'] if a['phase'].startswith('score:'))
    extra=max(0,z['timings']['B_FULL_1']-z['timings']['NATIVE_B'])/32
    target_seconds=1.3*8*(native+1024*extra+scorer+8)
    source_seconds=1.3*(sum(z['timings'][k] for k in SOURCE)*16+120)
    eligible=target_seconds<=3600 and source_seconds<=1800 and source_seconds+target_seconds<epoch(c['origin']['normal_compute_deadline'])-time.time()
    c.update(status='FROZEN',admission=dict(source_seconds=source_seconds,target_seconds=target_seconds,extra_readout_seconds_per_visit=extra,safety_factor=1.3,admitted=eligible,rule='old matched B online and longest independent scorer plus measured diagnostic overhead; full source 512 visits per condition plus loading/I/O'))
    save(root/'RESOLVED_CONFIG.json',c)
    if not eligible:raise RuntimeError('NOT_RUN_BUDGET: complete fixed matrix does not fit')
    save(root/'private/authorization.json',dict(experiment_id=ID,code_sha=c['code_sha'],config_sha256=sha(c),zero_parity_sha256=sha(z),scope='no training; five fixed source conditions; eight fixed target trajectories; no push; no monitoring'))
    return c['admission']

def supervise():
    from ..r9_current_first.storage import lease
    from .report import report
    c=config();root=Path(c['output_root']);auth=read(root/'private/authorization.json');normal=epoch(c['origin']['normal_compute_deadline']);hard=epoch(c['origin']['compute_deadline'])
    if auth['config_sha256']!=sha(c) or auth['code_sha']!=c['code_sha'] or time.time()>epoch(c['origin']['preflight_deadline']):raise ValueError('authorization/deadline')
    with lease(root/'supervisor',dict(experiment_id=ID,config_sha256=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('existing run requires ledger reconciliation')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)))
        state=dict(experiment_id=ID,status='RUNNING',source='NOT_RUN',jobs={j['id']:'NOT_RUN' for j in c['jobs']},recovery_used=False,engineering_repairs=c.get('engineering_repairs',[]));save(root/'RUN_STATE.json',state)
        def run(phase,deadline):
            r=launch(c,phase,min(normal,deadline),c.get('source_attempt',0) if phase=='source' else 0)
            if r['status']=='FAILED' and r['failure']['class']=='INFRASTRUCTURE' and phase!='source' and not state['recovery_used']:
                state['recovery_used']=True;save(root/'RUN_STATE.json',state);r=launch(c,phase,min(hard,deadline+1800,time.time()+1800),1,r['failure'])
            return r
        try:
            state['source']='RUNNING';save(root/'RUN_STATE.json',state);r=run('source',c.get('source_stage_deadline',time.time()+1800));state['source']=r['status'];save(root/'RUN_STATE.json',state)
            if r['status']!='COMPLETE':raise RuntimeError('paired source comparison failed')
            deadline=time.time()+3600
            for j in c['jobs']:
                state['jobs'][j['id']]='RUNNING';save(root/'RUN_STATE.json',state);r=run('online:'+j['id'],deadline)
                if r['status']=='COMPLETE':
                    if j['arm']=='B_ZERO':
                        old=next(x for x in c['reused_results'] if x['job']['arm']=='C0_CURRENT_STATS' and x['job']['order']==j['order'])
                        zero=read(root/'target'/j['id']/'online_complete.json')
                        if (zero['prediction_sha256'],zero['prediction_bytes'])!=(old['online']['prediction_sha256'],old['online']['prediction_bytes']):raise ValueError('B_ZERO/C0 compatible probability seals differ; stop carrier interpretation')
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
