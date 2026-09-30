"""Independent eight-hour orchestration; reuse existing kernels, seals and watchdogs."""
import copy,json,os,shutil,signal,subprocess,sys,time,traceback
from pathlib import Path
from ..r10_12h_core.run import read,save,sha,epoch,Guard,launch,terminate_group,OPS,LIMITS

ID='R10_ATTRIBUTION_SUP_8H_V1'
A_ARMS=('C0_CURRENT_STATS','B_PARENT_FULL','WARM_CONST_HALF','R10_RESET_ALL','R10_FORCE_WRITE')
B_ARMS=('SUP_RET','SUP_RET_CONST_HALF','SUP_STATIC')
SEED=20260924

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or os.environ.get('RUN_CONFIG_SHA',sha(c))!=sha(c):raise ValueError('attribution frozen config identity')
    return c

def prepare():
    from ..r8_ba.streams import rows_sha
    from ..r8_ba.journal import _digest
    from ..r9_current_first.storage import load_torch
    from ..r10_12h_core.run import check_lock as check_old_lock
    from .audit import CONTEXTS,actor_at
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PREVIOUS_ROOT']);oc=read(old/'RESOLVED_CONFIG.json');origin=read(root/'BUDGET_ORIGIN.json');ledger=read(old/'RESOURCE_LEDGER.json')
    if time.time()>epoch(origin['preflight_deadline']):raise TimeoutError('PREFLIGHT_INCOMPLETE')
    if read(old/'RUN_STATE.json')['status']!='COMPLETE':raise ValueError('old experiment is not complete')
    lock=read(old/'source_lock.json');check_old_lock(lock)
    if lock['payload']['config_sha256']!=sha(oc):raise ValueError('old frozen config changed')
    endpoints={k:actor_at(old,k)[1] for k in ('WARM','POST')}
    (root/'source').mkdir(exist_ok=True);(root/'private').mkdir(exist_ok=True)
    for name in ('WARM','POST'):
        p=root/'source'/name
        if not p.exists():p.symlink_to(old/'source'/name,target_is_directory=True)
        if p.resolve()!=(old/'source'/name).resolve():raise ValueError('read-only reused endpoint link')
    manifests=[];reuse=[]
    for i,m in enumerate(oc['manifests']):
        rows=read(m['path']);counts=sum(r['subset']=='remaining_dev' for r in rows)
        if len(rows)!=1024 or rows_sha(rows)!=m['sha256'] or counts!=m['principal']:raise ValueError('old manifest coverage/identity')
        p=root/'private'/f'order{i}.json';save(p,rows);manifests.append(dict(m,path=str(p)))
    for job in oc['jobs']:
        p=old/'target'/job['id'];online=read(p/'online_complete.json');score=read(p/'score_complete.json');rows=read(oc['manifests'][job['order']]['path'])
        if online['identity']['rows_sha256']!=rows_sha(rows) or online['visits']!=len(rows) or score['online_sha256']!=sha(online) or score['source_lock_sha256']!=lock['sha256'] or score['scalar_sha256']!=_digest(p/'scalars.private.jsonl'):raise ValueError('old output/score seal mismatch')
        scalars=[json.loads(s) for s in (p/'scalars.private.jsonl').read_text().splitlines()]
        if len(scalars)!=len(rows) or score['principal']!=sum(r['subset']=='remaining_dev' for r in rows):raise ValueError('old score coverage')
        for i,(s,r) in enumerate(zip(scalars,rows)):
            if s['visit']!=i+1 or (s['content'],s['domain'],s['subset'])!=(r['group_id'],r['domain'],r['subset']):raise ValueError('old scalar order/eligibility mismatch')
        if job['arm'].startswith('GR_RET_EMA'):
            meta=read(p/'checkpoint.0.json');snapshot=load_torch(p/'checkpoint.0.pt',meta['sha256']);host=snapshot['host'];resolved=host['identity']['resolved']
            if host['identity']['code_sha']!=oc['code_sha'] or resolved['artifact']!=endpoints['POST']['artifact'] or resolved['source_job']!='POST' or host['diagnostic']!=('CONST_HALF' if job['arm'].endswith('CONST_HALF') else None):raise ValueError('old actual deployed actor identity')
            if snapshot['identity']['context_sha256']!=online['identity']['context_sha256']:raise ValueError('old deployment context')
        reuse.append(dict(job=job,online=online,score=score,status='VERIFIED_REUSE',original_code_sha=oc['code_sha']))
    prior=oc['old_profile_prior_cost'].copy()
    for k in OPS:prior[k]+=ledger['operations'][k]
    c=dict(experiment_id=ID,base_sha=os.environ['BASE_SHA'],code_sha=os.environ['RUN_SHA'],previous_root=str(old),previous_config_sha256=sha(oc),previous_code_sha=oc['code_sha'],bindings=oc['bindings'],gpu_assignments=oc['gpu_assignments'],output_root=str(root),origin=origin,old_profile_prior_cost=prior,manifests=manifests,old_jobs=oc['jobs'],reused_results=reuse,endpoints=endpoints,post=1024,val_indices=oc['val_indices'],counterfactual_contexts=CONTEXTS,scheduler=oc['scheduler'],candidate_training=['SUP_RET','SUP_STATIC'],training=[],jobs=[dict(id=f'{a}_o{o}',arm=a,order=o,seed=SEED) for a in A_ARMS+B_ARMS for o in (0,1)],recovery_policy=dict(max_jobs=1,max_extra_attempts_per_job=1,reserve_seconds=1800),target_unblind='only after all planned jobs terminal',patient_dependence=oc['patient_dependence'],references=dict(old_profile=os.environ['OLD_PROFILE']),status='PROVISIONAL_PREFLIGHT')
    save(root/'RESOLVED_CONFIG.json',c);save(root/'REUSE_VERIFICATION.json',dict(previous_code_sha=oc['code_sha'],previous_config_sha256=sha(oc),verified_trajectories=len(reuse),same_backbone_and_B=True,same_preprocess_and_score_kernels=True,manifest_sha256=[m['sha256'] for m in manifests],post_artifact=endpoints['POST']['artifact'],warm_artifact=endpoints['WARM']['artifact']));return c

def resolve_job(c,job):
    a=job['arm'];root=Path(c['output_root']);name=None;diagnostic=None;method='GR_RET_EMA';arm='GR_RET_EMA'
    if a in ('C0_CURRENT_STATS','B_PARENT_FULL'):arm='C0' if a=='C0_CURRENT_STATS' else 'B_CARRIER_FULL';method=arm
    elif a=='WARM_CONST_HALF':name='WARM';diagnostic='CONST_HALF'
    elif a in ('R10_RESET_ALL','R10_FORCE_WRITE'):name='POST';diagnostic={'R10_RESET_ALL':'RESET_ALL','R10_FORCE_WRITE':'FORCE_WRITE'}[a]
    elif a in B_ARMS:name='SUP_STATIC' if a=='SUP_STATIC' else 'SUP_RET';method=name;arm=name;diagnostic='CONST_HALF' if a.endswith('CONST_HALF') else None
    else:raise ValueError('unregistered attribution arm')
    artifact=read(root/'source'/name/'complete.json')['artifact'] if name else None
    return dict(arm=arm,method=method,seed=SEED,source_job=name,artifact=artifact,diagnostic=diagnostic)

def source_lock(c,job):
    resolved=resolve_job(c,job);payload=dict(experiment_id=ID,config_sha256=sha(c),resolved=resolved,previous_source_sha256=sha(c['endpoints']))
    return dict(schema=ID+'_SOURCE_LOCK',payload=payload,sha256=sha(payload))

def check_lock(lock):
    if lock.get('schema')!=ID+'_SOURCE_LOCK' or lock.get('sha256')!=sha(lock['payload']) or lock['payload']['experiment_id']!=ID:raise ValueError('attribution source lock')

def dispatch_source(c,guard,phase,failure):
    import torch
    from ..r10_use_write_rl.assets import open_source
    from ..r10_use_write_rl.source import Source
    from ..r10_use_write_rl.learning import rng_state,restore_rng
    from ..r10_use_write_rl.evaluation import episode
    from ..r9_current_first.storage import PhaseJournal,load_torch,write_torch
    from .audit import preflight,counterfactual,AuditTrainer,actor_at
    with open_source(c['bindings'],c['gpu_assignments'][0],guard) as (data,seg,oracles,controller):
        handle=guard.meter.attach(seg.model);source=Source(data,oracles,seg,controller,sha(c['bindings']))
        try:
            if phase=='preflight':return preflight(c,source,guard)
            if phase=='counterfactual':return counterfactual(c,source,guard)
            method=phase.split(':',1)[1]
            if method not in c['training']:raise ValueError('unadmitted training branch')
            warm,_=actor_at(c['previous_root'],'WARM');actor=copy.deepcopy(warm)
            binding=dict(experiment_id=ID,config_sha256=sha(c),method=method,total=1024)
            t=AuditTrainer(actor,controller,source,SEED,method,binding,warm,guard,total=1024)
            path=Path(c['output_root'])/'source'/method;path.mkdir(parents=True,exist_ok=True);journal=PhaseJournal(path,'fit',binding)
            if journal.completed():
                meta=read(journal.root/'latest.json');t.restore(load_torch(journal.root/f"checkpoint.{meta['slot']}.pt",meta['sha256'])['snapshot'])
            else:
                journal.begin(t,failure)
                while t.steps<1024:
                    guard();journal.append(t.step())
                    if t.steps%50==0 or t.steps==1024:journal.checkpoint(t)
                journal.complete(dict(steps=t.steps))
            artifact=dict(file='actor.pt',sha256=write_torch(path/'actor.pt',dict(actor=actor.state_dict(),method=method,steps=1024,config_sha256=sha(c))))
            saved=rng_state();a=copy.deepcopy(actor).eval();a.requires_grad_(False)
            try:rows=[episode(source,a,SEED,i,static=method=='SUP_STATIC',guard=guard) for i in c['val_indices']]
            finally:restore_rng(saved)
            save(path/'validation.json',dict(indices=c['val_indices'],rows=rows,selection=False));result=dict(schema=ID+'_SOURCE_COMPLETE',config_sha256=sha(c),method=method,steps=1024,optimizer_updates=2048,artifact=artifact,warm_artifact=c['endpoints']['WARM']['artifact'])
            save(path/'complete.json',result);return result
        finally:handle.remove()

def dispatch_target(c,guard,phase,failure):
    from ..r10_use_write_rl.factory import construct
    from ..r10_use_write_rl.target import online,score,retire_probabilities
    from ..r8_ba.streams import rows_sha
    scoring=phase.startswith('score:');job=next(j for j in c['jobs'] if j['id']==phase.split(':',1)[1]);root=Path(c['output_root']);p=root/'target'/job['id'];lock=source_lock(c,job);manifest=c['manifests'][job['order']];rows=read(manifest['path'])
    if rows_sha(rows)!=manifest['sha256']:raise ValueError('frozen arrival manifest')
    if scoring:
        seal=read(p/'online_complete.json');result=score(rows,c['bindings']['target_root'],p,job['id'],seal['identity']['context_sha256'],lock,guard,failure,check_lock);retire_probabilities(p);return result
    resolved=resolve_job(c,job);host,close=construct(resolved,c,root,lock);handle=guard.meter.attach(host.segmenter.model);step=host.step
    def checked(image):
        before=guard.meter.cost.copy();result=step(image);counts=tuple(guard.meter.cost[k]-before[k] for k in OPS)
        if counts!=((1,0,0,0) if job['arm']=='C0_CURRENT_STATS' else (2,0,0,0)):raise ValueError('target physical operator contract')
        return result
    host.step=checked
    try:return online(host,rows,c['bindings']['target_root'],p,job['id'],lock,guard,failure,check_lock)
    finally:handle.remove();close()

def worker():
    import torch
    from ..r9_current_first.physical import Meter
    from ..r10_use_write_rl.assets import gpu_policy
    from ..r10_use_write_rl.runtime import classification
    c=config();phase=os.environ['RUN_PHASE'];root=Path(c['output_root']);guard=Guard(c,phase);start=time.time();meter=None;failure=None;result=None;attempt=os.environ.get('RUN_ATTEMPT','0');old_failure=json.loads(os.environ['RUN_FAILURE']) if os.environ.get('RUN_FAILURE') else None
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    try:
        if phase.startswith('score:'):result=dispatch_target(c,guard,phase,old_failure)
        else:
            gpu_policy(c['gpu_assignments'][0])
            with Meter(None,guard.observe) as meter:
                guard.meter=meter
                result=dispatch_target(c,guard,phase,old_failure) if phase.startswith('online:') else dispatch_source(c,guard,phase,old_failure)
    except BaseException as e:failure=dict(reason=str(e),error_type=type(e).__name__,**{'class':'RESOURCE' if isinstance(e,TimeoutError) else classification(e)},evidence=dict(config_sha256=sha(c),phase=phase,attempt=attempt));traceback.print_exc()
    record=dict(phase=phase,attempt=attempt,status='FAILED' if failure else 'COMPLETE',failure=failure,result=result,cost=meter.cost if meter else dict.fromkeys((*OPS,'gpu_seconds'),0),started=start,ended=time.time(),wall_seconds=time.time()-start,config_sha256=sha(c))
    save(root/'attempts'/(phase.replace(':','_')+'.'+attempt+'.json'),record);return record

def admit():
    c=config();root=Path(c['output_root']);audit=read(root/'WRITER_AUDIT.private.json');oldp=read(c['references']['old_profile']);oldledger=read(Path(c['previous_root'])/'RESOURCE_LEDGER.json')
    if read(root/'attempts/preflight.0.json')['status']!='COMPLETE':raise ValueError('preflight did not complete')
    max_online=max(a['wall_seconds'] for a in oldledger['attempts'] if a['phase'].startswith('online:GR_RET_EMA'))
    max_score=max(a['wall_seconds'] for a in oldledger['attempts'] if a['phase'].startswith('score:'))
    target_each=1.3*(max_online+max_score+8)
    # Conservative old source validation rate already contains the 1.3 max factor.
    val=16*oldp['measurements']['validation_episode']['gpu_seconds']
    diagnostic=1.3*(32*49*oldp['measurements']['online/POLICY']['gpu_seconds']+60)
    stage_a=10*target_each+diagnostic
    costs={m:max(audit['profiles'][m]['conservative_seconds_per_round'],oldp['measurements']['train/'+m]['gpu_seconds'])*1024+val+120 for m in ('SUP_RET','SUP_STATIC')}
    selected=[];candidates=[]
    for method,targets in (('SUP_RET',4),('SUP_STATIC',2)):
        proposed=selected+[method];train=sum(costs[m] for m in proposed);n=sum(4 if m=='SUP_RET' else 2 for m in proposed)
        ok=train<=4*3600 and n*target_each<=3600 and stage_a<=1.5*3600 and stage_a+train+n*target_each<epoch(c['origin']['normal_compute_deadline'])-time.time()
        candidates.append(dict(method=method,source_seconds=costs[method],target_seconds=targets*target_each,admitted=ok))
        if ok:selected.append(method)
    if stage_a>1.5*3600:raise RuntimeError('OVER_CAP: fixed attribution stage')
    c.update(status='FROZEN',training=selected,admission=dict(stage_a_seconds=stage_a,source_counterfactual_seconds=diagnostic,per_target_including_score_seconds=target_each,source_training_seconds=sum(costs[m] for m in selected),stage_b_target_seconds=sum(4 if m=='SUP_RET' else 2 for m in selected)*target_each,candidates=candidates,safety_factor=1.3,source_cost_rule='max(old complete-block conservative rate,new 32-round rate*1.3), plus 16 validation and 120s startup/checkpoint/I/O'))
    save(root/'RESOLVED_CONFIG.json',c);save(root/'private/authorization.json',dict(experiment_id=ID,config_sha256=sha(c),code_sha=c['code_sha'],scope='five controls, fixed source counterfactual, admitted existing SUP branches only; no push',at=time.time()));return c['admission']

def supervise():
    from ..r9_current_first.storage import lease
    c=config();root=Path(c['output_root']);auth=read(root/'private/authorization.json');normal=epoch(c['origin']['normal_compute_deadline']);hard=epoch(c['origin']['compute_deadline'])
    if auth['config_sha256']!=sha(c) or auth['code_sha']!=c['code_sha'] or time.time()>epoch(c['origin']['preflight_deadline']):raise ValueError('authorization/preflight deadline')
    with lease(root/'supervisor',dict(experiment_id=ID,config_sha256=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('existing run requires reconciliation; never reset T0')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)))
        state=dict(experiment_id=ID,status='RUNNING',training={m:'NOT_RUN' if m in c['training'] else 'NOT_RUN_BUDGET' for m in c['candidate_training']},counterfactual='NOT_RUN',jobs={j['id']:'NOT_RUN' for j in c['jobs']},recovery_used=False)
        for j in c['jobs']:
            if j['arm'] in B_ARMS and ('SUP_STATIC' if j['arm']=='SUP_STATIC' else 'SUP_RET') not in c['training']:state['jobs'][j['id']]='NOT_RUN_BUDGET'
        save(root/'RUN_STATE.json',state)
        def run(phase,deadline):
            r=launch(c,phase,min(normal,deadline))
            if r['status']=='FAILED' and r['failure']['class']=='INFRASTRUCTURE' and phase!='counterfactual' and not state['recovery_used']:
                state['recovery_used']=True;save(root/'RUN_STATE.json',state);start=time.time();r=launch(c,phase,min(hard,deadline+1800,start+1800),1,r['failure'])
            return r
        def trajectory(job,deadline):
            r=run('online:'+job['id'],deadline)
            if r['status']=='COMPLETE':r=run('score:'+job['id'],deadline)
            state['jobs'][job['id']]=r['status'];save(root/'RUN_STATE.json',state)
            if r['failure'] and r['failure']['class'] in ('ISOLATION','NUMERICAL','IDENTITY_OR_IMPLEMENTATION'):raise RuntimeError('stop at deterministic/scientific failure: '+r['failure']['reason'])
        try:
            deadline=time.time()+5400
            r=run('counterfactual',deadline);state['counterfactual']=r['status'];save(root/'RUN_STATE.json',state)
            if r['status']!='COMPLETE':raise RuntimeError('source counterfactual failed; retain evidence')
            for job in c['jobs']:
                if job['arm'] in A_ARMS:trajectory(job,deadline)
            deadline=time.time()+4*3600
            for method in c['training']:
                state['training'][method]='RUNNING';save(root/'RUN_STATE.json',state);r=run('train:'+method,deadline);state['training'][method]=r['status'];save(root/'RUN_STATE.json',state)
                if r['status']!='COMPLETE':raise RuntimeError('source training failed: '+method)
            deadline=time.time()+3600
            for job in c['jobs']:
                method='SUP_STATIC' if job['arm']=='SUP_STATIC' else 'SUP_RET'
                if job['arm'] in B_ARMS and method in c['training']:trajectory(job,deadline)
        except BaseException as e:state['stop_reason']=str(e);traceback.print_exc()
        for group in ('jobs','training'):
            for key,value in state[group].items():
                if value in ('RUNNING','NOT_RUN'):state[group][key]='NOT_RUN_STOPPED'
        state['status']='COMPLETE' if all(v=='COMPLETE' for v in state['jobs'].values()) and state['counterfactual']=='COMPLETE' and all(v=='COMPLETE' for v in state['training'].values()) else 'PARTIAL';state['ended']=time.time();save(root/'RUN_STATE.json',state)
        from .report import report
        report(c,state);return state

def watch():
    c=config();root=Path(c['output_root']);deadline=epoch(c['origin']['absolute_deadline']);env=dict(os.environ,RUN_MODE='supervise',RUN_CONFIG_SHA=sha(c))
    p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,start_new_session=True);save(root/'watchdog.json',dict(pid=os.getpid(),supervisor_pid=p.pid,absolute_deadline=deadline))
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
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',reuse=len(prepare()['reused_results']))))
    elif mode=='preflight':
        c=config();r=launch(c,'preflight',epoch(c['origin']['preflight_deadline']));print(json.dumps(r));sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='worker':r=worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='admit':print(json.dumps(admit()))
    elif mode=='supervise':print(json.dumps(supervise()))
    elif mode=='watch':sys.exit(watch())
    else:raise ValueError('explicit mode required')
