"""User-amended fast screen: existing full actor plus GraTa, no source refit."""
import concurrent.futures,json,os,time,subprocess,signal,sys
from pathlib import Path
from . import run as original,score as scoring
from .method import SEED,ARMS as ORIGINAL_ARMS
from ..r10_12h_core.run import read,save,sha
from ..r8_ba.journal import _digest
from ..r9_current_first.storage import lease

ID='R18_QUICK_EXISTING_RL_GRATA_V1'
ARMS=('RL_GRATA','RL_ORIGINAL','FIXED_GRATA')
base=original.base

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or sha(c)!=os.environ['RUN_CONFIG_SHA']:raise ValueError('quick frozen identity')
    return c

def prepare():
    root=Path(os.environ['RUN_ROOT']);prior=read(root/'private/parent-config.json');origin=read(root/'BUDGET_ORIGIN.json')
    c=dict(prior,experiment_id=ID,code_sha=os.environ['RUN_SHA'],output_root=str(root),origin=origin,
           parent_stage_root=prior['output_root'],parent_stage_config_sha256=sha(prior),target_arms=list(ARMS),conditions=['C0','G',*ARMS],
           source=dict(retraining=False,policy='original sealed R10 POST GR_RET_EMA',warm_steps=1000,post_steps=1024,selection='unchanged historical endpoint'),
           budget_rule='latest user three-hour total wall budget; no retraining; primary pair first; full trajectories only')
    save(root/'RESOLVED_CONFIG.json',c);save(root/'RUN_STATE.json',dict(status='PREPARING',jobs={},target_scores_embargoed=True));return c

def freeze(c):
    root=Path(c['output_root']);prior=Path(c['parent_stage_root']);p=read(prior/'SOURCE_PRECHECK.json')
    if not p['passed'] or read(prior/'RUN_STATE.json')['target_accesses']!=0:raise ValueError('mechanical qualification/source transition')
    actor=Path(c['original_source_root'])/'source/POST'/c['old_actor']['file']
    if _digest(actor)!=c['old_actor']['sha256']:raise ValueError('unchanged source actor seal')
    old=Path(c['parent_root']);receipt=read(old/'SCORER_RECEIPT.json');data=old/'score/all-scalars.private.jsonl'
    if receipt['status']!='COMPLETE' or _digest(data)!=receipt['scalar_sha256']:raise ValueError('historical scalar seal')
    values={}
    for line in data.open():
        row=json.loads(line)
        if row['condition'] in ('C0','G'):values.setdefault((row['condition'],row['order']),[]).append(row)
    if set(values)!={(a,o) for a in ('C0','G') for o in (0,1)}:raise ValueError('four references required')
    refs=[]
    for (arm,o),rows in values.items():
        rows.sort(key=lambda x:x['visit']);manifest=read(c['manifests'][o]['path'])
        if len(rows)!=1951 or any((x['content'],x['domain'],x['subset'])!=(y['group_id'],y['domain'],y['subset']) for x,y in zip(rows,manifest)):raise ValueError('full reference pairing')
        f=root/'private'/f'ref_{arm}_o{o}.json';save(f,rows);refs.append(dict(condition=arm,order=o,values_path=str(f),values_sha256=sha(rows),parent_scalar_sha256=receipt['scalar_sha256']))
    ref=dict(file=str(actor),sha256=c['old_actor']['sha256'],source='unchanged old POST actor; no new source fit')
    payload=dict(experiment_id=ID,config_sha256=sha(c),code_sha=c['code_sha'],actors={'POST':ref},reused=refs,manifests=c['manifests'],arms=list(ARMS),policy_seed=SEED,native_seed=c['native_seed'],qualification_reused_from=sha(p),source_retraining=False)
    lock=dict(payload=payload,sha256=sha(payload));save(root/'EXPERIMENT_LOCK.json',lock);return lock

def report(c,state):
    scoring.report(c,state)
    root=Path(c['output_root']);p=root/'public/REPORT.md';text=p.read_text()
    start=text.index('RL_ORIGINAL is');end=text.index('Primary priority signal:',start)
    text=text[:start]+('RL_ORIGINAL and RL_GRATA use the SAME sealed historical R10 POST actor and carrier. RL_GRATA adds native GraTa before the unchanged full RL action/memory path. FIXED_GRATA fixes gain .8, residual zero and writer .5. C0 and G reuse exact full historical references. Six new full streams each1951/1695; no WARM arm and no source retraining in this quick screen. The original source phase was cancelled by user scope/time amendment before any target access; its artifacts and actual GPU cost remain retained.\n\n')+text[end:]
    text=text.replace('Attribution to RL additionally requires positive matched gains over WARM_GRATA and FIXED_GRATA in both orders.','RL_GRATA versus FIXED_GRATA is a diagnostic of the deployed learned policy; without matched new supervised/RL fitting it cannot isolate the effect of RL training.')
    text=text.replace('Source hard/soft and action diagnostics are retained.','New source performance metrics are NA: no source selection/refitting is performed. Earlier mechanical qualification is reused and charged.')
    text=text.replace('Full order/domain/paired aggregates, source hard/soft metrics and fixed endpoints are in the accompanying files.','Full order/domain/paired aggregates are in accompanying files; source fitting metrics are NA for this screen.')
    text+='\nUser three-hour amendment: unchanged original T0; computation ends22:25 Beijing October4 and absolute execution22:45. Partial conditions remain incomplete, not performance failures. A negative result concerns this fixed-policy coupling and does not prove that a refitted policy cannot help.\n'
    p.write_text(text)
    old=Path(c['parent_stage_root']);attempts=[read(x) for x in (old/'attempts').glob('*.json')];new=read(root/'RESOURCE_LEDGER.json')
    save(root/'public/CAMPAIGN_COST.json',dict(prior_source_qualification_gpu_seconds=sum(x['cost']['gpu_seconds'] for x in attempts),quick_gpu_seconds=new['gpu_seconds'],total_gpu_seconds=new['gpu_seconds']+sum(x['cost']['gpu_seconds'] for x in attempts),prior_source_operations_lower_bound=True,T0=c['origin']['T0']))

def run_task(c,phase,assignment,seconds,job=None,attempt=0,failure=None):
    root=Path(c['output_root']);start=time.time();deadline=min(start+seconds,c['origin']['absolute_deadline_epoch']-60 if phase=='score' else c['origin']['normal_compute_deadline_epoch'])
    env=dict(os.environ,RUN_MODE='worker',RUN_PHASE=phase,RUN_CONFIG_SHA=sha(c),RUN_STARTED=str(start),RUN_DEADLINE=str(deadline),RUN_ASSIGNMENT=json.dumps(assignment),RUN_ATTEMPT=str(attempt),CUDA_VISIBLE_DEVICES='' if phase=='score' else str(assignment['physical_id']))
    if job:env['RUN_JOB']=json.dumps(job)
    if failure:env['RUN_FAILURE']=json.dumps(failure)
    log=root/'logs'/f'{phase}.{attempt}.log'
    with log.open('ab') as f:
        p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,start_new_session=True,stdout=f,stderr=subprocess.STDOUT)
        ident=base.process_identity(p);ident.update(phase=phase,assignment=assignment,started=start,deadline=deadline,attempt=attempt);save(root/'processes'/f'{phase}.json',ident)
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


def supervise():
    c=config();root=Path(c['output_root']);state=read(root/'RUN_STATE.json');jobs=[]
    with lease(root/'supervisor_control',dict(experiment_id=ID,config=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('duplicate quick execution')
        save(root/'execution_started.json',dict(at=time.time(),pid=os.getpid(),config_sha256=sha(c)))
        try:
            profile=read(Path(c['parent_stage_root'])/'SOURCE_PRECHECK.json')['profiles']
            target=1.3*(2*1951*(profile['RL_ORIGINAL']+.15)+4*1951*(profile['FIXED_GRATA']+.15)+6*60)
            admit=dict(target_gpu_seconds=target,projected_parallel_wall_seconds=target/2,remaining_compute_wall_seconds=c['origin']['normal_compute_deadline_epoch']-time.time(),source_retraining=False,safety_factor=1.3)
            admit['admitted']=target/2<admit['remaining_compute_wall_seconds'];save(root/'PROFILE_ADMISSION.json',admit)
            if not admit['admitted']:raise RuntimeError('NOT_RUN_USER_TIME_BUDGET')
            lock=freeze(c);jobs=[dict(id=f'{a}_o{o}',arm=a,order=o,seed=SEED) for a in ARMS for o in (0,1)]
            state.update(status='TARGET_RUNNING',jobs={j['id']:'NOT_RUN' for j in jobs},target_lock_sha256=lock['sha256']);save(root/'RUN_STATE.json',state)
            for offset in range(0,len(jobs),2):
                if time.time()>=c['origin']['normal_compute_deadline_epoch']-30:break
                batch=jobs[offset:offset+2]
                with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                    fs={pool.submit(run_task,c,'online_'+j['id'],g,c['origin']['normal_compute_deadline_epoch']-time.time(),j):j for j,g in zip(batch,c['gpu_assignments'])}
                    for f in concurrent.futures.as_completed(fs):state['jobs'][fs[f]['id']]=f.result()['status'];save(root/'RUN_STATE.json',state);base.update_ledger(c,state)
            state['jobs']={k:'NOT_RUN_USER_TIME_LIMIT' if v=='NOT_RUN' else v for k,v in state['jobs'].items()};state['status']='TARGET_MATRIX_TERMINAL';save(root/'RUN_STATE.json',state)
            row=run_task(c,'score',None,1200);state['jobs']['score']=row['status'];state.update(status='COMPLETE' if row['status']=='COMPLETE' and all(state['jobs'][j['id']]=='COMPLETE' for j in jobs) else 'PARTIAL',target_scores_embargoed=False)
        except BaseException as e:state.update(status='STOPPED',stop_reason=str(e))
        state.update(ended=time.time(),delivery='PENDING_LOCAL_GITHUB');save(root/'RUN_STATE.json',state);base.update_ledger(c,state);report(c,state)

def main():
    original.ID=ID;original.config=config;original.ARMS=ARMS;base.ID=ID;base.config=config;scoring.ARMS=ARMS
    mode=os.environ['RUN_MODE']
    if mode=='prepare':print(json.dumps(dict(config_sha256=sha(prepare()))))
    elif mode=='worker':raise SystemExit(0 if original.worker()['status']=='COMPLETE' else 1)
    elif mode=='watch':base.watch()
    elif mode=='supervise':supervise()
