"""Small serial driver; unchanged R10 kernels, separately sealed protocol and budget."""
import copy,csv,hashlib,json,math,os,signal,statistics,subprocess,sys,time,traceback
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path

ID='R10_12H_CORE_V1'
SEED=20260924
VAL=[16*m+i for m in range(4) for i in (0,4,10,13)]
ARMS=('N','G','GR_RET_EMA','GR_RET_EMA_CONST_HALF')
OPS=('model_forwards','backward_calls','optimizer_steps','vjp_calls')
LIMITS=dict(zip(OPS,(12000000,2000000,1000000,0)))

def read(p):return json.loads(Path(p).read_text())
def sha(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp')
    with t.open('w') as f:json.dump(x,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
def epoch(s):return datetime.fromisoformat(s).timestamp()
def config():
    c=read(os.environ['RUN_CONFIG']);expected=os.environ.get('RUN_CONFIG_SHA')
    if expected and sha(c)!=expected:raise ValueError('frozen config changed')
    if c['experiment_id']!=ID:raise ValueError('run identity')
    return c

def subset(orders,n,tag):
    base=orders[0]
    if len({r['group_id'] for r in base})!=len(base):raise ValueError('duplicate content requires explicit policy')
    if any({r['group_id'] for r in order}!={r['group_id'] for r in base} for order in orders):raise ValueError('order populations differ')
    n=min(n,len(base));strata=defaultdict(list)
    for r in base:strata[(r['domain'],r['subset'])].append(r)
    quotas={k:n*len(v)//len(base) for k,v in strata.items()}
    for k in sorted(strata,key=lambda k:(-(n*len(strata[k])%len(base)),k))[:n-sum(quotas.values())]:quotas[k]+=1
    keep={r['group_id'] for k,rs in strata.items() for r in sorted(rs,key=lambda r:hashlib.sha256(f"{ID}|{tag}|{r['group_id']}".encode()).digest())[:quotas[k]]}
    return [[r for r in rows if r['group_id'] in keep] for rows in orders]

def project(profile,warm,post,n):
    m=profile['measurements']
    def cost(k):return m[k]['gpu_seconds']+m[k].get('cpu_wall_seconds',0)
    source=cost('setup/source')+warm*cost('train/WARM')+post*cost('train/GR_RET_EMA')+16*cost('validation_episode')+(warm+post)*cost('source_log')+(math.ceil(warm/50)+math.ceil(post/50)+6)*cost('source_checkpoint')+60
    target=0
    for a in ('N_SOURCE_EVAL','G_CTTA','POLICY','POLICY'):
        # Old rows already include 1.3*max; keep cold spikes. Added read/fsync margin per visit.
        target+=2*(cost('setup/'+a)+n*(cost('online/'+a)+cost('target_append')+cost('score_visit')+.13)+math.ceil(n/50)*cost('target_checkpoint/'+a)+cost('score_seal')+30)
    return dict(source_seconds=source,target_seconds=target,safety_factor=1.3,additional_unmeasured_read_io_seconds_per_visit=.13)

def prepare():
    old=read(os.environ['OLD_CONFIG']);profile=read(os.environ['OLD_PROFILE']);root=Path(os.environ['RUN_ROOT']);origin=read(root/'BUDGET_ORIGIN.json')
    from ..r10_use_write_rl.assets import validate_metadata
    from ..r7_source_prep.registry import verified
    from ..r10_use_write_rl.streams import sequence
    from ..r1.plan import registration_digest
    from ..r8_ba.streams import rows_sha
    validate_metadata(old['bindings'])
    if profile['identity']['code_sha']!=old['code_sha'] or profile['identity']['gpu_assignments']!=old['gpu_assignments']:raise ValueError('profile binding')
    candidates=[];selected=None
    for tier,w,p in (('A',2000,4000),('B',1000,1024),('C',512,512)):
        estimate=project(profile,w,p,1024);candidates.append(dict(tier=tier,warm=w,post=p,**estimate))
        if estimate['source_seconds']<=3*3600:selected=(tier,w,p);break
    if selected is None:raise RuntimeError('TRAINING_BUDGET_INSUFFICIENT')
    tier,w,p=selected
    n=next((n for n in (1024,512) if project(profile,w,p,n)['target_seconds']<=4.5*3600),None)
    if n is None:raise RuntimeError('OVER_CAP')
    ref=old['bindings']['refs']['target'];reg=json.loads(verified(ref['path'],ref['sha256'],16*1024**2));tag=registration_digest(reg)
    orders=subset([sequence(reg,0),sequence(reg,1)],n,tag)
    manifests=[]
    for order,rows in enumerate(orders):
        path=root/'private'/f'order{order}.json';save(path,rows)
        manifests.append(dict(path=str(path),sha256=rows_sha(rows),visits=len(rows),principal=sum(r['subset']=='remaining_dev' for r in rows),strata=dict(Counter(f"{r['domain']}|{r['subset']}" for r in rows))))
    jobs=[dict(id=f'{a}_o{o}',arm=a,order=o,seed=None if a=='N' else 20260907 if a=='G' else SEED) for a in ARMS for o in range(2)]
    c=dict(experiment_id=ID,base_sha=os.environ['BASE_SHA'],code_sha=os.environ['RUN_SHA'],bindings=old['bindings'],gpu_assignments=old['gpu_assignments'],output_root=str(root),origin=origin,tier=tier,warm=w,post=p,val_indices=VAL,manifests=manifests,jobs=jobs,projection=project(profile,w,p,n),tier_candidates=candidates,registration_digest=tag,sampling='SHA256 R10_12H_CORE_V1|registration_digest|group_id; largest-remainder domain/subset quotas; original relative orders',patient_dependence='Unknown; content visits are not independent patients',old_profile_identity=profile['identity'],old_profile_prior_cost=profile['prior_cost'],scheduler=dict(warmup=100,warm_peak=3e-4,warm_end=3e-5,post_peak=3e-5,post_end=3e-6),recovery_policy=dict(max_jobs=1,max_extra_attempts_per_job=1),peak_vram_bytes=max(x['max_reserved_bytes'] for x in profile['measurements'].values()))
    save(root/'RESOLVED_CONFIG.json',c);return c

def check_lock(lock):
    p=lock.get('payload',{})
    if lock.get('schema')!=ID+'_SOURCE_LOCK' or lock.get('sha256')!=sha(p) or set(p.get('sources',{}))!={'WARM','POST'} or not p.get('validation_complete'):raise ValueError('core source endpoints not sealed')

def seal_sources(c):
    root=Path(c['output_root']);rs={k:read(root/'source'/k/'complete.json') for k in ('WARM','POST')}
    for k,n in (('WARM',c['warm']),('POST',c['post'])):
        if rs[k]['steps']!=n or rs[k]['config_sha256']!=sha(c):raise ValueError('fixed endpoint identity')
    p=dict(config_sha256=sha(c),sources={k:sha(v) for k,v in rs.items()},validation_complete=True)
    lock=dict(schema=ID+'_SOURCE_LOCK',payload=p,sha256=sha(p));save(root/'source_lock.json',lock);return lock

class Guard:
    def __init__(self,c,phase):
        self.c=c;self.root=Path(c['output_root']);self.phase=phase;self.deadline=float(os.environ['RUN_DEADLINE']);self.last=0;self.meter=None
        self.prior={k:0 for k in OPS}
        for p in (self.root/'attempts').glob('*.json'):
            for k in OPS:self.prior[k]+=read(p).get('cost',{}).get(k,0)
        for k in OPS:self.prior[k]+=c['old_profile_prior_cost'].get(k,0)
    def observe(self,cost=None):
        if time.time()>=self.deadline-20:raise TimeoutError('stage deadline; closing reserve preserved')
        if cost and any(cost[k]+self.prior[k]>LIMITS[k] for k in OPS):raise RuntimeError('aggregate operation cap')
        if time.monotonic()-self.last>10:
            size=sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file())
            if size+32*1024**2>=64*1024**3:raise RuntimeError('aggregate storage cap')
            save(self.root/'active_cost.json',dict(phase=self.phase,at=time.time(),cost=cost or {},disk_bytes=size));self.last=time.monotonic()
    def __call__(self):
        if self.meter:self.meter.check()
        else:self.observe()

def source(c,guard,smoke=False,failure=None):
    import torch
    from ..r10_use_write_rl.assets import open_source
    from ..r10_use_write_rl.source import Source
    from ..r10_use_write_rl.controller import Actor
    from ..r10_use_write_rl.learning import Trainer,rng_state,restore_rng
    from ..r10_use_write_rl.evaluation import episode
    from ..r9_current_first.storage import PhaseJournal,write_torch,load_torch
    root=Path(c['output_root'])/('smoke' if smoke else 'source')
    with open_source(c['bindings'],c['gpu_assignments'][0],guard) as (data,seg,oracles,controller):
        handle=guard.meter.attach(seg.model);s=Source(data,oracles,seg,controller,sha(c['bindings']));actor=Actor(SEED)
        try:
            for name,method,n in (('WARM','WARM',c['warm']),('POST','GR_RET_EMA',c['post'])):
                binding=dict(experiment_id=ID,config_sha256=sha(c),stage=name,total=n,smoke=smoke)
                trainer=Trainer(actor,controller,s,SEED,method,binding,actor if name=='POST' else None,guard,total=n)
                path=root/name;path.mkdir(parents=True,exist_ok=True);journal=PhaseJournal(path,'fit',binding)
                done=journal.completed()
                if done:
                    meta=read(journal.root/'latest.json');trainer.restore(load_torch(journal.root/f"checkpoint.{meta['slot']}.pt",meta['sha256'])['snapshot'])
                else:
                    journal.begin(trainer,failure if journal.root.exists() else None)
                    while trainer.steps<(2 if smoke else n):
                        guard();row=trainer.step();journal.append(row)
                        if trainer.steps%50==0 or trainer.steps==(2 if smoke else n):journal.checkpoint(trainer)
                    journal.complete(dict(steps=trainer.steps))
                artifact=dict(file='actor.pt',sha256=write_torch(path/'actor.pt',dict(actor=actor.state_dict(),method=method,steps=trainer.steps,config_sha256=sha(c))))
                save(path/'complete.json',dict(config_sha256=sha(c),method=method,steps=trainer.steps,artifact=artifact))
            if not smoke:
                saved=rng_state();a=copy.deepcopy(actor).eval();a.requires_grad_(False)
                try:rows=[episode(s,a,SEED,i,guard=guard) for i in c['val_indices']]
                finally:restore_rng(saved)
                if sorted(Counter(r['mode'] for r in rows).values())!=[4]*4:raise ValueError('validation balance')
                save(root/'validation.json',dict(indices=c['val_indices'],rows=rows,selection=False));seal_sources(c)
        finally:handle.remove()
    return dict(warm=2 if smoke else c['warm'],post=2 if smoke else c['post'])

def resolved(job,artifact):
    a=job['arm'];policy=a.startswith('GR_RET_EMA')
    return dict(arm={'N':'N_SOURCE_EVAL','G':'G_CTTA'}.get(a,'GR_RET_EMA'),method='GR_RET_EMA' if policy else a,seed=job['seed'],source_job='POST' if policy else None,artifact=artifact if policy else None,diagnostic='CONST_HALF' if a.endswith('CONST_HALF') else None)

def target(c,guard,job,scoring=False,failure=None):
    from ..r10_use_write_rl.target import online,score,retire_probabilities
    from ..r10_use_write_rl.factory import construct
    from ..r8_ba.streams import rows_sha
    root=Path(c['output_root']);lock=read(root/'source_lock.json');check_lock(lock)
    if lock['payload']['config_sha256']!=sha(c):raise ValueError('source/config identity')
    for name,h in lock['payload']['sources'].items():
        if sha(read(root/'source'/name/'complete.json'))!=h:raise ValueError('source receipt changed')
    manifest=c['manifests'][job['order']];rows=read(manifest['path'])
    if rows_sha(rows)!=manifest['sha256']:raise ValueError('stream identity')
    jobroot=root/'target'/job['id']
    if scoring:
        seal=read(jobroot/'online_complete.json')
        result=score(rows,c['bindings']['target_root'],jobroot,job['id'],seal['identity']['context_sha256'],lock,guard,failure,check_lock)
        retire_probabilities(jobroot);return result
    artifact=read(root/'source'/'POST'/'complete.json')['artifact']
    host,close=construct(resolved(job,artifact),c,root,lock)
    model=host.native.model if hasattr(host,'native') else host.segmenter.model
    handle=guard.meter.attach(model);step=host.step;expected=(9,2,1,0) if job['arm']=='G' else (1,0,0,0) if job['arm']=='N' else (2,0,0,0)
    def counted(image):
        before=guard.meter.cost.copy();result=step(image)
        actual=tuple(guard.meter.cost[k]-before[k] for k in OPS)
        if actual!=expected:raise ValueError(f'actual per-image operation contract {actual} != {expected}')
        return result
    host.step=counted
    try:return online(host,rows,c['bindings']['target_root'],jobroot,job['id'],lock,guard,failure,check_lock)
    finally:handle.remove();close()

def worker():
    import torch
    from ..r9_current_first.physical import Meter
    from ..r10_use_write_rl.assets import gpu_policy
    from ..r10_use_write_rl.runtime import classification
    c=config();phase=os.environ['RUN_PHASE'];root=Path(c['output_root']);guard=Guard(c,phase);start=time.time();meter=None;failure=None;result=None
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    attempt=os.environ.get('RUN_ATTEMPT','0');prior_failure=json.loads(os.environ['RUN_FAILURE']) if os.environ.get('RUN_FAILURE') else None
    try:
        if phase.startswith('score:'):
            job=next(j for j in c['jobs'] if j['id']==phase.split(':',1)[1]);result=target(c,guard,job,True,prior_failure)
        else:
            gpu_policy(c['gpu_assignments'][0])
            with Meter(None,guard.observe) as meter:
                guard.meter=meter
                if phase in ('source','smoke'):result=source(c,guard,phase=='smoke',prior_failure)
                else:result=target(c,guard,next(j for j in c['jobs'] if j['id']==phase.split(':',1)[1]),False,prior_failure)
    except BaseException as e:
        failure=dict(reason=str(e),error_type=type(e).__name__,**{'class':'RESOURCE' if isinstance(e,TimeoutError) else classification(e)},evidence=dict(config_sha256=sha(c),phase=phase,attempt=attempt));traceback.print_exc()
    cost=meter.cost if meter else dict.fromkeys((*OPS,'gpu_seconds'),0)
    receipt=dict(phase=phase,attempt=attempt,status='FAILED' if failure else 'COMPLETE',failure=failure,result=result,cost=cost,started=start,ended=time.time(),wall_seconds=time.time()-start,config_sha256=sha(c))
    save(root/'attempts'/(phase.replace(':','_')+'.'+attempt+'.json'),receipt)
    return receipt

def terminate_group(p):
    try:os.killpg(p.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        p.wait(timeout=10)
    # A child can outlive a parent which handled SIGTERM itself.
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass

def launch(c,phase,deadline,attempt=0,failure=None):
    root=Path(c['output_root']);env=os.environ.copy();env.update(RUN_MODE='worker',RUN_PHASE=phase,RUN_DEADLINE=str(deadline),RUN_ATTEMPT=str(attempt),RUN_CONFIG_SHA=sha(c))
    if failure:env['RUN_FAILURE']=json.dumps(failure)
    if phase.startswith('score:'):env['CUDA_VISIBLE_DEVICES']=''
    if time.time()>=deadline-30:raise TimeoutError('no stage budget remaining')
    logfile=root/'logs'/(phase.replace(':','_')+f'.{attempt}.log');logfile.parent.mkdir(exist_ok=True)
    with logfile.open('ab') as log:
        p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        save(root/'active_process.json',dict(pid=p.pid,pgid=p.pid,phase=phase,deadline=deadline,active=True))
        timed_out=False
        try:p.wait(timeout=max(.1,deadline-time.time()))
        except subprocess.TimeoutExpired:timed_out=True;terminate_group(p)
        finally:
            if p.poll() is None:terminate_group(p)
            save(root/'active_process.json',dict(pid=p.pid,pgid=p.pid,phase=phase,active=False,returncode=p.returncode,timed_out=timed_out))
    receipt=root/'attempts'/(phase.replace(':','_')+f'.{attempt}.json')
    if receipt.exists():return read(receipt)
    active=read(root/'active_cost.json') if (root/'active_cost.json').exists() else {}
    r=dict(phase=phase,attempt=attempt,status='FAILED',failure={'class':'RESOURCE' if timed_out else 'PROCESS_EXIT','reason':'supervisor timeout' if timed_out else f'child exit {p.returncode}','evidence':dict(pid=p.pid)},cost=active.get('cost',{}) if active.get('phase')==phase else {},wall_seconds=None)
    save(receipt,r);return r

def supervise():
    c=config();root=Path(c['output_root']);origin=c['origin'];normal=epoch(origin['normal_compute_deadline']);hard=epoch(origin['compute_deadline'])
    if time.time()>epoch(origin['preflight_deadline']):raise TimeoutError('PREFLIGHT_INCOMPLETE')
    auth=read(root/'private'/'authorization.json')
    if auth['config_sha256']!=sha(c) or auth['code_sha']!=c['code_sha']:raise ValueError('private authorization binding')
    from ..r9_current_first.storage import lease
    with lease(root/'supervisor',dict(experiment_id=ID,config_sha256=sha(c))):
        if (root/'execution_started.json').exists():raise ValueError('restart requires reconciliation; T0 cannot reset')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)))
        state=dict(experiment_id=ID,status='RUNNING',origin=origin,source='NOT_RUN',jobs={j['id']:'NOT_RUN' for j in c['jobs']},recovery_used=False)
        save(root/'RUN_STATE.json',state)
        def run(phase,deadline):
            r=launch(c,phase,deadline)
            if r['status']=='FAILED' and r['failure']['class']=='INFRASTRUCTURE' and not state['recovery_used'] and time.time()<hard-60:
                state['recovery_used']=True;save(root/'RUN_STATE.json',state)
                r=launch(c,phase,min(hard,deadline+3600),1,r['failure'])
            return r
        source_start=time.time();r=run('source',min(normal,source_start+3*3600));state['source']=r['status'];save(root/'RUN_STATE.json',state)
        if r['status']=='COMPLETE':
            target_start=time.time();deadline=min(normal,target_start+4.5*3600)
            for job in c['jobs']:
                try:
                    r=run('online:'+job['id'],deadline)
                    if r['status']=='COMPLETE':r=run('score:'+job['id'],deadline)
                    state['jobs'][job['id']]=r['status']
                except TimeoutError:state['jobs'][job['id']]='NOT_RUN_BUDGET'
                save(root/'RUN_STATE.json',state)
                if r.get('failure') and r['failure']['class'] in ('ISOLATION','IDENTITY_OR_IMPLEMENTATION','NUMERICAL'):break
        state['status']='COMPLETE' if state['source']=='COMPLETE' and all(v=='COMPLETE' for v in state['jobs'].values()) else 'INCOMPLETE'
        state['ended']=time.time();save(root/'RUN_STATE.json',state);report(c,state)
    return state

def report(c,state):
    root=Path(c['output_root']);summary=[];details=[];traces=[]
    for j in c['jobs']:
        item=dict(method=j['arm'],order=j['order'],status=state['jobs'][j['id']],visits=None,scored=None,Dice_OD=None,Dice_OC=None,Dice_macro=None,Dice_pooled=None)
        p=root/'target'/j['id'];complete=p/'score_complete.json'
        if item['status']=='COMPLETE' and complete.exists():
            rows=[json.loads(line) for line in (p/'scalars.private.jsonl').read_text().splitlines()];cells=defaultdict(list);pool=[]
            for row in rows:
                if row['subset']=='remaining_dev':
                    for m in row['metrics']:cells[row['domain'],m['channel']].append(m['dice']);pool.append(m['dice'])
            if len(cells)!=8:raise ValueError('result domain/channel coverage')
            channels=sorted({ch for _,ch in cells})
            means={ch:statistics.mean(statistics.mean(v) for (d,k),v in cells.items() if k==ch) for ch in channels}
            item.update(visits=len(rows),scored=sum(r['subset']=='remaining_dev' for r in rows),Dice_macro=statistics.mean(means.values()),Dice_pooled=statistics.mean(pool))
            for ch,val in means.items():item['Dice_'+ch.upper()]=val
            for (domain,ch),vs in sorted(cells.items()):details.append(dict(method=j['arm'],order=j['order'],domain=domain,channel=ch,dice=statistics.mean(vs),n=len(vs)))
            tracepath=p/'visits.jsonl'
            if tracepath.exists():
                values=[json.loads(x) for x in tracepath.read_text().splitlines()]
                traces.append(dict(method=j['arm'],order=j['order'],n=len(values),means={k:statistics.mean(x[k] for x in values if k in x) for k in ('write','gain','state_norm','mass') if any(k in x for x in values)}))
        summary.append(item)
    fields=list(dict.fromkeys(k for x in summary for k in x))
    with (root/'RESULTS.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(summary)
    save(root/'domain_results.json',details);save(root/'behavior.json',traces)
    attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))]
    ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(origin=c['origin'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a.get('cost',{}).get('gpu_seconds',0) for a in attempts),operations={k:sum(a.get('cost',{}).get(k,0) for a in attempts) for k in OPS},disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),attempts=attempts,recovery_used=state.get('recovery_used',False));save(root/'RESOURCE_LEDGER.json',ledger)
    delta=[]
    for order in range(2):
        by={r['method']:r for r in summary if r['order']==order};a=by['GR_RET_EMA']['Dice_macro']
        for other in ('N','G','GR_RET_EMA_CONST_HALF'):
            b=by[other]['Dice_macro'];delta.append(dict(order=order,comparison='R10-'+other,Dice_macro_delta=None if a is None or b is None else a-b))
    save(root/'differences.json',delta)
    lines=[f'# {ID}',f'Status: {state["status"]}',f'Code: {c["code_sha"]}',f'Training tier: {c["tier"]}; requested WARM={c["warm"]}, POST={c["post"]}; source state={state["source"]}.','Single policy seed; fixed carrier; two development streams. No family selection, SUP comparison, cross-seed or blind-test claim.','Unfinished entries are missing, never zero. All trajectories use the same content cohort.',f'Actual wall seconds: {ledger["actual_wall_seconds"]:.3f}; GPU-worker seconds: {ledger["gpu_worker_seconds"]:.3f}; historical charged seconds: {c["origin"]["history_charged_seconds"]}.','\n```json',json.dumps(dict(results=summary,differences=delta,domains=details,behavior=traces,attempts=attempts),indent=2),'```']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n');return summary

def main():
    mode=os.environ.get('RUN_MODE')
    if mode=='prepare':print(json.dumps({k:v for k,v in prepare().items() if k in ('tier','warm','post','projection','manifests')}))
    elif mode=='worker':
        r=worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='smoke':
        c=config();r=launch(c,'smoke',epoch(c['origin']['preflight_deadline']));print(json.dumps(r));sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='supervise':print(json.dumps(supervise()))
    else:raise ValueError('explicit run mode required')
