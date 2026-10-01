"""One frozen uncertainty-gate factor; existing shared meter, journal and watchdog."""
import os,json,time,traceback
from pathlib import Path
from ..r10_12h_core.run import read,save,sha,epoch,launch,OPS
from ..r13_full_coverage.run import worker as physical_worker,watch
from ..r8_ba.streams import rows_sha
from ..r8_ba.journal import _digest
from .view import construct,NAME,LOW,HIGH,WEIGHTS
ID='R14_UNCERTAINTY_FLIP_V1'

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or os.environ.get('RUN_CONFIG_SHA',sha(c))!=sha(c):raise ValueError('frozen config identity')
    return c

def lock(c,name):
    if name not in ('C0','CV_H025',NAME):raise ValueError('unregistered condition')
    p=dict(experiment_id=ID,condition=name,config_sha256=sha(c),weights=[1.] if name=='C0' else [.75,.25],gate=[.4,.6] if name==NAME else None,checkpoint_sha256=c['bindings']['checkpoint_sha256'])
    return dict(schema=ID+'_LOCK',payload=p,sha256=sha(p))

def check_lock(l):
    p=l.get('payload',{});n=p.get('condition')
    if l.get('schema')!=ID+'_LOCK' or l.get('sha256')!=sha(p) or p.get('experiment_id')!=ID or n not in ('C0','CV_H025',NAME) or p.get('weights')!=([1.] if n=='C0' else [.75,.25]) or p.get('gate')!=([.4,.6] if n==NAME else None) or (LOW,HIGH,WEIGHTS)!=(.4,.6,(.75,.25)):raise ValueError('gate/weight identity')

def sealed(part):
    p=Path(part['path']);on=read(p/'online_complete.json');sc=read(p/'score_complete.json')
    if on!=part['online'] or sc!=part['score'] or sc['scalar_sha256']!=_digest(p/'scalars.private.jsonl'):raise ValueError('reused scalar/online seal changed')
    return part

def prepare():
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PREVIOUS_ROOT']);oc=read(old/'RESOLVED_CONFIG.json');ledger=read(old/'RESOURCE_LEDGER.json');origin=read(root/'BUDGET_ORIGIN.json')
    if read(old/'RUN_STATE.json')['status']!='COMPLETE' or not read(old/'REUSE_QUALIFICATION.json')['passed'] or time.time()>epoch(origin['preflight_deadline']):raise ValueError('previous/preflight qualification')
    if abs(ledger['prior_package_charged_seconds']+ledger['actual_wall_seconds']-origin['prior_package_charged_seconds'])>.01 or origin['prior_gpu_worker_seconds']+origin['new_cap_seconds']>86400 or epoch(origin['absolute_deadline'])>1790952587.771787-3600:raise ValueError('budget/campaign lineage')
    manifests=[];reuse=[]
    for o,m in enumerate(oc['full_manifests']):
        rows=read(m['path'])
        if len(rows)!=1951 or rows_sha(rows)!=m['sha256'] or sum(r['subset']=='remaining_dev' for r in rows)!=1695:raise ValueError('full arrivals')
        p=root/'private'/f'order{o}.json';save(p,rows);manifests.append(dict(m,path=str(p)))
        for n in ('C0','G'):
            x=next(x for x in oc['reused_results'] if x['condition']==n and x['order']==o);reuse.append(dict(condition=n,order=o,parts=[sealed(x)]))
        q=next(x for x in oc['reused_results'] if x['condition']=='CV_H025' and x['order']==o);p=old/'target'/f'CV_H025_o{o}';new=sealed(dict(path=str(p),online=read(p/'online_complete.json'),score=read(p/'score_complete.json')));reuse.append(dict(condition='CV_H025',order=o,parts=[sealed(q),new]))
    source=old/'SOURCE_COMPARISON.reused.json';src=read(source)
    if src['status']!='COMPLETE' or len(src['rows'])!=64:raise ValueError('old source complete')
    costs=oc['old_profile_prior_cost'].copy()
    for k in OPS:costs[k]+=ledger['operations'][k]
    c=dict(experiment_id=ID,code_sha=os.environ['RUN_SHA'],base_sha=os.environ['BASE_SHA'],origin=origin,output_root=str(root),previous_root=str(old),bindings=oc['bindings'],gpu_assignments=oc['gpu_assignments'],old_profile_prior_cost=costs,manifests=manifests,jobs=[dict(id=f'{NAME}_o{o}',arm=NAME,order=o,seed=20260924) for o in (0,1)],reused_results=reuse,reused_source_path=str(source),reused_source_sha256=sha(src),source_indices=src['indices'],source_reference=[x for x in src['rows'] if x['condition'] in ('C0','CV_H025')],candidate_conditions=[NAME],patient_dependence=oc['patient_dependence'],gate=[LOW,HIGH],weights=list(WEIGHTS),status='PREFLIGHT')
    save(root/'RESOLVED_CONFIG.json',c);save(root/'RESOURCE_LEDGER.json',origin);return c

def preflight(c,guard):
    import torch
    from ..r10_carrier.source import data,bn_record,parameter_stamp
    from ..r10_carrier.diagnostic import source_item
    names=('C0','CV_H025',NAME);hosts={};handles=[];checks=[];timings={n:0. for n in names}
    try:
        for n in names:hosts[n]=construct(n,c,lock(c,n));handles.append(guard.meter.attach(hosts[n][0].segmenter.model))
        before={n:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for n,(h,_) in hosts.items()}
        if any(v!=before['C0'] for v in before.values()) or any(m['training'] or m['track_running_stats'] or m['running_mean'] is not None or m['running_var'] is not None for m in before['C0']['BN'].values()):raise ValueError('same frozen current-statistics backbones')
        with torch.no_grad(),data(c,guard) as src:
            for i in (0,16,32,48):
                for v in (0,4,8,12,16,20,24,28):
                    guard();image,_,mode=source_item(src,i,v);out={}
                    for n,(h,_) in hosts.items():
                        torch.cuda.synchronize();t=time.perf_counter();out[n]=h.step(image)[0];torch.cuda.synchronize();timings[n]+=time.perf_counter()-t
                    p=out['C0'].float().sigmoid();mask=(p>=.4)&(p<=.6);expected=torch.where(mask,out['CV_H025'],out['C0'])
                    if not torch.equal(out[NAME],expected) or not torch.equal(out[NAME][~mask],out['C0'][~mask]) or not torch.equal(out[NAME].sigmoid()[~mask],out['C0'].sigmoid()[~mask]):raise ValueError('protected logits/probability or gate equation mismatch')
                    checks.append(dict(episode=i,visit=v,mode=mode,equation_exact=True,protected_probability_exact=True,eligible_fraction=float(mask.float().mean())))
        after={n:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for n,(h,_) in hosts.items()}
        if after!=before:raise ValueError('qualification changed backbone')
        for h,_ in hosts.values():h.check_frozen(True)
        proof=dict(passed=True,images=32,logical_visits=96,model_forwards=160,checks=checks,timings=timings,parameters_BN_unchanged=True,config_sha256=sha(c));save(Path(c['output_root'])/'ZERO_PARITY.json',proof);return dict(passed=True,visits=96)
    finally:
        for h in handles:h.remove()
        for _,close in hosts.values():close()

def source(c,guard):
    import torch
    from ..r10_carrier.source import data
    from ..r10_carrier.diagnostic import source_item
    from ..r9_current_first.validation import dice
    h,close=construct(NAME,c,lock(c,NAME));handle=guard.meter.attach(h.segmenter.model);root=Path(c['output_root']);rows=[]
    try:
        with torch.no_grad(),data(c,guard) as src:
            for i in c['source_indices']:
                hard=[];soft=[];eligible=[]
                for v in range(32):
                    guard();image,label,mode=source_item(src,i,v);z,t=h.step(image);hd,sd=dice(z.sigmoid(),label);hard.append(hd);soft.append(sd);eligible.append(t['uncertain_fraction'])
                h.check_frozen(True);hd=torch.tensor(hard,dtype=torch.float64).mean(0);sd=torch.tensor(soft,dtype=torch.float64).mean(0);rows.append(dict(condition=NAME,episode=i,mode=mode,visits=32,hard_OD=float(hd[0]),hard_OC=float(hd[1]),hard_Dice=float(hd.mean()),soft_OD=float(sd[0]),soft_OC=float(sd[1]),soft_Dice=float(sd.mean()),eligible_fraction=sum(eligible)/32));save(root/'SOURCE_COMPARISON.json',dict(status='RUNNING',new_rows=rows,completed_episodes=len(rows),planned_episodes=16))
        if guard.meter.cost['model_forwards']!=1024 or len(rows)!=16:raise ValueError('source exact count')
        if sha(read(c['reused_source_path']))!=c['reused_source_sha256']:raise ValueError('old source changed')
        save(root/'SOURCE_COMPARISON.json',dict(status='COMPLETE',new_rows=rows,reused_rows=c['source_reference'],new_visits=512,episodes=16,indices=c['source_indices'],selection=False));return dict(visits=512,episodes=16)
    finally:handle.remove();close()

def target_action(c,guard,phase,failure):
    if phase=='source':return source(c,guard)
    from ..r10_use_write_rl.target import online,score,retire_probabilities
    j=next(j for j in c['jobs'] if j['id']==phase.split(':',1)[1]);p=Path(c['output_root'])/'target'/j['id'];m=c['manifests'][j['order']];rows=read(m['path']);lk=lock(c,NAME)
    if rows_sha(rows)!=m['sha256']:raise ValueError('frozen full arrivals')
    if phase.startswith('score:'):
        on=read(p/'online_complete.json');r=score(rows,c['bindings']['target_root'],p,j['id'],on['identity']['context_sha256'],lk,guard,failure,check_lock);retire_probabilities(p);return r
    host,close=construct(NAME,c,lk);handle=guard.meter.attach(host.segmenter.model);step=host.step
    def counted(image):
        before=guard.meter.cost.copy();r=step(image)
        if tuple(guard.meter.cost[k]-before[k] for k in OPS)!=(2,0,0,0):raise ValueError('gate physical operation contract')
        return r
    host.step=counted
    try:return online(host,rows,c['bindings']['target_root'],p,j['id'],lk,guard,failure,check_lock)
    finally:handle.remove();close()

def admit():
    c=config();root=Path(c['output_root']);z=read(root/'ZERO_PARITY.json');old=read(Path(c['previous_root'])/'RESOURCE_LEDGER.json')
    if not z['passed'] or read(root/'attempts/preflight.0.json')['status']!='COMPLETE':raise ValueError('qualification failed')
    per=max(a['wall_seconds']/927 for a in old['attempts'] if a['phase'].startswith('online:'));score=max(a['wall_seconds']/927 for a in old['attempts'] if a['phase'].startswith('score:'));target_seconds=1.3*(3902*(per+score+.13)+120);source_seconds=1.3*(z['timings'][NAME]*16+120);eligible=source_seconds<=900 and target_seconds<=3600 and source_seconds+target_seconds<epoch(c['origin']['normal_compute_deadline'])-time.time() and time.time()<epoch(c['origin']['preflight_deadline'])
    c.update(status='FROZEN',admission=dict(source_seconds=source_seconds,target_seconds=target_seconds,safety_factor=1.3,admitted=eligible));save(root/'RESOLVED_CONFIG.json',c)
    if not eligible:raise RuntimeError('NOT_RUN_BUDGET')
    save(root/'private/authorization.json',dict(experiment_id=ID,code_sha=c['code_sha'],config_sha256=sha(c),qualification_sha256=sha(z),scope='successor three; fixed0.4-0.6 original confidence gate/.25 flip; source512 and full3902 targets; no score-dependent tuning'))
    return c['admission']

def supervise():
    from ..r9_current_first.storage import lease
    from .report import report
    c=config();root=Path(c['output_root']);auth=read(root/'private/authorization.json');normal=epoch(c['origin']['normal_compute_deadline']);hard=epoch(c['origin']['compute_deadline'])
    if auth['config_sha256']!=sha(c) or auth['code_sha']!=c['code_sha'] or time.time()>epoch(c['origin']['preflight_deadline']):raise ValueError('authorization/deadline')
    with lease(root/'supervisor_control',dict(experiment_id=ID)):
        if (root/'execution_started.json').exists():raise ValueError('reentry needs ledger reconciliation')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)));state=dict(experiment_id=ID,status='RUNNING',source='RUNNING',jobs={j['id']:'NOT_RUN' for j in c['jobs']},recovery_used=False);save(root/'RUN_STATE.json',state)
        def run(phase,deadline):
            r=launch(c,phase,min(normal,deadline))
            if r['status']=='FAILED' and r['failure']['class']=='INFRASTRUCTURE' and phase!='source' and not state['recovery_used']:
                state['recovery_used']=True;save(root/'RUN_STATE.json',state);r=launch(c,phase,min(hard,deadline+900,time.time()+900),1,r['failure'])
            return r
        try:
            r=run('source',time.time()+900);state['source']=r['status'];save(root/'RUN_STATE.json',state)
            if r['status']!='COMPLETE':raise RuntimeError('source comparison execution failed')
            deadline=time.time()+3600
            for j in c['jobs']:
                state['jobs'][j['id']]='RUNNING';save(root/'RUN_STATE.json',state);r=run('online:'+j['id'],deadline)
                if r['status']=='COMPLETE':r=run('score:'+j['id'],deadline)
                state['jobs'][j['id']]=r['status'];save(root/'RUN_STATE.json',state)
                if r['failure'] and r['failure']['class'] in ('ISOLATION','NUMERICAL','IDENTITY_OR_IMPLEMENTATION'):raise RuntimeError(r['failure']['reason'])
        except BaseException as e:state['stop_reason']=str(e);traceback.print_exc()
        for k,v in state['jobs'].items():
            if v in ('RUNNING','NOT_RUN'):state['jobs'][k]='NOT_RUN_STOPPED'
        state.update(status='COMPLETE' if state['source']=='COMPLETE' and all(v=='COMPLETE' for v in state['jobs'].values()) else 'PARTIAL',ended=time.time());save(root/'RUN_STATE.json',state);report(c,state);return state

def main():
    mode=os.environ['RUN_MODE']
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',jobs=len(prepare()['jobs']))))
    elif mode=='worker':r=physical_worker(config(),preflight,target_action,160);__import__('sys').exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='preflight':
        c=config();root=Path(c['output_root'])
        with (root/'qualification_started.json').open('x') as f:json.dump(dict(at=time.time(),config_sha256=sha(c)),f)
        r=launch(c,'preflight',min(time.time()+600,epoch(c['origin']['preflight_deadline'])));print(json.dumps(r));__import__('sys').exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='admit':print(json.dumps(admit()))
    elif mode=='supervise':print(json.dumps(supervise()))
    elif mode=='watch':__import__('sys').exit(watch(config()))
    else:raise ValueError('explicit mode')
