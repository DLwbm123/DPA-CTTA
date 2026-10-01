"""Final source-only mechanism; no target worker dispatch exists."""
import os,json,time,traceback,statistics
from pathlib import Path
from ..r10_12h_core.run import read,save,sha,epoch,launch,OPS
from ..r13_full_coverage.run import worker as physical_worker,watch
from ..r14_uncertainty_flip.run import source as shared_source
from .view import NAME,RULE,construct
ID='R15_DECISION_SUPPORT_V1'

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or os.environ.get('RUN_CONFIG_SHA',sha(c))!=sha(c) or c['jobs'] or c['target_policy']!='NOT_RUN_SOURCE_ONLY':raise ValueError('frozen source-only identity')
    return c

def lock(c,name):
    if name not in ('C0','CV_H025',NAME):raise ValueError('source condition')
    p=dict(experiment_id=ID,condition=name,config_sha256=sha(c),rule=RULE if name==NAME else None,weights=[1.] if name=='C0' else [.75,.25],checkpoint_sha256=c['bindings']['checkpoint_sha256'])
    return dict(schema=ID+'_LOCK',payload=p,sha256=sha(p))

def prepare():
    root=Path(os.environ['RUN_ROOT']);old=Path(os.environ['PREVIOUS_ROOT']);oc=read(old/'RESOLVED_CONFIG.json');ledger=read(old/'RESOURCE_LEDGER.json');origin=read(root/'BUDGET_ORIGIN.json');src=read(old/'SOURCE_COMPARISON.json')
    if read(old/'RUN_STATE.json')['status']!='COMPLETE' or src['status']!='COMPLETE' or len(src['new_rows'])!=16 or time.time()>epoch(origin['preflight_deadline']):raise ValueError('previous source/deadline')
    if abs(ledger['prior_package_charged_seconds']+ledger['actual_wall_seconds']-origin['prior_package_charged_seconds'])>.01 or origin['prior_gpu_worker_seconds']+origin['new_cap_seconds']>86400 or epoch(origin['absolute_deadline'])>1790952587.771787-3600:raise ValueError('campaign/cost')
    costs=oc['old_profile_prior_cost'].copy()
    for k in OPS:costs[k]+=ledger['operations'][k]
    c=dict(experiment_id=ID,code_sha=os.environ['RUN_SHA'],base_sha=os.environ['BASE_SHA'],origin=origin,output_root=str(root),previous_root=str(old),bindings=oc['bindings'],gpu_assignments=oc['gpu_assignments'],old_profile_prior_cost=costs,jobs=[],target_policy='NOT_RUN_SOURCE_ONLY',rule=RULE,reused_source_path=str(old/'SOURCE_COMPARISON.json'),reused_source_sha256=sha(src),source_indices=src['indices'],source_reference=src['reused_rows']+src['new_rows'],patient_dependence='Same exposed source simulator validation; no independent replication or target soft claim',status='PREFLIGHT')
    save(root/'RESOLVED_CONFIG.json',c);save(root/'RESOURCE_LEDGER.json',origin);return c

def preflight(c,guard):
    import torch
    from ..r10_carrier.source import data,bn_record,parameter_stamp
    from ..r10_carrier.diagnostic import source_item
    hosts={};handles=[];rows=[];names=('C0','CV_H025',NAME);timings={n:0. for n in names}
    try:
        for n in names:hosts[n]=construct(n,c,lock(c,n));handles.append(guard.meter.attach(hosts[n][0].segmenter.model))
        before={n:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for n,(h,_) in hosts.items()}
        if any(v!=before['C0'] for v in before.values()) or any(m['training'] or m['track_running_stats'] or m['running_mean'] is not None or m['running_var'] is not None for m in before['C0']['BN'].values()):raise ValueError('same immutable current-statistics models')
        with torch.no_grad(),data(c,guard) as src:
            for i in (0,16,32,48):
                for v in (0,4,8,12,16,20,24,28):
                    guard();image,_,mode=source_item(src,i,v);out={}
                    for n,(h,_) in hosts.items():
                        torch.cuda.synchronize();t=time.perf_counter();out[n]=h.step(image)[0];torch.cuda.synchronize();timings[n]+=time.perf_counter()-t
                    a=out['C0'];q=out['CV_H025'];z=out[NAME];change=(a.sigmoid()>=.5)!=(q.sigmoid()>=.5)
                    if not torch.equal(z,torch.where(change,q,a)) or not torch.equal(z.sigmoid()>=.5,q.sigmoid()>=.5) or not torch.equal(z.sigmoid()[~change],a.sigmoid()[~change]):raise ValueError('decision support/probability preservation')
                    rows.append(dict(episode=i,visit=v,mode=mode,quarter_hard_exact=True,native_probability_on_agreement_exact=True,support_fraction=float(change.float().mean())))
        after={n:dict(parameters=parameter_stamp(h.segmenter),BN=bn_record(h.segmenter)) for n,(h,_) in hosts.items()}
        if after!=before:raise ValueError('model/BN changed')
        for h,_ in hosts.values():h.check_frozen(True)
        save(Path(c['output_root'])/'ZERO_PARITY.json',dict(passed=True,images=32,logical_visits=96,model_forwards=160,rows=rows,timings=timings,parameters_BN_unchanged=True,config_sha256=sha(c)));return dict(passed=True,visits=96)
    finally:
        for h in handles:h.remove()
        for _,close in hosts.values():close()

def source_action(c,guard,phase,failure):
    if phase!='source' or failure:raise ValueError('source-only policy; no retries or target dispatch')
    result=shared_source(c,guard,NAME,construct,lock);root=Path(c['output_root']);s=read(root/'SOURCE_COMPARISON.json');refs={r['episode']:r for r in c['source_reference'] if r['condition']=='CV_H025'};delta=[abs(r[k]-refs[r['episode']][k]) for r in s['new_rows'] for k in ('hard_OD','hard_OC','hard_Dice')];proof=dict(passed=max(delta)<=1e-12,episodes=16,max_hard_metric_delta=max(delta),tolerance=1e-12,selection=False);save(root/'SOURCE_HARD_PARITY.json',proof)
    if not proof['passed']:
        s['status']='FAILED_REFERENCE_PARITY';save(root/'SOURCE_COMPARISON.json',s);raise ValueError('decision preservation or old source path differs')
    return result

def admit():
    c=config();root=Path(c['output_root']);z=read(root/'ZERO_PARITY.json')
    if not z['passed'] or read(root/'attempts/preflight.0.json')['status']!='COMPLETE':raise ValueError('source qualification')
    projected=1.3*(z['timings'][NAME]*16+120);eligible=projected<=900 and projected<epoch(c['origin']['normal_compute_deadline'])-time.time() and time.time()<epoch(c['origin']['preflight_deadline']);c.update(status='FROZEN',admission=dict(source_seconds=projected,safety_factor=1.3,admitted=eligible));save(root/'RESOLVED_CONFIG.json',c)
    if not eligible:raise RuntimeError('NOT_RUN_BUDGET')
    save(root/'private/authorization.json',dict(experiment_id=ID,code_sha=c['code_sha'],config_sha256=sha(c),qualification_sha256=sha(z),scope='final fourth successor SOURCE_ONLY512 accesses; no targets, no source or qualification retry'));return c['admission']

def supervise():
    from ..r9_current_first.storage import lease
    from .report import report
    c=config();root=Path(c['output_root']);a=read(root/'private/authorization.json')
    if a['config_sha256']!=sha(c) or a['code_sha']!=c['code_sha'] or time.time()>epoch(c['origin']['preflight_deadline']):raise ValueError('frozen authorization/deadline')
    with lease(root/'supervisor_control',dict(experiment_id=ID)):
        if (root/'execution_started.json').exists():raise ValueError('reentry needs ledger reconciliation')
        save(root/'execution_started.json',dict(at=time.time(),config_sha256=sha(c)));state=dict(experiment_id=ID,status='RUNNING',source='RUNNING',jobs={},targets='NOT_RUN_SOURCE_ONLY',recovery_used=False);save(root/'RUN_STATE.json',state)
        try:r=launch(c,'source',min(epoch(c['origin']['normal_compute_deadline']),time.time()+900));state['source']=r['status']
        except BaseException as e:state.update(source='FAILED',stop_reason=str(e));traceback.print_exc()
        state.update(status='COMPLETE' if state['source']=='COMPLETE' else 'PARTIAL',ended=time.time());save(root/'RUN_STATE.json',state);report(c,state);return state

def main():
    mode=os.environ['RUN_MODE']
    if mode=='prepare':print(json.dumps(dict(status='PREPARED',target_jobs=len(prepare()['jobs']))))
    elif mode=='worker':r=physical_worker(config(),preflight,source_action,160);__import__('sys').exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='preflight':
        c=config();root=Path(c['output_root'])
        with (root/'qualification_started.json').open('x') as f:json.dump(dict(at=time.time(),config_sha256=sha(c)),f)
        r=launch(c,'preflight',min(epoch(c['origin']['preflight_deadline']),time.time()+600));print(json.dumps(r));__import__('sys').exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='admit':print(json.dumps(admit()))
    elif mode=='supervise':print(json.dumps(supervise()))
    elif mode=='watch':__import__('sys').exit(watch(config()))
    else:raise ValueError('explicit mode')
