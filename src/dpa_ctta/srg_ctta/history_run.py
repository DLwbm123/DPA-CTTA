"""Frozen-encoder history readouts and independent CPU scoring."""
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .method import ARMS,NORMALIZED,Host,intensity_views
from ..r78_cross_patient.objective import BNState,Field,reference,direct_logits
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..source_correction.method import correction
from ..boundary_amplitude.run import controller
from ..r51_same_state.run import read,save

LEARNERS=ARMS+('CW1','CW2')
POLICIES=('RESET','RETAIN')
STAGES=('PRE','POST')
REFERENCES=('SOURCE_EVAL','CURRENT_BN','D_OLD','R_OLD','R_SHUFFLE_OLD')
BLOCKS=(0,1,2,3,0)
EXPECTED=3600


from .run import schedule,guard_assets
from .feature_history import Stream,JOBS as HISTORY_JOBS

def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r37_cw_orientation.method import Host as Native,candidates
    root=Path(c['output_root']);started=time.time();status=dict(status='QUALIFYING',started=started,predictions=0,arrivals=0,legacy_parity=0)
    save(root/'RUN_STATE.json',status)
    def guard():
        if time.time()-started>c['gpu_cap_seconds']-20:raise TimeoutError('fixed GPU worker cap')
    try:
        if c['rdc'] or c['engram'] or c['new_training'] or c['grouping']!='CONTENT_DEVELOPMENT_USER_AUTHORIZED':raise ValueError('scope mismatch')
        ledger=read(Path(c['old_head_root'])/'RESOURCE_LEDGER.json')
        if not ledger['all_workers_retired'] or any(a['exit_code'] for a in ledger['attempts'][-2:]):raise ValueError('previous run must be complete')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        if c.get('strict_deterministic'):
            torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=read(root/'private/panel.json')
        if len(panel)!=8 or len({r['group_id'] for r in panel})!=8 or any('mask_path' in r for r in panel):raise ValueError('eight content-only stream records required')
        opened=guard_assets(c,panel)
        for denied in (root/'scorer/panel.json',Path(c['old_head_root'])/'public/RESULTS.json'):
            try:denied.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('access guard did not reject forbidden file')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        base,_=build_reference_model('fundus');base.load_state_dict(state);base.cuda().eval().requires_grad_(False)
        legacy=BNState(copy.deepcopy(base));headroot=Path(c['old_head_root']);prior=Path(c['prior_root'])
        heads={}
        for arm,folder,key,epoch in (('D',prior,'D',150),('R',headroot,'R',100)):
            payload=torch.load(folder/'private'/f'head_{key}.pt',map_location='cpu',weights_only=True)
            if payload['selected_episode']!=epoch:raise ValueError('frozen source head changed')
            head=Field().cuda().eval().requires_grad_(False);head.load_state_dict(payload['state_dict']);heads[arm]=head
        before={n:p._version for n,p in base.named_parameters()};buffers={n:b.clone() for n,b in base.named_buffers()}
        images=[read_pixels(r['image_path'],'fundus',r['image_size']) for r in panel]
        oldrows=read(headroot/'private/predictions.json') if c.get('check_legacy_parity',True) else []
        lookup={(r['pair'],r['condition'],r['arm']):r['index'] for r in oldrows if r['mode']=='DIRECT'}
        legacy_names=dict(CURRENT_BN='B0',D_OLD='D_OLD',R_OLD='R',R_SHUFFLE_OLD='R_SHUFFLE')
        cache={};feature_cache={};source_logits={};head_checks=0;prob_cache={};diags={};records=[];step_rows=[];reset_predictions={};reset_mismatches=0
        def prediction(z):return z.detach().cpu().numpy()[0]>=0
        def diagnostics(z,z0):
            dz=z-z0;dp=z.sigmoid()-z0.sigmoid()
            return dict(logit_rms=float(dz.square().mean().sqrt()),probability_abs_mean=float(dp.abs().mean()),changed_pixels=int((prediction(z)!=prediction(z0)).sum()))
        def append(row,pred,d,prob=None):
            ix=len(records)
            with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(np.packbits(pred).tobytes())
            if c.get('save_probabilities'):
                if prob is None:raise ValueError('calibration probability missing')
                with (root/'probabilities'/f'{ix:04d}.q8').open('xb') as f:f.write(prob)
            records.append(dict(index=ix,**row,diagnostics=d,nesting_violation_pixels=int((pred[1]&~pred[0]).sum())))
            status['predictions']=len(records)
        # Frozen predictors are evaluated once per unique image/condition, then reused
        # at repeated arrivals; the adapting learner never consumes this cache.
        for case in range(8):
            for k,spec in enumerate(FINAL):
                guard();pixels=perturb(images[case],spec,SEED+case);x=normalize(pixels).cuda()
                with torch.no_grad():
                    out=base(x);ze=out[0];feat=out[2]
                    if feat.shape!=(1,32,512,512) or base.seg_head.weight.shape!=(2,32,1,1):raise ValueError('frozen feature/head interface differs')
                    if not torch.equal(ze,base.seg_head(feat)):raise ValueError('exact frozen head reconstruction failed')
                    feature_cache[case,k]=feat.detach().clone();source_logits[case,k]=ze.detach().clone();head_checks+=1
                    zc=legacy(x,legacy.zero());ref=reference(x,zc)
                    outputs=dict(SOURCE_EVAL=ze,CURRENT_BN=zc,D_OLD=direct_logits(zc,ref,heads['D'](ref.inputs)),R_OLD=correction(zc,ref,heads['R'],'R'),R_SHUFFLE_OLD=correction(zc,ref,heads['R'],'R_SHUFFLE'))
                    for arm,z in outputs.items():
                        pred=prediction(z);cache[case,k,arm]=pred
                        if c.get('save_probabilities'):prob_cache[case,k,arm]=(z.detach().sigmoid()*255).round().byte().cpu().numpy()[0].tobytes()
                        diags[case,k,arm]=diagnostics(z,ze)
                        if arm in legacy_names and c.get('check_legacy_parity',True):
                            ix=lookup[case,k,legacy_names[arm]];old=(headroot/'predictions'/f'{ix:04d}.bits').read_bytes()
                            if np.packbits(pred).tobytes()!=old:raise ValueError('legacy reference parity failed: '+arm)
                            status['legacy_parity']+=1
        for order in range(2):
            for arrival in schedule(order):
                for arm in REFERENCES:
                    key=arrival['case'],arrival['condition'],arm
                    append(dict(**arrival,order=order,arm=arm,policy='REFERENCE',stage='STATIC'),cache[key],diags[key],prob_cache.get(key))
        save(root/'RUN_STATE.json',status)
        core_forwards=0;native_forwards=0;native_backwards=0;trajectories=[]
        repeat_logits=[];qualification_arrivals=0;qualification_exact=False;state_events=0
        def equal(a,b):
            if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
            if isinstance(a,torch.Tensor):return torch.equal(a,b)
            return a==b
        for order in range(2):
            stream=Stream(base.seg_head.weight[:,:,0,0]);tick=time.time()
            for pos,arrival in enumerate(schedule(order)):
                guard();case=arrival['case'];k=arrival['condition'];z0=source_logits[case,k]
                pre,post,d=stream.step(feature_cache[case,k][0],z0[0]);state_events+=5
                if any(not torch.equal(pre[a,p],z0[0].double()) for a,p in HISTORY_JOBS if p=='RESET' or pos==0):raise ValueError('source initial or RESET PRE identity failed')
                if order==0:repeat_logits.extend((pre['FEATURE_Q','RETAIN'].cpu(),post['FEATURE_Q','RETAIN'].cpu()))
                for arm,policy in HISTORY_JOBS:
                    for stage,zz in (('PRE',pre[arm,policy]),('POST',post[arm,policy])):
                        z=zz[None];pred=prediction(z)
                        if policy=='RESET' and arrival['block'] in (0,4):
                            key=arm,order,case,stage
                            if arrival['block']==0:reset_predictions[key]=pred.copy()
                            else:reset_mismatches+=int((pred!=reset_predictions[key]).sum())
                        append(dict(**arrival,order=order,arm=arm,policy=policy,stage=stage),pred,dict(**diagnostics(z,z0),history_images_before=pos if policy=='RETAIN' else 0,**d))
                    step_rows.append(dict(**arrival,order=order,arm=arm,policy=policy,**d))
                status['arrivals']+=len(HISTORY_JOBS)
                if order==0 and pos==1:
                    avg=(time.time()-tick)/2;projection=time.time()-started+(80+40-2)*avg*1.35+320+60
                    ok=projection<=c['gpu_cap_seconds']-20
                    save(root/'QUALIFICATION.json',dict(status='PASS' if ok else 'NOT_ADMITTED',projected_worker_seconds=projection,measured_shared_arrival_seconds=avg,source_head_reconstruction_exact=head_checks==32,source_frames=head_checks,head_features=32,no_parameter_updates=True,final_label_reads=0))
                    if not ok:raise TimeoutError('frozen history matrix not admitted')
                    status['status']='GENERATING'
                if pos%8==7:status.update(current_arm='SHARED_HISTORY',policy='BY_READOUT',order=order,position=pos+1);save(root/'RUN_STATE.json',status)
            if order==0:
                probe=Stream(base.seg_head.weight[:,:,0,0])
                for pos,arrival in enumerate(schedule(0)):
                    guard();case=arrival['case'];k=arrival['condition'];pre,post,d=probe.step(feature_cache[case,k][0],source_logits[case,k][0])
                    if not torch.equal(pre['FEATURE_Q','RETAIN'].cpu(),repeat_logits[2*pos]) or not torch.equal(post['FEATURE_Q','RETAIN'].cpu(),repeat_logits[2*pos+1]):raise ValueError('cached-feature history logits repeat failed')
                    qualification_arrivals+=1
                if not equal(stream.snapshot(),probe.snapshot()):raise ValueError('history state repeat failed')
                qualification_exact=True;del probe,repeat_logits
                save(root/'FULL_REPEATABILITY.json',dict(status='PASS',arrivals=40,paired_readouts=80,logits_exact=True,history_state_exact=True,scope='Same process, cached frozen features, FEATURE_Q RETAIN forward PRE/POST; all three persistent history states; not encoder rerun or native trajectories'))
            trajectories.extend(dict(arm=a,policy=p,order=order,arrivals=40,parameter_updates=0) for a,p in HISTORY_JOBS)
            del stream
        for steps in (1,2):
            arm='CW'+str(steps)
            for policy in POLICIES:
                for order in range(2):
                    conf=copy.deepcopy(candidates()[3]);conf.setdefault('params',{})['online_steps_per_arrival']=steps
                    host=Native(state,conf,SEED,'finite_stream',device='cuda:0');initial=host.snapshot()
                    try:
                        for pos,arrival in enumerate(schedule(order)):
                            guard();case=arrival['case'];k=arrival['condition']
                            if policy=='RESET':host.restore(initial)
                            pixels=perturb(images[case],FINAL[k],SEED+case);x=normalize(pixels).cuda()
                            with torch.no_grad():z0=base(x)[0];pre=host._readonly(host.native.model,pixels)
                            post,d=host.step(pixels);post=post.cuda()
                            native_forwards+=1+sum(v['counts']['forwards'] for v in d['native']);native_backwards+=sum(v['counts']['backwards'] for v in d['native'])
                            for stage,z in (('PRE',pre),('POST',post)):
                                pred=prediction(z)
                                if policy=='RESET' and arrival['block'] in (0,4):
                                    key=(arm,order,case,stage)
                                    if arrival['block']==0:reset_predictions[key]=pred.copy()
                                    else:reset_mismatches+=int((pred!=reset_predictions[key]).sum())
                                append(dict(**arrival,order=order,arm=arm,policy=policy,stage=stage),pred,dict(**diagnostics(z,z0),optimizer_age=host.native.steps),(z.detach().sigmoid()*255).round().byte().cpu().numpy()[0].tobytes() if c.get('save_probabilities') else None)
                            status['arrivals']+=1
                            if pos%8==7:status.update(current_arm=arm,policy=policy,order=order,position=pos+1);save(root/'RUN_STATE.json',status)
                        trajectories.append(dict(arm=arm,policy=policy,order=order,arrivals=40,optimizer_steps=40*steps))
                    finally:host.close()
        if len(records)!=2320 or len(step_rows)!=640 or status['arrivals']!=960 or len(trajectories)!=24:raise ValueError('history matrix coverage differs')
        if not qualification_exact or qualification_arrivals!=40:raise ValueError('history repeat missing')
        if any(p._version!=before[n] or p.grad is not None for n,p in base.named_parameters()):raise ValueError('frozen source parameter changed')
        if any(not torch.equal(b,buffers[n]) for n,b in base.named_buffers()):raise ValueError('frozen source buffer changed')
        save(root/'private/predictions.json',records);save(root/'private/history.json',step_rows)
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',trajectories=trajectories,logical_scientific_arrivals=960,shared_history_arrivals=80,history_state_updates=state_events,history_readout_arrivals=640,source_frames=head_checks,source_head_reconstruction_exact=head_checks==32,encoder_cache_frames=32,qualification_arrivals=40,qualification_paired_readouts=80,source_immutable=True,history_parameter_updates=0,native_forwards=native_forwards,native_backwards=native_backwards,reset_return_mask_differing_pixels=reset_mismatches,source_support_paired=True,scope='Frozen cached full-resolution features; independent state per head; source-only fractional assignments'))
        save(root/'ACCESS_AUDIT.json',dict(source_image_files=len(opened),source_label_reads=0,evaluation_label_reads=0,prior_outcome_score_reads=0,new_source_training=False,future_images_used_for_update=False,source_encoder_frozen=True,patient_independence=False,previously_exposed_development_contents=True,cache_scope='Frozen features precomputed once per registered case/condition; state receives only current arrival features'))
        used=sum(p.stat().st_size for folder in (('private','predictions','gates','gate_predictions','logs')+(('probabilities',) if c.get('save_probabilities') else ())) for p in (root/folder).rglob('*') if p.is_file())
        if used>c['output_cap_bytes']:raise ValueError('output cap')
        status.update(status='GPU_COMPLETE',output_bytes=used,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        status.update(status='NOT_EVALUABLE',error_type=type(error).__name__,reason=str(error));raise
    finally:status.update(ended=time.time(),gpu_worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


from .paired_state import summary
from .score import unpack
from ..r78_cross_patient.score import metric,csvout

PRIMARY='FEATURE_Q_RETAIN-LOGIT_Q_RETAIN:PRE'
ALL_JOBS=list(HISTORY_JOBS)+[(a,p) for a in ('CW1','CW2') for p in ('RESET','RETAIN')]


def validate(c):
    assert c['design_version']=='SOURCE_PRESERVING_HISTORY_V1' and c['primary']==PRIMARY
    assert c['readout_jobs']==[list(x) for x in HISTORY_JOBS]
    assert c['expected_readouts']==2320 and c['expected_logical_arrivals']==960
    assert c['history_kappa']==3 and c['full_resolution_statistics'] and c['bootstrap']=='FIRST_CURRENT_IMAGE'
    assert c['active_rule']=='FP64_POPULATION_VARIANCE_GT_ZERO_NO_FLOOR'
    assert c['strict_deterministic'] and not c['check_legacy_parity'] and not c['save_probabilities']
    assert not any(c[k] for k in ('rdc','engram','new_training'))
    assert (c['gpu_cap_seconds'],c['cpu_cap_seconds'],c['output_cap_bytes'])==(1200,900,2147483648)


def score(c):
    from ..source_io import read_mask
    validate(c);root=Path(c['output_root']);start=time.time()
    ret=read(root/'processes/worker.json');state=read(root/'RUN_STATE.json');repeat=read(root/'FULL_REPEATABILITY.json');audit=read(root/'MECHANISM_AUDIT.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code']==0
    assert state['status']=='GPU_COMPLETE' and (state['predictions'],state['arrivals'])==(2320,960)
    assert repeat['status']=='PASS' and repeat['paired_readouts']==80 and repeat['history_state_exact']
    rows=read(root/'private/predictions.json');history=read(root/'private/history.json');panel=read(root/'scorer/panel.json')
    def key(r):return r['case'],r['block'],r['order'],r['arm'],r['policy'],r['stage']
    expected={(r['case'],r['block'],o,a,p,s) for o in range(2) for r in schedule(o) for a,p in ALL_JOBS for s in ('PRE','POST')}
    expected|={(r['case'],r['block'],o,a,'REFERENCE','STATIC') for o in range(2) for r in schedule(o) for a in REFERENCES}
    assert len(rows)==len(expected)==2320 and {key(r) for r in rows}==expected and len(history)==640 and len(panel)==8
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    sources={}
    for r in rows:
        if r['arm']=='SOURCE_EVAL':sources[r['case'],r['condition']]=unpack(root/'predictions'/f"{r['index']:04d}.bits")[1]
    cache={};grouped={};individual=[]
    for r in rows:
        if time.time()-start>c['cpu_cap_seconds']-10:raise TimeoutError('CPU score cap')
        raw,pred=unpack(root/'predictions'/f"{r['index']:04d}.bits");k=r['case'],r['condition'],raw
        if k not in cache:cache[k]=metric(pred,labels[r['case']],sources[r['case'],r['condition']])
        mm,ratios=cache[k];item=dict(**r,metrics=mm,ratios=ratios);individual.append(item)
        grouped.setdefault((r['arm'],r['policy'],r['stage']),[]).append(item)
    save(root/'scorer/individual.private.json',individual)
    def mean(v):
        a=[x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(a)) if a else None,len(a)
    absolute=[];cubes={};channels=[];transitions=[];diagnostics=[]
    for (a,p,s),rr in grouped.items():
        assert len(rr)==80
        cube=np.full((8,5,2,2),np.nan)
        for r in rr:cube[r['case'],r['block'],r['order']]=[m['dice_percent'] for m in r['metrics']]
        assert np.isfinite(cube).all();cubes[a,p,s]=cube
        assd,n=mean([m['assd_pixel'] for r in rr for m in r['metrics']]);hd,_=mean([m['hd95_pixel'] for r in rr for m in r['metrics']]);vcdr,vn=mean([r['ratios']['vcdr_abs_error'] for r in rr])
        absolute.append(dict(arm=a,policy=p,stage=s,dice_percent=float(cube.mean()),assd_pixel=assd,hd95_pixel=hd,surface_defined=n,surface_total=160,vcdr_mae=vcdr,vcdr_defined=vn,empty_predictions=sum(m['pred_empty'] for r in rr for m in r['metrics']),nesting_violation_pixels=sum(r['nesting_violation_pixels'] for r in rr)))
        for b in range(5):
            for o in range(2):
                for ch,name in enumerate(('OD','OC')):channels.append(dict(arm=a,policy=p,stage=s,block=b,order=o,channel=name,dice_percent=float(cube[:,b,o,ch].mean())))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','boundary3px'):
                for measure in ('corrected_fraction','spoiled_fraction'):
                    value,n=mean([r['metrics'][ch]['regions'][region][measure] for r in rr]);transitions.append(dict(arm=a,policy=p,stage=s,channel=name,region=region,measure=measure,macro_fraction=value,defined=n,total=80))
        d=dict(arm=a,policy=p,stage=s)
        for f in ('logit_rms','probability_abs_mean','changed_pixels','history_images_before'):
            vals=[r['diagnostics'][f] for r in rr if f in r['diagnostics']]
            if vals:d.update({f+'_mean':float(np.mean(vals)),f+'_min':float(np.min(vals)),f+'_max':float(np.max(vals))})
        diagnostics.append(d)
    def take(a,p='RETAIN',s='PRE'):return cubes[a,'REFERENCE','STATIC'] if a in REFERENCES else cubes[a,p,s]
    contrasts=[]
    def add(n,d):
        v=summary(n,d);contrasts.append(v);return v
    main=take('FEATURE_Q');primary=add(PRIMARY,main-take('LOGIT_Q'))
    sh=add('FEATURE_Q-SHUFFLE_Q:PRE',main-take('SHUFFLE_Q'))
    first=add('FEATURE_Q-SOURCE:FIRST_BLOCK_PRE',main[:,0]-take('SOURCE_EVAL')[:,0])
    src=add('FEATURE_Q-SOURCE:ALL_PRE',main-take('SOURCE_EVAL'))
    back=add('FEATURE_Q-SOURCE:RETURN_PRE',main[:,4]-take('SOURCE_EVAL')[:,4])
    add('FEATURE_Q_RETAIN-RESET:POST',take('FEATURE_Q',s='POST')-take('FEATURE_Q','RESET','POST'))
    for a,p in ALL_JOBS:
        add(a+'_'+p+':POST-PRE',take(a,p,'POST')-take(a,p,'PRE'))
        if (a,p)!=('FEATURE_Q','RETAIN'):add('FEATURE_Q_RETAIN_POST-'+a+'_'+p,take('FEATURE_Q',s='POST')-take(a,p,'POST'))
    for a in REFERENCES:add('FEATURE_Q_RETAIN_POST-'+a,take('FEATURE_Q',s='POST')-take(a))
    def row(a,p='RETAIN',s='POST'):return next(x for x in absolute if (x['arm'],x['policy'],x['stage'])==(a,p,s))
    candidate=row('FEATURE_Q');source=row('SOURCE_EVAL','REFERENCE','STATIC')
    frontier=[(a,'REFERENCE','STATIC') for a in REFERENCES]+[('FEATURE_Q','RESET','POST'),('LOGIT_Q','RETAIN','POST'),('LOGIT_Q','RESET','POST'),('SHUFFLE_Q','RETAIN','POST')]+[(a,p,'POST') for a in ('CW1','CW2') for p in ('RESET','RETAIN')]
    gates=dict(primary_mean=primary['delta_pp']>=.5,primary_ci=primary['ci95_pp'][0]>0,shuffled=sh['ci95_pp'][0]>0,first_pre=first['ci95_pp'][0]>0,all_pre=src['ci95_pp'][0]>0,worst_pre=src['worst_cell_pp']>=-1,return_pre=back['delta_pp']>=-.2,
        strong_post=all(candidate['dice_percent']>row(a,p,s)['dice_percent'] for a,p,s in frontier),
        structure=all(candidate[f] is not None and candidate[f]<=source[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')) and candidate['surface_defined']>=source['surface_defined'] and candidate['vcdr_defined']>=source['vcdr_defined'],
        empty_nesting=candidate['empty_predictions']<=source['empty_predictions'] and candidate['nesting_violation_pixels']<=source['nesting_violation_pixels'],
        source_and_support=audit['source_immutable'] and audit['source_support_paired'] and audit['source_head_reconstruction_exact'] and audit['history_parameter_updates']==0)
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,gates=gates,advance_development_gate=all(gates.values()),readouts=2320,logical_scientific_arrivals=960,shared_history_arrivals=80,history_state_updates=400,readout_trajectories=24,absolute=absolute,contrasts=contrasts,method_scope='Independent source-calibrated frozen-feature history; not original GAIN; no parameter updates',patient_independence=False,interval_scope='Descriptive eight reused contents with condition/order/channel bound'))
    save(root/'public/DIAGNOSTICS.json',diagnostics)
    hr=[r for r in history if (r['arm'],r['policy'])==('FEATURE_Q','RETAIN')]
    assert len(hr)==80 and all(r['shared_source_support_paired'] and r['parameter_updates']==0 for r in history)
    support_rows=[]
    for ch,name in enumerate(('OD','OC')):
        active=[r['shared_full_active_dimensions'][ch] for r in hr]
        support_rows.append(dict(channel=name,arrivals=80,full_active_min=min(active),full_active_max=max(active),full_active_mean=float(np.mean(active)),margin_active_min=min(r['shared_margin_active_dimensions'][ch] for r in hr),support_background_min=min(r['shared_full_support'][ch][0] for r in hr),support_background_max=max(r['shared_full_support'][ch][0] for r in hr),support_foreground_min=min(r['shared_full_support'][ch][1] for r in hr),support_foreground_max=max(r['shared_full_support'][ch][1] for r in hr),source_support_paired=True,parameter_updates=0))
    csvout(root/'public/history_audit.csv',support_rows)
    for n,v in [('absolute',absolute),('condition_channel',channels),('transitions',transitions),('contrasts',contrasts)]:csvout(root/'public'/f'{n}.csv',v)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',readouts=2320,evaluation_label_files=8,absolute_rows=29,unique_metric_evaluations=len(cache),cpu_seconds=time.time()-start))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS' and c['source_sha']!='PREPARATION'
        sys.exit(controller(c))
    elif mode=='worker':worker(c)
    elif mode=='score':score(c)
    else:raise ValueError('unknown mode')
