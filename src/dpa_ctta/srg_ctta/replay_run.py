"""One frozen whole-image replay comparison; labels only in retired-worker CPU scoring."""
import os
from pathlib import Path
import sys
import time
import numpy as np
from .run import worker,controller,read,save,schedule,REFERENCES
from .image_replay import Host,KINDS
from .score import unpack
from ..r78_cross_patient.score import metric,interval,csvout

POLICIES={'DIVERSITY':('RESET','FRESH'),'RANDOM_MEMORY':('FRESH',),'FIFO':('FRESH',),'CURRENT':('RESET','FRESH')}
JOBS=[(a,p) for a,pp in POLICIES.items() for p in pp]+[(a,p) for a in ('CW1','CW2') for p in ('RESET','RETAIN')]
PRIMARY='DIVERSITY_FRESH-RANDOM_MEMORY_FRESH:OC_PRE'


def validate(c):
    assert c['design_version']=='IMAGE_REPLAY_V1' and c['primary']==PRIMARY
    assert tuple(c['core_arms'])==KINDS and {a:tuple(p) for a,p in c['policies_by_arm'].items()}==POLICIES
    assert tuple(c['normalized_arms'])==KINDS and c['qualification_arm']=='DIVERSITY'
    assert (c['expected_readouts'],c['core_arrivals'],c['native_arrivals'])==(2000,480,320)
    assert c['steps_per_arrival']==1 and c['strict_deterministic'] and c['repeat_first_reset'] and c['paired_state']
    assert c['repeat_policy']=='FRESH' and not c['save_probabilities']
    assert not c['check_legacy_parity'] and not c['reset_optimizer_per_arrival']
    assert not any(c[k] for k in ('rdc','engram','new_training'))
    assert (c['memory_capacity'],c['microbatches'],c['descriptor_quantiles'])==(2,3,16)
    assert c['memory_admission']=='AFTER_POST' and c['matched_step_l2']==.01


def contrast(name,d):
    if d.shape not in ((8,5,2,2),(8,5,2),(8,2,2),(8,2)) or not np.isfinite(d).all():
        raise ValueError('eight content groups; bound conditions/orders/channels')
    return dict(comparison=name,delta_pp=float(d.mean()),ci95_pp=interval(d),
        worst_cell_pp=float(d.min()),negative_cells=int((d<0).sum()),total_cells=int(d.size))


def score(c):
    from ..source_io import read_mask
    validate(c);root=Path(c['output_root']);start=time.time()
    retired=read(root/'processes/worker.json');state=read(root/'RUN_STATE.json')
    repeat=read(root/'FULL_REPEATABILITY.json');mechanism=read(root/'MECHANISM_AUDIT.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and retired['exit_code']==0 and not retired['active']
    assert state['status']=='GPU_COMPLETE' and (state['predictions'],state['arrivals'])==(2000,800)
    assert repeat['status']=='PASS' and repeat['arrivals']==40 and repeat['paired_readouts']==80 and repeat['extra_state_exact']
    assert mechanism['normalized_update_events']==480 and mechanism['normalized_events_matched']==480 and mechanism['zero_proposals']==0
    rows=read(root/'private/predictions.json');updates=read(root/'private/updates.json');panel=read(root/'scorer/panel.json')
    def key(r):return r['case'],r['block'],r['order'],r['arm'],r['policy'],r['stage']
    expected={(r['case'],r['block'],o,a,p,s) for o in range(2) for r in schedule(o) for a,p in JOBS for s in ('PRE','POST')}
    expected|={(r['case'],r['block'],o,a,'REFERENCE','STATIC') for o in range(2) for r in schedule(o) for a in REFERENCES}
    assert len(rows)==len(expected)==2000 and {key(r) for r in rows}==expected and len(updates)==480 and len(panel)==8
    # RESET memory is cleared on each image: diversity must equal the no-replay control.
    raw_lookup={key(r):(root/'predictions'/f"{r['index']:04d}.bits").read_bytes() for r in rows}
    for case in range(8):
        for block in range(5):
            for order in range(2):
                for stage in ('PRE','POST'):
                    assert raw_lookup[case,block,order,'DIVERSITY','RESET',stage]==raw_lookup[case,block,order,'CURRENT','RESET',stage]
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    sources={}
    for r in rows:
        if r['arm']=='SOURCE_EVAL':sources[r['case'],r['condition']]=unpack(root/'predictions'/f"{r['index']:04d}.bits")[1]
    cache={};grouped={};individual=[]
    for r in rows:
        if time.time()-start>c['cpu_cap_seconds']-10:raise TimeoutError('CPU scoring cap')
        raw,pred=unpack(root/'predictions'/f"{r['index']:04d}.bits");k=r['case'],r['condition'],raw
        if k not in cache:cache[k]=metric(pred,labels[r['case']],sources[r['case'],r['condition']])
        mm,ratios=cache[k];item=dict(**r,metrics=mm,ratios=ratios)
        individual.append(item);grouped.setdefault((r['arm'],r['policy'],r['stage']),[]).append(item)
    save(root/'scorer/individual.private.json',individual)
    def defined(v):
        good=[float(x) for x in v if x is not None and np.isfinite(x)]
        return (float(np.mean(good)) if good else None),len(good)
    cubes={};absolute=[];channels=[];transitions=[];region_cubes={};diagnostics=[]
    for (a,p,s),rr in grouped.items():
        assert len(rr)==80
        cube=np.full((8,5,2,2),np.nan)
        for r in rr:cube[r['case'],r['block'],r['order']]=[m['dice_percent'] for m in r['metrics']]
        assert np.isfinite(cube).all();cubes[a,p,s]=cube
        assd,n=defined([m['assd_pixel'] for r in rr for m in r['metrics']]);hd,_=defined([m['hd95_pixel'] for r in rr for m in r['metrics']]);vc,vn=defined([r['ratios']['vcdr_abs_error'] for r in rr])
        absolute.append(dict(arm=a,policy=p,stage=s,dice_percent=float(cube.mean()),od_dice_percent=float(cube[...,0].mean()),oc_dice_percent=float(cube[...,1].mean()),assd_pixel=assd,hd95_pixel=hd,surface_defined=n,surface_total=160,vcdr_mae=vc,vcdr_defined=vn,vcdr_total=80,empty_predictions=sum(m['pred_empty'] for r in rr for m in r['metrics']),nesting_violation_pixels=sum(r['nesting_violation_pixels'] for r in rr)))
        for b in range(5):
            for o in range(2):
                for ch,name in enumerate(('OD','OC')):channels.append(dict(arm=a,policy=p,stage=s,block=b,order=o,channel=name,dice_percent=float(cube[:,b,o,ch].mean())))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','boundary3px'):
                net=np.full((8,5,2),np.nan);corrected=[];spoiled=[];denoms=[];cc=hh=0
                for r in rr:
                    v=r['metrics'][ch]['regions'][region];den=v['denominator'];denoms.append(den)
                    if not den:continue
                    cv,hv=v['corrected_fraction'],v['spoiled_fraction'];corrected.append(cv);spoiled.append(hv)
                    cc+=round(cv*den);hh+=round(hv*den)
                    net[r['case'],r['block'],r['order']]=100*(cv-hv)
                region_cubes[a,p,s,ch,region]=net
                finite=bool(np.isfinite(net).all())
                transitions.append(dict(arm=a,policy=p,stage=s,channel=name,region=region,
                    corrected_macro_fraction=float(np.mean(corrected)) if corrected else None,
                    spoiled_macro_fraction=float(np.mean(spoiled)) if spoiled else None,
                    net_pp=float(net.mean()) if finite else None,ci95_net_pp=interval(net) if finite else None,
                    corrected_pixels=cc,spoiled_pixels=hh,edited_pixels=cc+hh,region_pixels=sum(denoms),
                    defined=len(corrected),total=80,reference='SOURCE_EVAL',scope='80 logical readouts, including repeated development contents'))
        d=dict(arm=a,policy=p,stage=s)
        for field in ('logit_rms','probability_abs_mean','changed_pixels','total_affine_norm','optimizer_age'):
            v=[r['diagnostics'][field] for r in rr if field in r['diagnostics']]
            if v:d[field+'_mean']=float(np.mean(v))
        diagnostics.append(d)
    assert len(absolute)==25
    def take(a,p='FRESH',s='PRE'):return cubes[a,'REFERENCE','STATIC'] if a in REFERENCES else cubes[a,p,s]
    comparisons=[]
    def add(name,d):
        result=contrast(name,d);comparisons.append(result);return result
    primary=add(PRIMARY,(take('DIVERSITY')-take('RANDOM_MEMORY'))[...,1])
    fifo=add('DIVERSITY_FRESH-FIFO_FRESH:OC_PRE',(take('DIVERSITY')-take('FIFO'))[...,1])
    current=add('DIVERSITY_FRESH-CURRENT_FRESH:OC_PRE',(take('DIVERSITY')-take('CURRENT'))[...,1])
    first=add('DIVERSITY_FRESH-SOURCE:FIRST_PRE',take('DIVERSITY')[:,0]-take('SOURCE_EVAL')[:,0])
    source=add('DIVERSITY_FRESH-SOURCE:ALL_PRE',take('DIVERSITY')-take('SOURCE_EVAL'))
    returned=add('DIVERSITY_FRESH-SOURCE:RETURN_PRE',take('DIVERSITY')[:,4]-take('SOURCE_EVAL')[:,4])
    for a,p in JOBS:
        for s in ('PRE','POST'):
            for ch,name in ((None,'ALL'),(0,'OD'),(1,'OC')):
                d=take('DIVERSITY',s='POST')-take(a,p,s)
                if ch is not None:d=d[...,ch]
                add('DIVERSITY_FRESH_POST-'+a+'_'+p+'_'+s+':'+name,d)
    for a in REFERENCES:
        for ch,name in ((None,'ALL'),(0,'OD'),(1,'OC')):
            d=take('DIVERSITY',s='POST')-take(a)
            if ch is not None:d=d[...,ch]
            add('DIVERSITY_FRESH_POST-'+a+':'+name,d)
    update_rows=[]
    for a,pp in POLICIES.items():
        for p in pp:
            rr=[r for r in updates if (r['arm'],r['policy'])==(a,p)];assert len(rr)==80
            assert all(r['optimizer_age']==1 and len(r['steps'])==1 and r['loss_backward_calls']==3 and r['forwards']==4 for r in rr)
            assert all(all(i<=r['memory_seen_before'] for i in r['replayed_arrival_ids']) for r in rr)
            assert [r['memory_seen_before'] for r in rr]==([0]*80 if p=='RESET' else list(range(40))*2)
            assert [r['memory_before'] for r in rr]==([0]*80 if p=='RESET' or a=='CURRENT' else [min(i,2) for i in range(40)]*2)
            ss=[r['steps'][0] for r in rr]
            out=dict(arm=a,policy=p,updates=80,matched=sum(x['norm_matched'] for x in ss),zero=sum(x['zero_proposal'] for x in ss),microbatches=240,loss_backward_calls=240)
            for f in ('raw_step_norm','actual_step_norm','direction_cosine','affine_gradient_norm','loss'):
                v=[x[f] for x in ss];out[f+'_mean']=float(np.mean(v));out[f+'_min']=float(np.min(v));out[f+'_max']=float(np.max(v))
            for f in ('memory_before','replay_slots','padded_current_slots','unique_replay_entries','memory_age_mean','memory_pair_distance'):
                out[f+'_mean'],out[f+'_defined']=defined([r[f] for r in rr])
            update_rows.append(out)
    def absrow(a,p='FRESH',s='POST'):return next(r for r in absolute if (r['arm'],r['policy'],r['stage'])==(a,p,s))
    candidate=absrow('DIVERSITY');baseline=absrow('SOURCE_EVAL','REFERENCE','STATIC')
    safety={}
    for region in ('foreground','boundary3px'):
        net=region_cubes['DIVERSITY','FRESH','PRE',1,region]
        safety[region]=bool(np.isfinite(net).all() and interval(net)[0]>0)
    gates=dict(primary_mean=primary['delta_pp']>=.5,primary_ci=primary['ci95_pp'][0]>0,
        fifo_control=fifo['ci95_pp'][0]>0,current_control=current['ci95_pp'][0]>0,
        first_pre=first['ci95_pp'][0]>0,all_pre=source['ci95_pp'][0]>0,
        worst_pre=source['worst_cell_pp']>=-1,return_pre=returned['delta_pp']>=-.2,
        oc_foreground_net=safety['foreground'],oc_boundary_net=safety['boundary3px'],
        matching=all(r['matched']==80 and r['zero']==0 for r in update_rows),
        strong_post=all(candidate['dice_percent']>absrow(a,p,s)['dice_percent'] for a,p,s in
            [(a,'REFERENCE','STATIC') for a in REFERENCES]+[('DIVERSITY','RESET','POST'),('CURRENT','FRESH','POST')]+[(a,p,'POST') for a,p in JOBS if a in ('CW1','CW2')]),
        no_empty_nesting=candidate['empty_predictions']<=baseline['empty_predictions'] and candidate['nesting_violation_pixels']<=baseline['nesting_violation_pixels'],
        structure_coverage=all(candidate[f] is not None and candidate[f]<=baseline[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')) and candidate['surface_defined']>=baseline['surface_defined'] and candidate['vcdr_defined']>=baseline['vcdr_defined'])
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=comparisons,gates=gates,
        advance_development_gate=all(gates.values()),absolute=absolute,readouts=2000,trajectories=20,
        stream_arrivals=800,qualification_extra_arrivals=40,patient_independence=False,
        method_scope='GOTTA-inspired class-agnostic capacity-two max-min whole-image replay; source-stratified Bernoulli quantile distance; no CSTU/epsilon',
        interval_scope='2000 descriptive bootstrap draws over eight reused contents; conditions/orders bound; no independent patient inference'))
    save(root/'public/DIAGNOSTICS.json',diagnostics)
    for name,data in [('absolute',absolute),('condition_channel',channels),('transitions',transitions),('contrasts',comparisons),('update_audit',update_rows)]:csvout(root/'public'/f'{name}.csv',data)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',readouts=2000,evaluation_label_files=8,
        unique_metric_evaluations=len(cache),absolute_rows=25,reset_control_bit_parity=True,cpu_seconds=time.time()-start))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS' and c['source_sha']!='PREPARATION'
        sys.exit(controller(c))
    elif mode=='worker':worker(c,host_factory=Host)
    elif mode=='score':score(c)
    else:raise ValueError('unknown mode')
