"""New within-process deterministic state intervention; no historical scores used."""
import os
from pathlib import Path
import sys
import time
import numpy as np
from .run import worker,controller,read,save,schedule,REFERENCES
from .method import NORMALIZED
from ..r78_cross_patient.score import metric,interval,csvout
from .score import unpack

CORE=('STRAT','RANDOM','NOANCHOR','ENT_NORM','RG')
POLICIES={a:('RESET','RETAIN','FRESH') if a in ('STRAT','RANDOM') else ('RESET','RETAIN') for a in CORE}
JOBS=[(a,p) for a in CORE for p in POLICIES[a]]+[(a,p) for a in ('CW1','CW2') for p in ('RESET','RETAIN')]
PRIMARY='STRAT_FRESH-STRAT_RETAIN:PRE'


def validate(c):
    assert tuple(c['core_arms'])==CORE and {a:tuple(p) for a,p in c['policies_by_arm'].items()}==POLICIES
    assert c['paired_state'] and c['strict_deterministic'] and c['repeat_first_reset']
    assert not c['reset_optimizer_per_arrival'] and not c['check_legacy_parity']
    assert (c['expected_readouts'],c['core_arrivals'],c['native_arrivals'],c['expected_legacy_parity'])==(2960,960,320,0)
    assert c['primary']==PRIMARY


def summary(name,delta):
    if delta.shape not in ((8,5,2,2),(8,2,2)):raise ValueError('eight content groups with bound orders/channels required')
    if not np.isfinite(delta).all():raise ValueError('nonfinite paired contrast')
    return dict(comparison=name,delta_pp=float(delta.mean()),ci95_pp=interval(delta),worst_cell_pp=float(delta.min()),negative_cells=int((delta<0).sum()),total_cells=int(delta.size))


def score(c):
    from ..source_io import read_mask
    from scipy.ndimage import binary_erosion,distance_transform_edt
    root=Path(c['output_root']);start=time.time();retired=read(root/'processes/worker.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not retired['active'] and retired['exit_code']==0
    assert read(root/'RUN_STATE.json')['status']=='GPU_COMPLETE'
    assert read(root/'FULL_REPEATABILITY.json')['status']=='PASS'
    rows=read(root/'private/predictions.json');updates=read(root/'private/updates.json');panel=read(root/'scorer/panel.json')
    def key(r):return r['case'],r['block'],r['order'],r['arm'],r['policy'],r['stage']
    expected={(r['case'],r['block'],o,a,p,s) for o in range(2) for r in schedule(o) for a,p in JOBS for s in ('PRE','POST')}
    expected|={(r['case'],r['block'],o,a,'REFERENCE','STATIC') for o in range(2) for r in schedule(o) for a in REFERENCES}
    assert len(rows)==len(expected)==2960 and {key(r) for r in rows}==expected and len(updates)==960 and len(panel)==8
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    sources={}
    for r in rows:
        if r['arm']=='SOURCE_EVAL':sources[r['case'],r['condition']]=unpack(root/'predictions'/f"{r['index']:04d}.bits")[1]
    grouped={};cache={};individual=[]
    for r in rows:
        if time.time()-start>c['cpu_cap_seconds']-10:raise TimeoutError('paired scorer cap')
        raw,pred=unpack(root/'predictions'/f"{r['index']:04d}.bits");k=r['case'],r['condition'],raw
        if k not in cache:cache[k]=metric(pred,labels[r['case']],sources[r['case'],r['condition']])
        mm,ratios=cache[k];item=dict(**r,metrics=mm,ratios=ratios)
        individual.append(item);grouped.setdefault((r['arm'],r['policy'],r['stage']),[]).append(item)
    save(root/'scorer/individual.private.json',individual)
    cubes={};absolute=[];channels=[];transitions=[];diagnostics=[]
    def mean_defined(values):
        valid=[float(x) for x in values if x is not None and np.isfinite(x)]
        return (float(np.mean(valid)) if valid else None),len(valid)
    for (a,p,s),rr in grouped.items():
        assert len(rr)==80
        cube=np.full((8,5,2,2),np.nan)
        for r in rr:cube[r['case'],r['block'],r['order']]=[m['dice_percent'] for m in r['metrics']]
        assert np.isfinite(cube).all();cubes[a,p,s]=cube
        assd,defined=mean_defined([m['assd_pixel'] for r in rr for m in r['metrics']]);hd,_=mean_defined([m['hd95_pixel'] for r in rr for m in r['metrics']]);vcdr,vdefined=mean_defined([r['ratios']['vcdr_abs_error'] for r in rr])
        absolute.append(dict(arm=a,policy=p,stage=s,dice_percent=float(cube.mean()),assd_pixel=assd,hd95_pixel=hd,surface_defined=defined,surface_total=160,vcdr_mae=vcdr,vcdr_defined=vdefined,vcdr_total=80,empty_predictions=sum(m['pred_empty'] for r in rr for m in r['metrics']),nesting_violation_pixels=sum(r['nesting_violation_pixels'] for r in rr)))
        for b in range(5):
            for o in range(2):
                for ch,name in enumerate(('OD','OC')):channels.append(dict(arm=a,policy=p,stage=s,block=b,order=o,channel=name,dice_percent=float(cube[:,b,o,ch].mean())))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','boundary3px'):
                for measure in ('corrected_fraction','spoiled_fraction'):
                    value,n=mean_defined([r['metrics'][ch]['regions'][region][measure] for r in rr]);transitions.append(dict(arm=a,policy=p,stage=s,channel=name,region=region,measure=measure,macro_fraction=value,defined=n,total=80,reference='SOURCE_EVAL'))
        dd=dict(arm=a,policy=p,stage=s)
        for field in ('logit_rms','probability_abs_mean','changed_pixels','total_affine_norm','optimizer_age'):
            vals=[r['diagnostics'][field] for r in rr if field in r['diagnostics']]
            if vals:dd.update({field+'_mean':float(np.mean(vals)),field+'_min':float(np.min(vals)),field+'_max':float(np.max(vals))})
        diagnostics.append(dd)
    def take(a,p='RETAIN',s='PRE'):
        return cubes[a,'REFERENCE','STATIC'] if a in REFERENCES else cubes[a,p,s]
    contrasts=[]
    def add(n,d):
        out=summary(n,d);contrasts.append(out);return out
    primary=add(PRIMARY,take('STRAT','FRESH')-take('STRAT'))
    for a in ('STRAT','RANDOM'):
        for s in ('PRE','POST'):
            if (a,s)!=('STRAT','PRE'):add(a+'_FRESH-'+a+'_RETAIN:'+s,take(a,'FRESH',s)-take(a,'RETAIN',s))
    add('STRAT_STATE_EFFECT-RANDOM_STATE_EFFECT:PRE',(take('STRAT','FRESH')-take('STRAT'))-(take('RANDOM','FRESH')-take('RANDOM')))
    first=add('STRAT_FRESH-SOURCE_EVAL:FIRST_EXPOSURE_PRE',take('STRAT','FRESH')[:,0]-take('SOURCE_EVAL')[:,0])
    source=add('STRAT_FRESH-SOURCE_EVAL:ALL_PRE',take('STRAT','FRESH')-take('SOURCE_EVAL'))
    returned=add('STRAT_FRESH-SOURCE_EVAL:RETURN_PRE',take('STRAT','FRESH')[:,4]-take('SOURCE_EVAL')[:,4])
    random=add('STRAT_FRESH-RANDOM_FRESH:PRE',take('STRAT','FRESH')-take('RANDOM','FRESH'))
    for a,p,s in [(a,p,s) for a,p in JOBS for s in ('PRE','POST')]+[(a,'REFERENCE','STATIC') for a in REFERENCES]:
        if (a,p,s)!=('STRAT','FRESH','POST'):add('STRAT_FRESH_POST-'+a+'_'+p+'_'+s,take('STRAT','FRESH','POST')-take(a,p,s))
    add('STRAT_FRESH:RETURN_POST-FIRST_POST',take('STRAT','FRESH','POST')[:,4]-take('STRAT','FRESH','POST')[:,0])
    update_rows=[]
    for a,p in [(a,p) for a in CORE for p in POLICIES[a]]:
        rr=[r for r in updates if (r['arm'],r['policy'])==(a,p)];steps=[x for r in rr for x in r['steps']]
        assert len(rr)==80 and len(steps)==160
        expected_age=[2 if p in ('RESET','FRESH') else 2*(r['arrival']+1) for r in rr]
        assert [r['optimizer_age'] for r in rr]==expected_age
        out=dict(arm=a,policy=p,updates=160,matched=sum(x['norm_matched'] is True for x in steps),zero_proposals=sum(x['zero_proposal'] for x in steps))
        for f in ('raw_step_norm','actual_step_norm','direction_cosine','anchor','anchor_gradient_norm','gate_fraction'):
            v=[x[f] for x in steps if x[f] is not None]
            if v:out.update({f+'_mean':float(np.mean(v)),f+'_min':float(np.min(v)),f+'_max':float(np.max(v))})
        update_rows.append(out)
    gate_rows=[];boundary=[]
    for y in labels:
        surfaces=y^np.stack([binary_erosion(z,border_value=0) for z in y]);boundary.append(np.stack([distance_transform_edt(~z)<=3 if z.any() else np.zeros_like(z) for z in surfaces]))
    accum={}
    for r in updates:
        _,gate=unpack(root/'gates'/f"{r['index']:04d}.bits");_,pred=unpack(root/'gate_predictions'/f"{r['index']:04d}.bits");error=pred!=labels[r['case']]
        for ch,name in enumerate(('OD','OC')):
            for region,mask in [('all',np.ones_like(gate[ch])),('truth_boundary3px',boundary[r['case']][ch])]:
                selected=mask&gate[ch];outside=mask&~gate[ch];density=float(error[ch][mask].mean());excluded=float(error[ch][outside].mean()) if outside.any() else None
                accum.setdefault((r['arm'],r['policy'],name,region),[]).append(dict(selected_fraction=float(gate[ch][mask].mean()),outside_error_enrichment=excluded/density if excluded is not None and density else None))
    for (a,p,ch,region),rr in accum.items():
        out=dict(arm=a,policy=p,channel=ch,region=region,total=len(rr))
        for f in ('selected_fraction','outside_error_enrichment'):out[f],out[f+'_defined']=mean_defined([r[f] for r in rr])
        gate_rows.append(out)
    candidate=next(x for x in absolute if (x['arm'],x['policy'],x['stage'])==('STRAT','FRESH','POST'));baseline=next(x for x in absolute if x['arm']=='SOURCE_EVAL')
    def absrow(a,p='RETAIN',s='POST'):return next(x for x in absolute if (x['arm'],x['policy'],x['stage'])==(a,p,s))
    gates=dict(primary_mean=primary['delta_pp']>=.5,primary_ci=primary['ci95_pp'][0]>0,first_exposure_positive=first['ci95_pp'][0]>0,source_all_positive=source['ci95_pp'][0]>0,source_worst_no_harm=source['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,quota_random_pre_positive=random['ci95_pp'][0]>0,matched_norms=all(x['matched']==160 and x['zero_proposals']==0 for x in update_rows if x['arm'] in NORMALIZED),strong_post_frontier=all(candidate['dice_percent']>x for x in [baseline['dice_percent'],absrow('RG')['dice_percent'],absrow('CW1')['dice_percent'],absrow('CW2')['dice_percent']]),no_new_empty=candidate['empty_predictions']<=baseline['empty_predictions'],no_new_nesting=candidate['nesting_violation_pixels']<=baseline['nesting_violation_pixels'],structure_and_coverage=all(candidate[f] is not None and candidate[f]<=baseline[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')) and candidate['surface_defined']>=baseline['surface_defined'] and candidate['vcdr_defined']>=baseline['vcdr_defined'])
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,gates=gates,advance_development_gate=all(gates.values()),readouts=2960,trajectories=32,stream_arrivals=1280,qualification_extra_arrivals=40,strict_deterministic=True,historical_predictions_used=False,patient_independence=False,interval_scope='Descriptive paired bootstrap of eight reused contents binding conditions, orders and channels',absolute=absolute))
    save(root/'public/DIAGNOSTICS.json',diagnostics)
    for name,rr in [('absolute',absolute),('condition_channel',channels),('transitions',transitions),('contrasts',contrasts),('update_audit',update_rows),('gate_audit',gate_rows)]:csvout(root/'public'/f'{name}.csv',rr)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',readouts=2960,evaluation_label_files=8,unique_metric_evaluations=len(cache),cpu_seconds=time.time()-start))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='controller':sys.exit(controller(c))
    elif mode=='worker':worker(c)
    elif mode=='score':score(c)
    else:raise ValueError('unknown mode')
