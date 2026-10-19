"""One frozen dense masked-calibration comparison and independent CPU scoring."""
import os
from pathlib import Path
import sys
import time
import numpy as np
from .run import worker,controller,read,save,schedule,REFERENCES
from .masked_calibration import Host,KINDS
from .paired_state import summary
from ..r78_cross_patient.score import metric,csvout
from .score import unpack

POLICIES={'MARS':('RESET','FRESH'),'MASK_HARD':('FRESH',),'FIXED_RHO':('FRESH',),'ENT_NORM':('RESET','RETAIN','FRESH')}
JOBS=[(a,p) for a,pp in POLICIES.items() for p in pp]+[(a,p) for a in ('CW1','CW2') for p in ('RESET','RETAIN')]
PRIMARY='MARS_FRESH-FIXED_RHO_FRESH:PRE'


def validate(c):
    assert c['design_version']=='MASKED_CALIBRATION_V1' and c['primary']==PRIMARY
    assert tuple(c['core_arms'])==KINDS and {a:tuple(p) for a,p in c['policies_by_arm'].items()}==POLICIES
    assert tuple(c['normalized_arms'])==KINDS and c['qualification_arm']=='MARS'
    assert (c['expected_readouts'],c['core_arrivals'],c['native_arrivals'],c['masked_arrivals'],c['plain_arrivals'])==(2160,560,320,320,240)
    assert c['steps_per_arrival']==1 and c['masked_views']==2 and c['rank_weight']==.2 and c['fixed_softness']==.3
    assert c['strict_deterministic'] and c['repeat_first_reset'] and c['paired_state'] and c['save_probabilities']
    assert not c['check_legacy_parity'] and not c['reset_optimizer_per_arrival']
    assert not any(c[k] for k in ('rdc','engram','new_training'))


def calibration(q,pred,truth,region):
    if not region.any():return None
    probability=q[region].astype(np.float64)/255
    wrong=(pred[region]!=truth[region])
    confidence=np.maximum(probability,1-probability)
    bins=np.minimum((confidence*10).astype(int),9)
    count=np.bincount(bins,minlength=10)
    correct=np.bincount(bins,weights=~wrong,minlength=10)
    total_conf=np.bincount(bins,weights=confidence,minlength=10)
    return dict(brier=float(np.square(probability-truth[region]).mean()),
                ece=float(np.abs(correct-total_conf).sum()/count.sum()),pixels=int(count.sum()))


def score(c):
    from ..source_io import read_mask
    from scipy.ndimage import binary_dilation
    validate(c);root=Path(c['output_root']);started=time.time()
    retired=read(root/'processes/worker.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and retired['exit_code']==0 and not retired['active']
    state=read(root/'RUN_STATE.json');repeat=read(root/'FULL_REPEATABILITY.json')
    assert state['status']=='GPU_COMPLETE' and (state['predictions'],state['arrivals'])==(2160,880)
    assert repeat['status']=='PASS' and repeat['arrivals']==40 and repeat['paired_readouts']==80
    rows=read(root/'private/predictions.json');updates=read(root/'private/updates.json')
    panel=read(root/'scorer/panel.json');assert len(panel)==8
    def key(r):return r['case'],r['block'],r['order'],r['arm'],r['policy'],r['stage']
    expected={(r['case'],r['block'],o,a,p,s) for o in range(2) for r in schedule(o) for a,p in JOBS for s in ('PRE','POST')}
    expected|={(r['case'],r['block'],o,a,'REFERENCE','STATIC') for o in range(2) for r in schedule(o) for a in REFERENCES}
    assert len(rows)==len(expected)==2160 and {key(r) for r in rows}==expected and len(updates)==560
    assert len(list((root/'probabilities').glob('*.q8')))==2160
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    sources={}
    for r in rows:
        if r['arm']=='SOURCE_EVAL':sources[r['case'],r['condition']]=unpack(root/'predictions'/f"{r['index']:04d}.bits")[1]
    cache={};grouped={};cal=[];individual=[]
    for r in rows:
        if time.time()-started>c['cpu_cap_seconds']-10:raise TimeoutError('CPU scoring cap')
        raw,pred=unpack(root/'predictions'/f"{r['index']:04d}.bits")
        k=r['case'],r['condition'],raw
        if k not in cache:cache[k]=metric(pred,labels[r['case']],sources[r['case'],r['condition']])
        mm,ratios=cache[k];item=dict(**r,metrics=mm,ratios=ratios)
        individual.append(item);grouped.setdefault((r['arm'],r['policy'],r['stage']),[]).append(item)
        q=np.frombuffer((root/'probabilities'/f"{r['index']:04d}.q8").read_bytes(),dtype=np.uint8).reshape(2,512,512)
        for ch in range(2):
            roi=binary_dilation(sources[r['case'],r['condition']][ch],iterations=3)
            for name,mask in [('all',np.ones_like(roi)),('source_foreground_plus3px',roi)]:
                values=calibration(q[ch],pred[ch],labels[r['case']][ch],mask)
                cal.append(dict(arm=r['arm'],policy=r['policy'],stage=r['stage'],channel=ch,region=name,**(values or dict(brier=None,ece=None,pixels=0))))
    save(root/'scorer/individual.private.json',individual)
    def defined(values):
        v=[float(x) for x in values if x is not None and np.isfinite(x)]
        return (float(np.mean(v)) if v else None),len(v)
    cubes={};absolute=[];channels=[];transitions=[];diagnostics=[]
    for (a,p,s),rr in grouped.items():
        assert len(rr)==80
        cube=np.full((8,5,2,2),np.nan)
        for r in rr:cube[r['case'],r['block'],r['order']]=[m['dice_percent'] for m in r['metrics']]
        assert np.isfinite(cube).all();cubes[a,p,s]=cube
        assd,n=defined([m['assd_pixel'] for r in rr for m in r['metrics']]);hd,_=defined([m['hd95_pixel'] for r in rr for m in r['metrics']]);vcdr,vn=defined([r['ratios']['vcdr_abs_error'] for r in rr])
        absolute.append(dict(arm=a,policy=p,stage=s,dice_percent=float(cube.mean()),assd_pixel=assd,hd95_pixel=hd,surface_defined=n,surface_total=160,vcdr_mae=vcdr,vcdr_defined=vn,vcdr_total=80,empty_predictions=sum(m['pred_empty'] for r in rr for m in r['metrics']),nesting_violation_pixels=sum(r['nesting_violation_pixels'] for r in rr)))
        for b in range(5):
            for o in range(2):
                for ch,name in enumerate(('OD','OC')):channels.append(dict(arm=a,policy=p,stage=s,block=b,order=o,channel=name,dice_percent=float(cube[:,b,o,ch].mean())))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','boundary3px'):
                for measure in ('corrected_fraction','spoiled_fraction'):
                    value,n=defined([r['metrics'][ch]['regions'][region][measure] for r in rr]);transitions.append(dict(arm=a,policy=p,stage=s,channel=name,region=region,measure=measure,macro_fraction=value,defined=n,total=80,reference='SOURCE_EVAL'))
        d=dict(arm=a,policy=p,stage=s)
        for field in ('logit_rms','probability_abs_mean','changed_pixels','total_affine_norm','optimizer_age'):
            v=[r['diagnostics'][field] for r in rr if field in r['diagnostics']]
            if v:d.update({field+'_mean':float(np.mean(v)),field+'_min':float(np.min(v)),field+'_max':float(np.max(v))})
        diagnostics.append(d)
    def take(a,p='FRESH',s='PRE'):return cubes[a,'REFERENCE','STATIC'] if a in REFERENCES else cubes[a,p,s]
    contrasts=[]
    def add(name,d):
        out=summary(name,d);contrasts.append(out);return out
    primary=add(PRIMARY,take('MARS')-take('FIXED_RHO'))
    first=add('MARS_FRESH-SOURCE:FIRST_PRE',take('MARS')[:,0]-take('SOURCE_EVAL')[:,0])
    source=add('MARS_FRESH-SOURCE:ALL_PRE',take('MARS')-take('SOURCE_EVAL'))
    returned=add('MARS_FRESH-SOURCE:RETURN_PRE',take('MARS')[:,4]-take('SOURCE_EVAL')[:,4])
    hard=add('MARS_FRESH-MASK_HARD_FRESH:PRE',take('MARS')-take('MASK_HARD'))
    ent=add('MARS_FRESH-ENT_FRESH:PRE',take('MARS')-take('ENT_NORM'))
    for a,p in JOBS:
        for s in ('PRE','POST'):
            if (a,p,s)!=('MARS','FRESH','POST'):add('MARS_FRESH_POST-'+a+'_'+p+'_'+s,take('MARS',s='POST')-take(a,p,s))
    for a in REFERENCES:add('MARS_FRESH_POST-'+a,take('MARS',s='POST')-take(a))
    update_rows=[]
    for a,pp in POLICIES.items():
        for p in pp:
            rr=[r for r in updates if (r['arm'],r['policy'])==(a,p)];assert len(rr)==80 and all(len(r['steps'])==1 for r in rr)
            assert [r['optimizer_age'] for r in rr]==[1 if p in ('RESET','FRESH') else r['arrival']+1 for r in rr]
            ss=[r['steps'][0] for r in rr]
            out=dict(arm=a,policy=p,updates=80,matched=sum(x['norm_matched'] for x in ss),zero=sum(x['zero_proposal'] for x in ss))
            for f in ('raw_step_norm','actual_step_norm','direction_cosine','ranking','gate_fraction'):
                v=[x[f] for x in ss];out.update({f+'_mean':float(np.mean(v)),f+'_min':float(np.min(v)),f+'_max':float(np.max(v))})
            update_rows.append(out)
    calibration_rows=[]
    for a,p,s in grouped:
        for ch in range(2):
            for region in ('all','source_foreground_plus3px'):
                rr=[r for r in cal if (r['arm'],r['policy'],r['stage'],r['channel'],r['region'])==(a,p,s,ch,region)]
                assert len(rr)==80
                out=dict(arm=a,policy=p,stage=s,channel=('OD','OC')[ch],region=region,total=80)
                for f in ('brier','ece'):out[f],out[f+'_defined']=defined([r[f] for r in rr])
                calibration_rows.append(out)
    def absolute_row(a,p='FRESH',s='POST'):return next(x for x in absolute if (x['arm'],x['policy'],x['stage'])==(a,p,s))
    candidate=absolute_row('MARS');baseline=absolute_row('SOURCE_EVAL','REFERENCE','STATIC')
    gates=dict(primary_mean=primary['delta_pp']>=.5,primary_ci=primary['ci95_pp'][0]>0,hard_control=hard['ci95_pp'][0]>0,entropy_control=ent['ci95_pp'][0]>0,
        first_pre=first['ci95_pp'][0]>0,all_pre=source['ci95_pp'][0]>0,worst_pre=source['worst_cell_pp']>=-1,return_pre=returned['delta_pp']>=-.2,
        actual_step_matching=all(x['matched']==80 and x['zero']==0 for x in update_rows),
        strong_post_frontier=all(candidate['dice_percent']>absolute_row(a,p,s)['dice_percent'] for a,p,s in [(a,'REFERENCE','STATIC') for a in REFERENCES]+[('ENT_NORM','FRESH','POST'),('MARS','RESET','POST'),('CW1','RETAIN','POST'),('CW2','RETAIN','POST'),('CW1','RESET','POST'),('CW2','RESET','POST')]),
        no_empty_or_nesting=candidate['empty_predictions']<=baseline['empty_predictions'] and candidate['nesting_violation_pixels']<=baseline['nesting_violation_pixels'],
        structure_and_coverage=all(candidate[f] is not None and candidate[f]<=baseline[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')) and candidate['surface_defined']>=baseline['surface_defined'] and candidate['vcdr_defined']>=baseline['vcdr_defined'])
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,gates=gates,advance_development_gate=all(gates.values()),absolute=absolute,readouts=2160,trajectories=22,stream_arrivals=880,qualification_extra_arrivals=40,patient_independence=False,method_scope='MARS ranking-loss transfer, K2, one matched step; not full original MARS',interval_scope='Descriptive paired bootstrap over eight reused contents binding order, condition and channel'))
    save(root/'public/DIAGNOSTICS.json',diagnostics)
    for name,data in [('absolute',absolute),('condition_channel',channels),('transitions',transitions),('contrasts',contrasts),('update_audit',update_rows),('calibration_q8',calibration_rows)]:csvout(root/'public'/f'{name}.csv',data)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',readouts=2160,evaluation_label_files=8,unique_metric_evaluations=len(cache),probability_bits=8,calibration_scope='per-image macro, q8 approximation, full image and source foreground plus3px',cpu_seconds=time.time()-started))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='controller':
        root=Path(c['output_root'])
        assert read(root/'CPU_CHECK.json')['status']=='PASS' and c['source_sha']!='PREPARATION'
        sys.exit(controller(c))
    elif mode=='worker':worker(c,host_factory=Host)
    elif mode=='score':score(c)
    else:raise ValueError('unknown mode')
