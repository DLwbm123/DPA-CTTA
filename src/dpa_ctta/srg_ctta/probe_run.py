"""Frozen common-candidate probe-risk qualification; no online selection claim."""
import os
from pathlib import Path
import sys
import time
import numpy as np
from scipy.stats import rankdata
from .run import worker,controller,read,save,schedule,REFERENCES
from .probe_risk import Host
from .paired_state import summary
from .score import unpack
from ..r78_cross_patient.score import metric,csvout

PRIMARY='FRESH_RANK_UTILITY_MINUS_BEST_CONTROL'
POLICIES=('RESET','FRESH','RETAIN')
JOBS=[('ENT_NORM',p) for p in POLICIES]+[(a,p) for a in ('CW1','CW2') for p in ('RESET','RETAIN')]
CONTROLS=('pre_entropy','pre_disagreement','negative_extra_drift','negative_function_change','consistency_gain','full_utility','negative_pre_entropy','negative_pre_disagreement','negative_pre_anchor_drift')


def validate(c):
    assert c['design_version']=='PROBE_RISK_QUALIFICATION_V1' and c['primary']==PRIMARY
    assert c['core_arms']==c['normalized_arms']==['ENT_NORM']
    assert c['policies_by_arm']=={'ENT_NORM':list(POLICIES)}
    assert (c['expected_readouts'],c['core_arrivals'],c['native_arrivals'])==(1520,240,320)
    assert c['steps_per_arrival']==1 and c['strict_deterministic'] and c['repeat_first_reset'] and c['paired_state']
    assert c['qualification_arm']=='ENT_NORM' and c['probes']==4 and c['anchor_beta']==1.
    assert c['probe_gamma']==[.95,1.05] and c['probe_noise_std']==.005 and c['probe_seed_offset']==700000
    assert not any(c[k] for k in ('rdc','engram','new_training','save_probabilities','check_legacy_parity','reset_optimizer_per_arrival'))
    assert c['harm_margin_pp']==.2 and c['matched_step_l2']==.01


def correlation(x,y):
    a=rankdata(x);b=rankdata(y)
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def auc(score,harm):
    if not harm.any() or harm.all():return None
    ranks=rankdata(score)
    n=int(harm.sum());m=len(harm)-n
    return float((ranks[harm].sum()-n*(n+1)/2)/(n*m))


def evidence(groups,bootstrap=2000):
    # Reused contents are the resampling unit; conditions, order and heads stay bound.
    names=('utility',)+CONTROLS
    def values(rows):
        y=np.array([r['gain'] for r in rows]);ss={n:np.array([r[n] for r in rows]) for n in names}
        corr={n:correlation(ss[n],y) for n in names}
        valid=corr['utility'] is not None and all(corr[n] is not None for n in CONTROLS)
        primary=corr['utility']-max(corr[n] if corr[n] is not None else 0. for n in CONTROLS) if corr['utility'] is not None else None
        return primary,corr,{n:auc(-ss[n],y<-.2) for n in names},valid
    flat=[r for g in groups for r in g];primary,cc,aa,valid=values(flat)
    rng=np.random.default_rng(20261018);draws=[];ur=[];ua=[]
    for _ in range(bootstrap):
        rows=[r for i in rng.integers(0,8,8) for r in groups[i]]
        d,c,a,_=values(rows)
        if d is not None:draws.append(d)
        if c['utility'] is not None:ur.append(c['utility'])
        if a['utility'] is not None:ua.append(a['utility'])
    def ci(x):return np.quantile(x,[.025,.975]).tolist() if x else None
    # Fixed half-coverage, offline ranking only. Random expectation is the full mean.
    quota=[]
    for ch in range(2):
        rows=[r for r in flat if r['channel']==ch];count=len(rows)//2
        for n in names:
            chosen=sorted(rows,key=lambda r:r[n],reverse=True)[:count]
            quota.append(dict(channel=('OD','OC')[ch],score=n,selected=count,total=len(rows),mean_gain_pp=float(np.mean([r['gain'] for r in chosen])),harm_fraction=float(np.mean([r['gain']<-.2 for r in chosen])),tie_at_cutoff=sorted([r[n] for r in rows],reverse=True)[count-1]==sorted([r[n] for r in rows],reverse=True)[count]))
        quota.append(dict(channel=('OD','OC')[ch],score='RANDOM_EXPECTATION',selected=count,total=len(rows),mean_gain_pp=float(np.mean([r['gain'] for r in rows])),harm_fraction=float(np.mean([r['gain']<-.2 for r in rows])),tie_at_cutoff=False))
    head_corr={name:correlation([r['utility'] for r in flat if r['channel']==ch],[r['gain'] for r in flat if r['channel']==ch]) for ch,name in enumerate(('OD','OC'))}
    return dict(primary=primary,primary_ci95=ci(draws),correlations=cc,harm_auroc=aa,utility_ci95=ci(ur),utility_auroc_ci95=ci(ua),all_scores_nonconstant=valid,head_correlations=head_corr,n=len(flat),harmful=sum(r['gain']<-.2 for r in flat),positive_utility_fraction=float(np.mean([r['utility']>0 for r in flat])),bootstrap_valid=len(draws)),quota


def score(c):
    from ..source_io import read_mask
    validate(c);root=Path(c['output_root']);start=time.time()
    ret=read(root/'processes/worker.json');state=read(root/'RUN_STATE.json');repeat=read(root/'FULL_REPEATABILITY.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code']==0
    assert state['status']=='GPU_COMPLETE' and (state['predictions'],state['arrivals'])==(1520,560)
    assert repeat['status']=='PASS' and repeat['paired_readouts']==80
    rows=read(root/'private/predictions.json');updates=read(root/'private/updates.json');panel=read(root/'scorer/panel.json')
    def key(r):return r['case'],r['block'],r['order'],r['arm'],r['policy'],r['stage']
    expected={(r['case'],r['block'],o,a,p,s) for o in range(2) for r in schedule(o) for a,p in JOBS for s in ('PRE','POST')}
    expected|={(r['case'],r['block'],o,a,'REFERENCE','STATIC') for o in range(2) for r in schedule(o) for a in REFERENCES}
    assert len(rows)==len(expected)==1520 and {key(r) for r in rows}==expected and len(updates)==240 and len(panel)==8
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    sources={}
    for r in rows:
        if r['arm']=='SOURCE_EVAL':sources[r['case'],r['condition']]=unpack(root/'predictions'/f"{r['index']:04d}.bits")[1]
    cache={};grouped={};individual=[];lookup={}
    for r in rows:
        if time.time()-start>c['cpu_cap_seconds']-10:raise TimeoutError('CPU score cap')
        raw,pred=unpack(root/'predictions'/f"{r['index']:04d}.bits");k=r['case'],r['condition'],raw
        if k not in cache:cache[k]=metric(pred,labels[r['case']],sources[r['case'],r['condition']])
        mm,ratios=cache[k];item=dict(**r,metrics=mm,ratios=ratios);individual.append(item);lookup[key(r)]=item
        grouped.setdefault((r['arm'],r['policy'],r['stage']),[]).append(item)
    save(root/'scorer/individual.private.json',individual)
    def mean(v):
        a=[x for x in v if x is not None and np.isfinite(x)]
        return float(np.mean(a)) if a else None,len(a)
    absolute=[];cubes={};channels=[];transitions=[]
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
    risk=[];quotas=[];update_audit=[]
    for p in POLICIES:
        groups=[[] for _ in range(8)];rr=[r for r in updates if r['policy']==p];assert len(rr)==80
        for r in rr:
            pre=lookup[r['case'],r['block'],r['order'],'ENT_NORM',p,'PRE'];post=lookup[r['case'],r['block'],r['order'],'ENT_NORM',p,'POST']
            sig=r['probe_risk'];assert sig['valid_regions']==[2,2]
            assert r['optimizer_age']==(r['arrival']+1 if p=='RETAIN' else 1)
            for ch in range(2):
                groups[r['case']].append(dict(channel=ch,gain=post['metrics'][ch]['dice_percent']-pre['metrics'][ch]['dice_percent'],**{n:sig[n][ch] for n in ('utility',)+CONTROLS}))
        ev,quota=evidence(groups);risk.append(dict(policy=p,**ev));quotas.extend(dict(policy=p,**q) for q in quota)
        ss=[r['steps'][0] for r in rr];assert all(len(r['steps'])==1 for r in rr)
        update_audit.append(dict(policy=p,updates=80,matched=sum(x['norm_matched'] for x in ss),zero=sum(x['zero_proposal'] for x in ss),min_actual_step=min(x['actual_step_norm'] for x in ss),max_actual_step=max(x['actual_step_norm'] for x in ss)))
    def take(p,s='PRE'):return cubes['ENT_NORM',p,s]
    source=cubes['SOURCE_EVAL','REFERENCE','STATIC'];contrasts=[]
    for p in POLICIES:
        for s in ('PRE','POST'):contrasts.append(summary('ENT_'+p+'-'+s+'-SOURCE',take(p,s)-source))
        contrasts.append(summary('ENT_'+p+':RETURN_PRE-SOURCE',take(p)[:,4]-source[:,4]))
        contrasts.append(summary('ENT_'+p+':FIRST_PRE-SOURCE',take(p)[:,0]-source[:,0]))
        contrasts.append(summary('ENT_'+p+':POST-PRE',take(p,'POST')-take(p)))
    contrasts.append(summary('FRESH-RESET:POST',take('FRESH','POST')-take('RESET','POST')))
    fresh=next(v for v in risk if v['policy']=='FRESH')
    gates=dict(primary=fresh['primary'] is not None and fresh['primary']>=.1 and fresh['primary_ci95'][0]>0,
        utility_correlation=fresh['correlations']['utility'] is not None and fresh['correlations']['utility']>=.3 and fresh['utility_ci95'][0]>0,
        harmful_discrimination=fresh['harm_auroc']['utility'] is not None and fresh['harm_auroc']['utility']>=.7 and fresh['utility_auroc_ci95'][0]>.5,
        both_heads=all(x is not None and x>0 for x in fresh['head_correlations'].values()),
        matching=all(x['matched']==80 and x['zero']==0 for x in update_audit))
    result=dict(status='COMPLETE',primary=dict(comparison=PRIMARY,delta_correlation=fresh['primary'],ci95=fresh['primary_ci95']),risk=risk,gates=gates,advance_to_new_controller_protocol=all(gates.values()),online_controller_tested=False,continual_benefit_established=False,readouts=1520,scientific_arrivals=560,trajectories=14,absolute=absolute,contrasts=contrasts,scope='Same-image candidate-risk qualification only; no online or future benefit conclusion; eight reused contents')
    save(root/'public/RESULTS.json',result)
    for n,v in [('absolute',absolute),('condition_channel',channels),('transitions',transitions),('contrasts',contrasts),('offline_half_coverage',quotas),('update_audit',update_audit)]:csvout(root/'public'/f'{n}.csv',v)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',readouts=1520,evaluation_label_files=8,risk_head_events_per_policy=160,unique_metric_evaluations=len(cache),cpu_seconds=time.time()-start,online_controller_tested=False))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS' and c['source_sha']!='PREPARATION'
        sys.exit(controller(c))
    elif mode=='worker':worker(c,host_factory=Host)
    elif mode=='score':score(c)
    else:raise ValueError('unknown mode')
