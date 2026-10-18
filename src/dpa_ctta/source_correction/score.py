"""Independent CPU scoring of a complete retired-worker prediction matrix."""
import os
from pathlib import Path
import time
import numpy as np
from .run import ARMS,DIRECT,MODES,read,save
from ..r78_cross_patient.score import metric,interval,csvout


def score(c):
    from ..source_io import read_mask
    root=Path(c['output_root']);started=time.time()
    retirement=read(root/'processes/worker.json');state=read(root/'RUN_STATE.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or retirement['active'] or retirement['exit_code']:
        raise ValueError('retired successful GPU worker and CPU-only scoring required')
    if state['status']!='GPU_COMPLETE' or state['predictions']!=672 or state['baseline_parity']!=448:
        raise ValueError('complete prediction/parity coverage required')
    audit=read(root/'MECHANISM_AUDIT.json')
    if audit['status']!='PASS': raise ValueError('mechanism qualification failed')
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json')
    expected={(p,k,a,m) for p in range(8) for k in range(4) for a in ARMS for m in (('DIRECT',) if a in DIRECT else MODES)}
    keys=[(r['pair'],r['condition'],r['arm'],r['mode']) for r in rows]
    if len(rows)!=672 or len(set(keys))!=672 or set(keys)!=expected or len(panel)!=8: raise ValueError('fixed matrix coverage')
    masks={}
    for r,key in zip(rows,keys):
        raw=(root/'predictions'/f'{r["index"]:04d}.bits').read_bytes()
        if len(raw)!=65536: raise ValueError('invalid packed prediction')
        masks[key]=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    shape=(8,4,len(ARMS),3,2)
    dice=np.zeros(shape);surface=np.full(shape,np.nan);hd=surface.copy();empty=np.zeros(shape,dtype=bool);vcdr=np.full(shape[:-1],np.nan)
    individual=[];groups={}
    for r,key in zip(rows,keys):
        if time.time()-started>c['cpu_cap_seconds']-10: raise TimeoutError('fixed scorer cap')
        p,k,a,m=key;ai=ARMS.index(a)
        metrics,ratios=metric(masks[key],labels[p],masks[p,k,'B0','DIRECT'])
        for mi in (range(3) if a in DIRECT else [MODES.index(m)]):
            dice[p,k,ai,mi]=[v['dice_percent'] for v in metrics]
            surface[p,k,ai,mi]=[v['assd_pixel'] if v['assd_pixel'] is not None else np.nan for v in metrics]
            hd[p,k,ai,mi]=[v['hd95_pixel'] if v['hd95_pixel'] is not None else np.nan for v in metrics]
            empty[p,k,ai,mi]=[v['pred_empty'] for v in metrics]
            vcdr[p,k,ai,mi]=ratios['vcdr_abs_error'] if ratios['vcdr_abs_error'] is not None else np.nan
        individual.append(dict(**r,metrics=metrics,ratios=ratios));groups.setdefault((a,m),[]).append(r['diagnostics'])
    save(root/'scorer/individual_metrics.private.json',individual)
    np.save(root/'scorer/dice.private.npy',dice)
    def mean(x): return float(np.nanmean(x)) if np.isfinite(x).any() else None
    absolute=[];cells=[];transitions=[];diagnostics=[]
    for ai,a in enumerate(ARMS):
        for mi,m in enumerate(MODES):
            if a in DIRECT and mi: continue
            label='DIRECT' if a in DIRECT else m
            absolute.append(dict(arm=a,mode=label,macro_dice_percent=float(dice[:,:,ai,mi].mean()),assd_pixel=mean(surface[:,:,ai,mi]),hd95_pixel=mean(hd[:,:,ai,mi]),surface_defined=int(np.isfinite(surface[:,:,ai,mi]).sum()),surface_total=64,vcdr_mae=mean(vcdr[:,:,ai,mi]),vcdr_defined=int(np.isfinite(vcdr[:,:,ai,mi]).sum()),vcdr_total=32,empty_predictions=int(empty[:,:,ai,mi].sum())))
            for k in range(4):
                for ch,channel in enumerate(('OD','OC')): cells.append(dict(arm=a,mode=label,condition=f'C{k}',channel=channel,macro_dice_percent=float(dice[:,k,ai,mi,ch].mean())))
            dd=groups[a,label];out=dict(arm=a,mode=label)
            for field in ('logit_rms','probability_abs_mean','changed_pixels','maximum_logit_change','nesting_violation_pixels','rms_match_target','rms_match_actual','rms_match_factor','norm','actual_norm'):
                numbers=[d[field] for d in dd if d.get(field) is not None]
                if numbers: out.update({field+'_mean':float(np.mean(numbers)),field+'_min':float(np.min(numbers)),field+'_max':float(np.max(numbers))})
            diagnostics.append(out)
        selected=[r for r in individual if r['arm']==a and r['mode'] in ('DIRECT','CROSS_MATCH')]
        for ch,channel in enumerate(('OD','OC')):
            for region in ('all','foreground','boundary3px'):
                for measure in ('corrected_fraction','spoiled_fraction'):
                    values=[r['metrics'][ch]['regions'][region][measure] for r in selected];valid=[v for v in values if v is not None]
                    transitions.append(dict(arm=a,channel=channel,region=region,measure=measure,macro_fraction=float(np.mean(valid)) if valid else None,defined=len(valid),total=len(values)))
    contrasts=[]
    def contrast(a,b,mode='CROSS_MATCH'):
        mi=MODES.index(mode);delta=dice[:,:,ARMS.index(a),mi]-dice[:,:,ARMS.index(b),mi]
        out=dict(comparison=a+'-'+b,mode=mode,delta_pp=float(delta.mean()),ci95_pp=interval(delta),condition_delta_pp=delta.mean((0,2)).tolist(),worst_content_condition_channel_pp=float(delta.min()),negative_cells=int((delta<0).sum()),total_cells=64)
        contrasts.append(out);return out
    for a in ARMS:
        if a!='R':
            for m in (MODES if a not in DIRECT else ('CROSS_MATCH',)): contrast('R',a,m)
    budget=contrast('R_BUDGET','D_RETRAIN');repeat=contrast('D_RETRAIN','D_OLD')
    primary=next(x for x in contrasts if x['comparison']=='R-D_RETRAIN')
    get=lambda name:next(x for x in contrasts if x['comparison']=='R-'+name and x['mode']=='CROSS_MATCH')
    x=ARMS.index('R');b=ARMS.index('B0');d=ARMS.index('D_RETRAIN');delta=dice[:,:,x,1]-dice[:,:,b,1]
    gates=dict(primary_mean=primary['delta_pp']>=.5,primary_ci=primary['ci95_pp'][0]>0,image_increment=get('R_GEOM')['ci95_pp'][0]>0,support_increment=get('R_BAND')['ci95_pp'][0]>0,budget_increment=audit['function_budget_exact']==32 and budget['ci95_pp'][0]>0,all_strong_controls_positive=all(get(a)['delta_pp']>0 for a in ('D_OLD','F','X_MATCH','P_MATCH','C_FULL')),original_no_harm=float(delta[:,0].mean())>=-.2,worst_no_harm=float(delta.min())>=-1,no_new_empty=not bool((empty[:,:,x,1]&~empty[:,:,b,1]).any()),surface_coverage=int(np.isfinite(surface[:,:,x,1]).sum())>=int(np.isfinite(surface[:,:,b,1]).sum()),vcdr_coverage=int(np.isfinite(vcdr[:,:,x,1]).sum())>=int(np.isfinite(vcdr[:,:,b,1]).sum()),structure_direction=mean(surface[:,:,x,1])<=mean(surface[:,:,b,1]) and mean(hd[:,:,x,1])<=mean(hd[:,:,b,1]) and mean(vcdr[:,:,x,1])<=mean(vcdr[:,:,b,1]))
    nesting={v['arm']:v.get('nesting_violation_pixels_mean',0.) for v in diagnostics if v['mode']=='DIRECT'}
    gates['no_new_nesting']=nesting['R']<=nesting['B0']
    result=dict(status='COMPLETE',final_readouts=672,baseline_parity=448,training_episodes=800,new_training=True,primary=primary,budget_comparison=budget,retraining_check=repeat,gates=gates,advance_development_gate=all(gates.values()),interval_scope='Descriptive bootstrap of eight previously exposed content pairs; patient dependence unknown; one training seed',patient_independence=False,query_development_exposed=True,absolute=absolute,contrasts=contrasts)
    save(root/'public/RESULTS.json',result);save(root/'public/DIAGNOSTICS.json',diagnostics)
    for name,values in [('absolute',absolute),('condition_channel',cells),('transitions',transitions),('contrasts',contrasts)]: csvout(root/'public'/f'{name}.csv',values)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',final_readouts=672,cpu_seconds=time.time()-started,final_label_files=8))
