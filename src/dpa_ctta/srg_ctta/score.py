"""Post-retirement metrics and gate audit; no feedback into adaptation."""
import os
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import binary_erosion,distance_transform_edt
from .run import ARMS,NORMALIZED,LEARNERS,POLICIES,STAGES,REFERENCES,BLOCKS,EXPECTED,schedule,read,save
from ..r78_cross_patient.score import metric,interval,csvout


def unpack(path):
    raw=path.read_bytes()
    if len(raw)!=65536:raise ValueError('packed prediction size')
    return raw,np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool)


def score(c):
    from ..source_io import read_mask
    root=Path(c['output_root']);started=time.time();state=read(root/'RUN_STATE.json');retired=read(root/'processes/worker.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or retired['active'] or retired['exit_code'] or state['status']!='GPU_COMPLETE':raise ValueError('successful retired worker required')
    rows=read(root/'private/predictions.json');updates=read(root/'private/updates.json');panel=read(root/'scorer/panel.json');audit=read(root/'MECHANISM_AUDIT.json')
    def key(r):return r['case'],r['block'],r['order'],r['arm'],r['policy'],r['stage']
    expected={(r['case'],r['block'],o,a,p,s) for o in range(2) for r in schedule(o) for a in LEARNERS for p in POLICIES for s in STAGES}
    expected|={(r['case'],r['block'],o,a,'REFERENCE','STATIC') for o in range(2) for r in schedule(o) for a in REFERENCES}
    if len(rows)!=EXPECTED or len({key(r) for r in rows})!=EXPECTED or {key(r) for r in rows}!=expected or len(updates)!=1280 or len(panel)!=8 or audit['legacy_parity']!=128:raise ValueError('fixed matrix coverage failed')
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    bykey={key(r):r for r in rows};bases={}
    for r in rows:
        if r['arm']=='CURRENT_BN':bases[r['case'],r['condition']]=unpack(root/'predictions'/f'{r["index"]:04d}.bits')[1]
    shape=(8,5,2,len(LEARNERS),2,2,2)
    dice=np.zeros(shape);assd=np.full(shape,np.nan);hd=assd.copy();empty=np.zeros(shape,dtype=bool);vcdr=np.full(shape[:-1],np.nan)
    rshape=(8,5,2,len(REFERENCES),2)
    rdice=np.zeros(rshape);rassd=np.full(rshape,np.nan);rhd=rassd.copy();rempty=np.zeros(rshape,dtype=bool);rvcdr=np.full(rshape[:-1],np.nan)
    cache={};individual=[];grouped={}
    for r in rows:
        if time.time()-started>c['cpu_cap_seconds']-10:raise TimeoutError('fixed CPU scorer cap')
        raw,pred=unpack(root/'predictions'/f'{r["index"]:04d}.bits');ck=(r['case'],r['condition'],raw)
        if ck not in cache:cache[ck]=metric(pred,labels[r['case']],bases[r['case'],r['condition']])
        mm,ratios=cache[ck];d=[m['dice_percent'] for m in mm];a=[m['assd_pixel'] if m['assd_pixel'] is not None else np.nan for m in mm];h=[m['hd95_pixel'] if m['hd95_pixel'] is not None else np.nan for m in mm];e=[m['pred_empty'] for m in mm];v=ratios['vcdr_abs_error'] if ratios['vcdr_abs_error'] is not None else np.nan
        idx=(r['case'],r['block'],r['order'])
        if r['arm'] in REFERENCES:
            idx+=REFERENCES.index(r['arm']),
            rdice[idx]=d;rassd[idx]=a;rhd[idx]=h;rempty[idx]=e;rvcdr[idx]=v
        else:
            idx+=(LEARNERS.index(r['arm']),POLICIES.index(r['policy']),STAGES.index(r['stage']))
            dice[idx]=d;assd[idx]=a;hd[idx]=h;empty[idx]=e;vcdr[idx]=v
        individual.append(dict(**r,metrics=mm,ratios=ratios));grouped.setdefault((r['arm'],r['policy'],r['stage']),[]).append(individual[-1])
    save(root/'scorer/individual.private.json',individual);np.save(root/'scorer/dice.private.npy',dice)
    def mean(x):return float(np.nanmean(x)) if np.isfinite(x).any() else None
    def take(a,policy='RETAIN',stage='POST',array=dice,refarray=rdice):
        return refarray[:,:,:,REFERENCES.index(a)] if a in REFERENCES else array[:,:,:,LEARNERS.index(a),POLICIES.index(policy),STAGES.index(stage)]
    absolute=[];channels=[];transitions=[];diagnostics=[]
    groups=[(a,p,s) for a in LEARNERS for p in POLICIES for s in STAGES]+[(a,'REFERENCE','STATIC') for a in REFERENCES]
    for a,p,s in groups:
        d=take(a,p,s);aa=take(a,p,s,assd,rassd);hh=take(a,p,s,hd,rhd);ee=take(a,p,s,empty,rempty);vv=take(a,p,s,vcdr,rvcdr);rr=grouped[a,p,s]
        absolute.append(dict(arm=a,policy=p,stage=s,dice_percent=float(d.mean()),assd_pixel=mean(aa),hd95_pixel=mean(hh),surface_defined=int(np.isfinite(aa).sum()),surface_total=160,vcdr_mae=mean(vv),vcdr_defined=int(np.isfinite(vv).sum()),vcdr_total=80,empty_predictions=int(ee.sum()),nesting_violation_pixels=sum(r['nesting_violation_pixels'] for r in rr)))
        for block,k in enumerate(BLOCKS):
            for order in range(2):
                for ch,name in enumerate(('OD','OC')):channels.append(dict(arm=a,policy=p,stage=s,block=block,condition=k,order=order,channel=name,dice_percent=float(d[:,block,order,ch].mean())))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','boundary3px'):
                for measure in ('corrected_fraction','spoiled_fraction'):
                    valid=[r['metrics'][ch]['regions'][region][measure] for r in rr if r['metrics'][ch]['regions'][region][measure] is not None]
                    transitions.append(dict(arm=a,policy=p,stage=s,channel=name,region=region,measure=measure,macro_fraction=float(np.mean(valid)) if valid else None,defined=len(valid),total=len(rr)))
        dd=dict(arm=a,policy=p,stage=s)
        for field in ('logit_rms','probability_abs_mean','changed_pixels','total_affine_norm','optimizer_age'):
            values=[r['diagnostics'][field] for r in rr if field in r['diagnostics']]
            if values:dd.update({field+'_mean':float(np.mean(values)),field+'_min':float(np.min(values)),field+'_max':float(np.max(values))})
        diagnostics.append(dd)
    contrasts=[]
    def summary(name,delta):
        x=dict(comparison=name,delta_pp=float(delta.mean()),ci95_pp=interval(delta),worst_cell_pp=float(delta.min()),negative_cells=int((delta<0).sum()),total_cells=int(delta.size));contrasts.append(x);return x
    primary=summary('STRAT-RANDOM:RETAIN_POST',take('STRAT')-take('RANDOM'))
    for control in LEARNERS+REFERENCES:
        if control not in ('STRAT','RANDOM'):summary('STRAT-'+control+':RETAIN_POST',take('STRAT')-take(control))
    prequential=summary('STRAT_RETAIN_PRE-STRAT_RESET_PRE',take('STRAT','RETAIN','PRE')-take('STRAT','RESET','PRE'))
    first=summary('STRAT_RETAIN_PRE-STRAT_RESET_PRE:FIRST_EXPOSURE',take('STRAT','RETAIN','PRE')[:,0]-take('STRAT','RESET','PRE')[:,0])
    pre_vs_entropy=summary('STRAT-ENT_NORM:RETAIN_PRE',take('STRAT','RETAIN','PRE')-take('ENT_NORM','RETAIN','PRE'))
    for a in LEARNERS:summary(a+':RETURN_C0_POST-FIRST_C0_POST',take(a)[:,4]-take(a)[:,0])
    anchor_return=summary('STRAT-NOANCHOR:RETURN_C0_PRE',take('STRAT','RETAIN','PRE')[:,4]-take('NOANCHOR','RETAIN','PRE')[:,4])
    # Gate errors are measured at the exact pre-second-update model that generated
    # the saved gate, not at an unrelated PRE or POST stream readout.
    gate_rows=[]
    boundaries=[]
    for y in labels:
        surfaces=y^np.stack([binary_erosion(v,border_value=0) for v in y])
        boundaries.append(np.stack([distance_transform_edt(~v)<=3 if v.any() else np.zeros_like(v) for v in surfaces]))
    for r in updates:
        _,gate=unpack(root/'gates'/f'{r["index"]:04d}.bits');_,pred=unpack(root/'gate_predictions'/f'{r["index"]:04d}.bits');error=pred!=labels[r['case']]
        for ch,name in enumerate(('OD','OC')):
            for region,mask in (('all',np.ones_like(gate[ch])),('truth_boundary3px',boundaries[r['case']][ch])):
                excluded=mask&~gate[ch];selected=mask&gate[ch]
                density=float(error[ch][mask].mean()) if mask.any() else None
                outside=float(error[ch][excluded].mean()) if excluded.any() else None
                inside=float(error[ch][selected].mean()) if selected.any() else None
                gate_rows.append(dict(arm=r['arm'],policy=r['policy'],channel=name,region=region,case=r['case'],selected_fraction=float(gate[ch][mask].mean()) if mask.any() else None,outside_error_density=outside,inside_error_density=inside,outside_error_enrichment=outside/density if outside is not None and density else None))
    gate_aggregate=[]
    for a in ARMS:
        for p in POLICIES:
            for ch in ('OD','OC'):
                for region in ('all','truth_boundary3px'):
                    rr=[r for r in gate_rows if (r['arm'],r['policy'],r['channel'],r['region'])==(a,p,ch,region)];out=dict(arm=a,policy=p,channel=ch,region=region,total=len(rr))
                    for field in ('selected_fraction','outside_error_density','inside_error_density','outside_error_enrichment'):
                        valid=[v[field] for v in rr if v[field] is not None];out[field]=float(np.mean(valid)) if valid else None;out[field+'_defined']=len(valid)
                    gate_aggregate.append(out)
    step_summary=[]
    for a in ARMS:
        for p in POLICIES:
            rr=[v for r in updates if r['arm']==a and r['policy']==p for v in r['steps']]
            out=dict(arm=a,policy=p,updates=len(rr),matched_norm_updates=sum(v['norm_matched'] is True for v in rr),zero_proposals=sum(v['zero_proposal'] for v in rr))
            for field in ('entropy','consistency','boundary','anchor','anchor_gradient_norm','raw_step_norm','actual_step_norm','gate_fraction'):
                values=[r[field] for r in rr];out.update({field+'_mean':float(np.mean(values)),field+'_min':float(np.min(values)),field+'_max':float(np.max(values))})
            step_summary.append(out)
    q=take('STRAT')-take('CURRENT_BN');baseline_empty=take('CURRENT_BN',array=empty,refarray=rempty)
    row=next(r for r in absolute if (r['arm'],r['policy'],r['stage'])==('STRAT','RETAIN','POST'));base_row=next(r for r in absolute if r['arm']=='CURRENT_BN')
    gates=dict(primary_mean=primary['delta_pp']>=.5,primary_ci=primary['ci95_pp'][0]>0,prequential_first_exposure=first['ci95_pp'][0]>0,prequential_vs_entropy=pre_vs_entropy['ci95_pp'][0]>0,anchor_return=anchor_return['ci95_pp'][0]>0,all_primary_updates_norm_matched=all(v['norm_matched'] for r in updates if r['arm'] in ('STRAT','RANDOM') for v in r['steps']),strong_frontier_mean=all(float((take('STRAT')-take(a)).mean())>0 for a in ('CURRENT_BN','CW1','CW2','R_OLD')),original_no_harm=float(q[:,0].mean())>=-.2,return_no_harm=float((take('STRAT','RETAIN','PRE')[:,4]-take('CURRENT_BN')[:,4]).mean())>=-.2,worst_no_harm=float(q.min())>=-1,no_new_empty=not bool((take('STRAT',array=empty,refarray=rempty)&~baseline_empty).any()),no_new_nesting=row['nesting_violation_pixels']<=base_row['nesting_violation_pixels'],surface_coverage=row['surface_defined']>=base_row['surface_defined'],vcdr_coverage=row['vcdr_defined']>=base_row['vcdr_defined'],structure_direction=row['assd_pixel'] is not None and row['hd95_pixel'] is not None and row['vcdr_mae'] is not None and row['assd_pixel']<=base_row['assd_pixel'] and row['hd95_pixel']<=base_row['hd95_pixel'] and row['vcdr_mae']<=base_row['vcdr_mae'])
    result=dict(status='COMPLETE',readouts=EXPECTED,learner_trajectories=40,stream_arrivals=1600,legacy_parity=128,new_source_training=False,primary=primary,prequential=prequential,first_exposure=first,prequential_vs_entropy=pre_vs_entropy,anchor_return=anchor_return,gates=gates,advance_development_gate=all(gates.values()),interval_scope='Descriptive bootstrap of eight previously exposed content groups, binding both orders and repeated conditions; one checkpoint and two short synthetic streams',patient_independence=False,absolute=absolute,contrasts=contrasts)
    save(root/'public/RESULTS.json',result);save(root/'public/DIAGNOSTICS.json',diagnostics)
    for name,rr in (('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('transitions',transitions),('gate_audit',gate_aggregate),('update_audit',step_summary)):csvout(root/'public'/f'{name}.csv',rr)
    if c.get('reset_optimizer_per_arrival'):
        prior=Path(c['state_prior_root'])
        if not read(prior/'RESOURCE_LEDGER.json')['all_workers_retired'] or read(prior/'SCORE_COMPLETE.json')['readouts']!=EXPECTED:
            raise ValueError('complete retired prior state comparison required')
        previous=np.load(prior/'scorer/dice.private.npy',allow_pickle=False)
        if previous.shape!=dice.shape:raise ValueError('prior paired state matrix mismatch')
        previous_rows=read(prior/'private/predictions.json')
        if [key(r) for r in previous_rows]!=[key(r) for r in rows]:raise ValueError('prior row identity mismatch')
        parity=0
        for r in rows:
            if r['policy'] in ('RESET','REFERENCE') or r['arm'] in ('CW1','CW2'):
                name=f"{r['index']:04d}.bits"
                if (root/'predictions'/name).read_bytes()!=(prior/'predictions'/name).read_bytes():
                    raise ValueError('unchanged control parity failed')
                parity+=1
        paired=[]
        def state_summary(name,delta):
            out=dict(comparison=name,delta_pp=float(delta.mean()),ci95_pp=interval(delta),worst_cell_pp=float(delta.min()),negative_cells=int((delta<0).sum()),total_cells=int(delta.size));paired.append(out);return out
        def old(a,stage):return previous[:,:,:,LEARNERS.index(a),1,STAGES.index(stage)]
        primary_state=state_summary('STRAT_AFFINE_ONLY-STRAT_AFFINE_AND_ADAM:RETAIN_PRE',take('STRAT','RETAIN','PRE')-old('STRAT','PRE'))
        for a in ARMS:
            for stage in STAGES:
                if (a,stage)!=('STRAT','PRE'):state_summary(a+'_AFFINE_ONLY-'+a+'_AFFINE_AND_ADAM:RETAIN_'+stage,take(a,'RETAIN',stage)-old(a,stage))
        state_summary('STRAT_STATE_CHANGE_MINUS_RANDOM_STATE_CHANGE:RETAIN_PRE',(take('STRAT','RETAIN','PRE')-old('STRAT','PRE'))-(take('RANDOM','RETAIN','PRE')-old('RANDOM','PRE')))
        source_first=state_summary('STRAT_AFFINE_ONLY-SOURCE_EVAL:FIRST_EXPOSURE_PRE',take('STRAT','RETAIN','PRE')[:,0]-take('SOURCE_EVAL')[:,0])
        source_all=state_summary('STRAT_AFFINE_ONLY-SOURCE_EVAL:ALL_PRE',take('STRAT','RETAIN','PRE')-take('SOURCE_EVAL'))
        state_summary('STRAT_AFFINE_ONLY-SOURCE_EVAL:RETURN_PRE',take('STRAT','RETAIN','PRE')[:,4]-take('SOURCE_EVAL')[:,4])
        state_summary('STRAT_AFFINE_ONLY-STRAT_RESET:POST',take('STRAT')-take('STRAT','RESET','POST'))
        state_gates=dict(primary_mean=primary_state['delta_pp']>=.5,primary_ci=primary_state['ci95_pp'][0]>0,first_exposure_positive=source_first['ci95_pp'][0]>0,source_all_positive=source_all['ci95_pp'][0]>0,source_worst_no_harm=source_all['worst_cell_pp']>=-1,quota_random_positive=primary['ci95_pp'][0]>0,norm_matched=gates['all_primary_updates_norm_matched'],no_new_empty=gates['no_new_empty'],no_new_nesting=gates['no_new_nesting'])
        save(root/'public/STATE_COMPARISON.json',dict(status='COMPLETE',primary=primary_state,contrasts=paired,gates=state_gates,advance_development_gate=all(state_gates.values()),unchanged_control_exact_parity=parity,core_retained_state='AFFINE_ONLY',native_retained_state='UNCHANGED_NATIVE',legacy_gate_scope='RESULTS.json preserves V2 routing gates; state decision uses this separate frozen gate'))
        csvout(root/'public/state_contrasts.csv',paired)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',readouts=EXPECTED,evaluation_label_files=8,cpu_seconds=time.time()-started,unique_metric_evaluations=len(cache)))
