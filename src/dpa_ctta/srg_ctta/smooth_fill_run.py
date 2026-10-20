"""P5 fixed smooth/fill composition qualification; no parameter updates."""
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .run import read,save
from .smooth_fill import ARMS,outputs
from .smooth_fill_control import controller,bytes_used
from .static_teacher import summary,region_counts
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..r78_cross_patient.score import metric,csvout,interval
from .score import unpack
PRIMARY='FILL_SMOOTH-SOURCE:OC_DICE'


def validate(c):
    assert c['design_version']=='SMOOTH_FILL_P5' and c['primary']==PRIMARY
    assert c['foreground_connectivity']==8 and c['background_connectivity']==4 and c['inputs']=='P1_COMPLETED_FROZEN_BITS'
    assert c['time_gpu_budgets_removed'] and c['storage_hard_bytes']==20000000000 and c['storage_working_bytes']==18000000000
    assert c['arms']==list(ARMS)
    assert c['unique_inputs']==32 and c['expected_readouts']==192
    assert c['seed']==20261018 and c['grouping']=='CONTENT_DEVELOPMENT_USER_AUTHORIZED'
    assert c['stream_blocks']==[0,1,2,3,0] and c['orders']==['forward','reverse']
    assert c['parameter_updates']==0 and c['repeat_all_inputs']
    assert not any(c[k] for k in ('rdc','engram','new_training','patient_independence'))
    assert c['output_cap_bytes']==2*1024**3


def worker(c):
    validate(c);root=Path(c['output_root']);prior=Path(c['input_root']);started=time.time()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and read(prior/'RESOURCE_LEDGER.json')['all_workers_retired']
    state=dict(status='GENERATING',unique_inputs=0,predictions=0,parameter_updates=0)
    save(root/'RUN_STATE.json',state)
    index=prior/'private/predictions.json';allowed={index.resolve()}|{p.resolve() for p in (prior/'predictions').glob('*.bits')}
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to((root/'scorer').resolve()) or p.is_relative_to(prior.resolve()) and p not in allowed:raise PermissionError('only frozen bits and their index; no labels/outcomes')
    sys.addaudithook(audit)
    for p in (root/'scorer/panel.json',prior/'public/RESULTS.json'):
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('label/outcome guard failed')
    src=read(index);lookup={(v['case'],v['condition'],v['arm']):v for v in src};rows=[];frozen=[];method_audit=[]
    for i in range(8):
        for k in range(4):
            masks=[];original=[]
            for arm in ('SOURCE','SOURCE_SMOOTH'):
                v=lookup[i,k,arm];path=prior/'predictions'/f"{v['index']:04d}.bits";original.append(path.read_bytes());masks.append(unpack(path)[1])
            zz=outputs(*masks);frozen.append((masks,zz))
            for a in ARMS:
                p=zz[a];ix=len(rows);packed=np.packbits(p).tobytes()
                if a in ('SOURCE','SMOOTH'):assert packed==original[('SOURCE','SMOOTH').index(a)]
                with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(packed)
                rows.append(dict(index=ix,case=i,condition=k,arm=a,nesting_violation_pixels=int((p[1]&~p[0]).sum()),changed_pixels=int((p!=masks[0]).sum())))
            for ch in range(2):
                m=zz['SMOOTH'][ch];p=zz['FILL_SMOOTH'][ch];q=zz['MASS_MATCH_SMOOTH'][ch]
                assert p.sum()==q.sum() and (p!=m).sum()==(q!=m).sum()
                assert not (m&~p).any() and not (m&~q).any()
                method_audit.append(dict(base='SMOOTH',channel=ch,changed_pixels=int((p!=m).sum()),removed_pixels=0,filled_pixels=int((p&~m).sum()),control_overlap_pixels=int(((p!=m)&(q!=m)).sum())))
            state.update(unique_inputs=i*4+k+1,predictions=len(rows))
    for masks,expected in frozen:
        actual=outputs(*masks)
        assert all(np.array_equal(actual[a],expected[a]) for a in ARMS)
    assert len(rows)==192
    save(root/'private/predictions.json',rows);save(root/'private/method_audit.json',method_audit)
    diag=[]
    for base in ('SMOOTH',):
        for ch in range(2):
            q=[v for v in method_audit if v['base']==base and v['channel']==ch]
            for key in ('changed_pixels','removed_pixels','filled_pixels','control_overlap_pixels'):diag.append(dict(base=base,channel=('OD','OC')[ch],measure=key,mean=float(np.mean([v[key] for v in q])),max=max(v[key] for v in q),total=sum(v[key] for v in q)))
    save(root/'public/DIAGNOSTICS.json',diag)
    save(root/'FULL_REPEATABILITY.json',dict(status='PASS',unique_inputs=32,binary_arrays=192,all_arms_exact=True,scope='Same frozen bits, repeated deterministic projections, same process; no network rerun'))
    save(root/'MECHANISM_AUDIT.json',dict(status='PASS',unique_readouts=192,unique_inputs=32,source_parameter_updates=0,model_forwards=0,backwards=0,optimizer_steps=0,source_bits_unchanged=True,smoothing_bits_unchanged=True,foreground_connectivity=8,background_connectivity=4,matched_removed_filled_counts=True,projection_applied_per_head=True,no_nesting_clamp=True))
    save(root/'ACCESS_AUDIT.json',dict(image_reads=0,weights_read=0,evaluation_label_reads=0,prior_outcome_score_reads=0,frozen_input_bit_files=64,rdc=False,engram=False,new_source_training=False))
    state.update(status='INFERENCE_COMPLETE',worker_body_seconds=time.time()-started,output_bytes=bytes_used(root));save(root/'RUN_STATE.json',state)


def score(c):
    from ..source_io import read_mask
    from scipy.ndimage import binary_erosion,distance_transform_edt
    validate(c);root=Path(c['output_root']);started=time.time();ret=read(root/'processes/worker.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code']==0
    status=read(root/'RUN_STATE.json');assert status['status']=='INFERENCE_COMPLETE' and status['predictions']==192 and status['unique_inputs']==32
    assert read(root/'FULL_REPEATABILITY.json')['all_arms_exact']
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json');assert len(panel)==8
    expected={(i,k,a) for i in range(8) for k in range(4) for a in ARMS}
    assert len(rows)==192 and {(r['case'],r['condition'],r['arm']) for r in rows}==expected
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    masks={(r['case'],r['condition'],r['arm']):unpack(root/'predictions'/f"{r['index']:04d}.bits")[1] for r in rows}
    metrics={};counts={};individual=[]
    for row in rows:
        i,k,a=row['case'],row['condition'],row['arm'];p=masks[i,k,a];base=masks[i,k,'SOURCE'];gt=labels[i]
        mm,rr=metric(p,gt,base);metrics[i,k,a]=(mm,rr);individual.append(dict(**row,metrics=mm,ratios=rr))
        for ch in range(2):
            y=gt[ch];surface=y^binary_erosion(y,border_value=0)
            bd=distance_transform_edt(~surface)<=3 if surface.any() else np.zeros_like(y)
            for region,mask in [('all',np.ones_like(y)),('foreground',y),('background',~y),('boundary3px',bd)]:
                counts[i,k,a,ch,region]=region_counts(p[ch],y,base[ch],mask)
    save(root/'scorer/individual.private.json',individual)
    def finite(v):
        a=[float(x) for x in v if x is not None and np.isfinite(x)]
        return (float(np.mean(a)) if a else None),len(a)
    cubes={};absolute=[];channels=[];regions=[]
    for a in ARMS:
        cube=np.array([[[metrics[i,k,a][0][ch]['dice_percent'] for ch in range(2)] for k in (0,1,2,3,0)] for i in range(8)])
        cubes[a]=cube
        weighted=[metrics[i,k,a] for i in range(8) for k in (0,1,2,3,0) for _ in range(2)]
        assd,n=finite([m['assd_pixel'] for mm,rr in weighted for m in mm]);hd,_=finite([m['hd95_pixel'] for mm,rr in weighted for m in mm]);vc,vn=finite([rr['vcdr_abs_error'] for mm,rr in weighted])
        nesting=sum(int((masks[i,k,a][1]&~masks[i,k,a][0]).sum()) for i in range(8) for k in (0,1,2,3,0) for _ in range(2))
        absolute.append(dict(arm=a,dice_percent=float(cube.mean()),od_dice_percent=float(cube[:,:,0].mean()),oc_dice_percent=float(cube[:,:,1].mean()),assd_pixel=assd,hd95_pixel=hd,vcdr_mae=vc,surface_defined=n,surface_total=160,vcdr_defined=vn,vcdr_total=80,empty_predictions=sum(m['pred_empty'] for mm,rr in weighted for m in mm),nesting_violation_pixels=nesting,unique_inputs=32,logical_readouts=80,unique_condition_macro_dice_percent=float(np.mean([metrics[i,k,a][0][ch]['dice_percent'] for i in range(8) for k in range(4) for ch in range(2)]))))
        for k in range(4):
            for ch,name in enumerate(('OD','OC')):
                channels.append(dict(arm=a,condition=k,channel=name,dice_percent=float(np.mean([metrics[i,k,a][0][ch]['dice_percent'] for i in range(8)]))))
        for ch,name in enumerate(('OD','OC')):
            for region in ('all','foreground','background','boundary3px'):
                u=[counts[i,k,a,ch,region] for i in range(8) for k in range(4)]
                cf=np.array([[counts[i,k,a,ch,region]['corrected']/counts[i,k,a,ch,region]['pixels'] if counts[i,k,a,ch,region]['pixels'] else np.nan for k in (0,1,2,3,0)] for i in range(8)])
                hf=np.array([[counts[i,k,a,ch,region]['spoiled']/counts[i,k,a,ch,region]['pixels'] if counts[i,k,a,ch,region]['pixels'] else np.nan for k in (0,1,2,3,0)] for i in range(8)])
                net=summary(a+':'+name+':'+region+':NET_PIXEL_PP',(cf-hf)*100) if np.isfinite(cf).all() and np.isfinite(hf).all() else None
                regions.append(dict(arm=a,channel=name,region=region,corrected_pixels_unique=sum(t['corrected'] for t in u),spoiled_pixels_unique=sum(t['spoiled'] for t in u),disagreement_pixels_unique=sum(t['disagreement'] for t in u),region_pixels_unique=sum(t['pixels'] for t in u),macro_corrected_fraction=finite(cf.flatten())[0],macro_spoiled_fraction=finite(hf.flatten())[0],macro_net_pp=net['delta_pp'] if net else None,net_ci_low_pp=net['ci95_pp'][0] if net else None,net_ci_high_pp=net['ci95_pp'][1] if net else None,macro_defined=int(np.isfinite(cf).sum()),macro_total=40))
    contrasts=[]
    for a in ARMS[1:]:
        for name,ch in [('ALL',None),('OD',0),('OC',1)]:
            delta=cubes[a]-cubes['SOURCE'];d=delta if ch is None else delta[:,:,ch]
            contrasts.append(summary(a+'-SOURCE:'+name+'_DICE',d))
        contrasts.append(summary(a+'-SOURCE:RETURN_DICE',cubes[a][:,4]-cubes['SOURCE'][:,4]))
    for control in ARMS:
        if control in ('SOURCE','FILL_SMOOTH'):continue
        contrasts.append(summary('FILL_SMOOTH-'+control+':OC_DICE',cubes['FILL_SMOOTH'][:,:,1]-cubes[control][:,:,1]))
    contrasts.append(summary('TOPO_SMOOTH-SMOOTH:OC_DICE',cubes['TOPO_SMOOTH'][:,:,1]-cubes['SMOOTH'][:,:,1]))
    primary=next(v for v in contrasts if v['comparison']==PRIMARY)
    base=absolute[0];checks=[]
    for a in ARMS[1:]:
        row=next(v for v in absolute if v['arm']==a);oc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:OC_DICE');allc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:ALL_DICE');returned=next(v for v in contrasts if v['comparison']==a+'-SOURCE:RETURN_DICE')
        key={v['region']:v for v in regions if v['arm']==a and v['channel']=='OC'}
        gates=dict(oc_mean=oc['delta_pp']>=.5,oc_ci=oc['ci95_pp'][0]>0,all_ci=allc['ci95_pp'][0]>0,worst=allc['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,oc_foreground_net=key['foreground']['net_ci_low_pp'] is not None and key['foreground']['net_ci_low_pp']>0,oc_boundary_net=key['boundary3px']['net_ci_low_pp'] is not None and key['boundary3px']['net_ci_low_pp']>0,structure=all(row[f] is not None and row[f]<=base[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')),coverage=row['surface_defined']>=base['surface_defined'] and row['vcdr_defined']>=base['vcdr_defined'],empty_nesting=row['empty_predictions']<=base['empty_predictions'] and row['nesting_violation_pixels']<=base['nesting_violation_pixels'])
        checks.append(dict(arm=a,gates=gates,worth_followup=all(gates.values())))
    controls={a:next(v for v in contrasts if v['comparison']=='FILL_SMOOTH-'+a+':OC_DICE')['ci95_pp'][0]>0 for a in ('SMOOTH','MASS_MATCH_SMOOTH','FILL_SOURCE','TOPO_SMOOTH')}
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,absolute=absolute,signal_checks=checks,mechanism_primary_pass=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0,advance_to_new_protocol=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0 and all(controls.values()) and next(v['worth_followup'] for v in checks if v['arm']=='FILL_SMOOTH'),static_only=True,parameter_updates=0,continual_benefit_tested=False,strong_control_checks=controls,unique_inputs=32,unique_predictions=192,logical_stream_readouts=480,patient_independence=False,scope='Eight reused development contents; weighted conditions [0,1,2,3,0], two identical readout orders; no independent-patient inference'))
    for n,rr in [('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('region_transitions',regions)]:csvout(root/'public'/f'{n}.csv',rr)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',evaluation_label_files=8,unique_inputs=32,unique_metric_evaluations=192,absolute_rows=6,region_rows=48,condition_channel_rows=48,contrast_rows=len(contrasts),cpu_seconds=time.time()-started))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='worker':worker(c)
    elif mode=='score':score(c)
    elif mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS'
        sys.exit(controller(c))
    else:raise ValueError('unknown mode')
