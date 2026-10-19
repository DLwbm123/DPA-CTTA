"""P1 static frozen translation-phase qualification; no parameter updates."""
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .run import guard_assets,read,save,controller
from .translation_phase import ARMS,outputs
from .static_teacher import summary,region_counts
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..r78_cross_patient.score import metric,csvout,interval
from .score import unpack
PRIMARY='PHASE_PROB-NOALIGN_PROB:OC_DICE'


def validate(c):
    assert c['design_version']=='TRANSLATION_PHASE_P1' and c['primary']==PRIMARY
    assert c['offsets']==[[0,0],[0,1],[1,0],[1,1]] and not c['history']
    assert c['input_boundary']=='reflect' and c['output_boundary']=='source_on_last_row_and_column' and c['probability_weights']==[.25]*4
    assert c['arms']==list(ARMS)
    assert c['unique_inputs']==32 and c['expected_readouts']==256
    assert c['seed']==20261018 and c['grouping']=='CONTENT_DEVELOPMENT_USER_AUTHORIZED'
    assert c['stream_blocks']==[0,1,2,3,0] and c['orders']==['forward','reverse']
    assert c['parameter_updates']==0 and c['repeat_all_inputs']
    assert not any(c[k] for k in ('rdc','engram','new_training','patient_independence'))
    assert c['gpu_cap_seconds']==1200 and c['cpu_cap_seconds']==900 and c['output_cap_bytes']==2*1024**3


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    validate(c);root=Path(c['output_root']);started=time.time()
    status=dict(status='QUALIFYING',unique_inputs=0,predictions=0,parameter_updates=0)
    save(root/'RUN_STATE.json',status)
    def guard():
        if time.time()-started>c['gpu_cap_seconds']-20: raise TimeoutError('fixed GPU cap')
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=read(root/'private/panel.json')
        assert len(panel)==8 and len({r['group_id'] for r in panel})==8 and all('mask_path' not in r for r in panel)
        opened=guard_assets(c,panel)
        for p in (root/'scorer/panel.json',Path(c['state_prior_root'])/'public/RESULTS.json'):
            try:p.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('access guard failed')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        model,_=build_reference_model('fundus');model.load_state_dict(state);model.cuda().eval().requires_grad_(False)
        before={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        rows=[];frozen=[];audits=[];forwards=0;tick=time.time()
        def run(x):
            nonlocal forwards
            zz,dd=outputs(model,x);forwards+=4
            return zz,dd
        with torch.no_grad():
            for case,item in enumerate(panel):
                pixels=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,spec in enumerate(FINAL):
                    guard();x=normalize(perturb(pixels,spec,SEED+case)).cuda();zz,diag=run(x)
                    if forwards==4:
                        import subprocess
                        raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True)
                        own=[v for v in raw.splitlines() if v.split(',')[0].strip()==str(os.getpid())]
                        assert len(own)==1 and c['gpu']['uuid'] in own[0] and all(k not in own[0].lower() for k in ('wangbomin','ctta','phase'))
                        save(root/'GPU_VISIBLE.private.json',dict(status='PASS',line=own[0]))
                    frozen.append({a:z.cpu() for a,z in zz.items()});audits.append(diag)
                    for a in ARMS:
                        z=zz[a];p=z.cpu().numpy()[0]>=0;ix=len(rows);dz=z-zz['SOURCE']
                        with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(np.packbits(p).tobytes())
                        rows.append(dict(index=ix,case=case,condition=k,arm=a,nesting_violation_pixels=int((p[1]&~p[0]).sum()),logit_rms=float(dz.square().mean().sqrt()),changed_pixels=int((p!=(zz['SOURCE'].cpu().numpy()[0]>=0)).sum())))
                    status.update(unique_inputs=case*4+k+1,predictions=len(rows))
                    if status['unique_inputs']==2:
                        avg=(time.time()-tick)/2;projection=time.time()-started+(64-2)*avg*1.35+60;admitted=projection<=1180
                        save(root/'QUALIFICATION.json',dict(status='PASS' if admitted else 'NOT_ADMITTED',projected_worker_seconds=projection,measured_unique_input_seconds=avg,full_output_bundles=64,final_label_reads=0))
                        if not admitted:raise TimeoutError('fixed coverage/repeat not admitted')
                        status['status']='GENERATING'
                    if status['unique_inputs']%8==0:save(root/'RUN_STATE.json',status)
            for case,item in enumerate(panel):
                pixels=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,spec in enumerate(FINAL):
                    guard();x=normalize(perturb(pixels,spec,SEED+case)).cuda();zz,diag=run(x)
                    if any(not torch.equal(z.cpu(),frozen[case*4+k][a]) for a,z in zz.items()) or diag!=audits[case*4+k]:raise ValueError('full encoder/output repeat differs')
            if any(not torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('source state changed')
        assert len(rows)==256 and forwards==256
        save(root/'private/predictions.json',rows)
        save(root/'private/transport.json',audits)
        save(root/'FULL_REPEATABILITY.json',dict(status='PASS',unique_inputs=32,logit_arrays=256,all_arms_exact=True,scope='All eight outputs, full encoder rerun, same process only'))
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',unique_inputs=32,unique_readouts=256,logical_stream_readouts=640,source_parameter_updates=0,optimizer_steps=0,backwards=0,model_forwards=forwards,source_parameters_and_buffers_immutable=True,bn_buffers_changed=False,common_overlap_source_fallback_pass=True,four_queries_shared_by_all_controls=True,frozen_source_probability_smoothing_control=True,gpu_visible_process_checked=True))
        save(root/'ACCESS_AUDIT.json',dict(registered_rgb_files=len(opened),source_label_reads=0,evaluation_label_reads=0,prior_outcome_score_reads=0,new_source_training=False,future_images_used_for_update=False,patient_independence=False,previously_exposed_development_contents=True))
        diagnostics=[]
        for key in audits[0]:
            vals=[d[key] for d in audits];diagnostics.append(dict(measure=key,mean=float(np.mean(vals)),max=float(max(vals)),min=float(min(vals))))
        for a in ARMS:
            rr=[r for r in rows if r['arm']==a]
            diagnostics.append(dict(measure=a+'_logit_rms',mean=float(np.mean([r['logit_rms'] for r in rr])),max=float(max(r['logit_rms'] for r in rr)),min=float(min(r['logit_rms'] for r in rr))))
            diagnostics.append(dict(measure=a+'_changed_pixels',mean=float(np.mean([r['changed_pixels'] for r in rr])),max=float(max(r['changed_pixels'] for r in rr)),min=float(min(r['changed_pixels'] for r in rr))))
        save(root/'public/DIAGNOSTICS.json',diagnostics)
        used=sum(p.stat().st_size for d in ('private','predictions','logs') for p in (root/d).rglob('*') if p.is_file());assert used<=c['output_cap_bytes']
        status.update(status='GPU_COMPLETE',output_bytes=used,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as e:
        status.update(status='NOT_EVALUABLE',error_type=type(e).__name__,reason=str(e));raise
    finally:status.update(gpu_worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


def score(c):
    from ..source_io import read_mask
    from scipy.ndimage import binary_erosion,distance_transform_edt
    validate(c);root=Path(c['output_root']);started=time.time();ret=read(root/'processes/worker.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code']==0
    status=read(root/'RUN_STATE.json');assert status['status']=='GPU_COMPLETE' and status['predictions']==256 and status['unique_inputs']==32
    assert read(root/'FULL_REPEATABILITY.json')['all_arms_exact']
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json');assert len(panel)==8
    expected={(i,k,a) for i in range(8) for k in range(4) for a in ARMS}
    assert len(rows)==256 and {(r['case'],r['condition'],r['arm']) for r in rows}==expected
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    masks={(r['case'],r['condition'],r['arm']):unpack(root/'predictions'/f"{r['index']:04d}.bits")[1] for r in rows}
    metrics={};counts={};individual=[]
    for row in rows:
        if time.time()-started>c['cpu_cap_seconds']-10:raise TimeoutError('CPU cap')
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
        if control in ('SOURCE','PHASE_PROB'):continue
        contrasts.append(summary('PHASE_PROB-'+control+':OC_DICE',cubes['PHASE_PROB'][:,:,1]-cubes[control][:,:,1]))
    primary=next(v for v in contrasts if v['comparison']==PRIMARY)
    base=absolute[0];checks=[]
    for a in ARMS[1:]:
        row=next(v for v in absolute if v['arm']==a);oc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:OC_DICE');allc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:ALL_DICE');returned=next(v for v in contrasts if v['comparison']==a+'-SOURCE:RETURN_DICE')
        key={v['region']:v for v in regions if v['arm']==a and v['channel']=='OC'}
        gates=dict(oc_mean=oc['delta_pp']>=.5,oc_ci=oc['ci95_pp'][0]>0,all_ci=allc['ci95_pp'][0]>0,worst=allc['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,oc_foreground_net=key['foreground']['net_ci_low_pp'] is not None and key['foreground']['net_ci_low_pp']>0,oc_boundary_net=key['boundary3px']['net_ci_low_pp'] is not None and key['boundary3px']['net_ci_low_pp']>0,structure=all(row[f] is not None and row[f]<=base[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')),coverage=row['surface_defined']>=base['surface_defined'] and row['vcdr_defined']>=base['vcdr_defined'],empty_nesting=row['empty_predictions']<=base['empty_predictions'] and row['nesting_violation_pixels']<=base['nesting_violation_pixels'])
        checks.append(dict(arm=a,gates=gates,worth_followup=all(gates.values())))
    controls={a:next(v for v in contrasts if v['comparison']=='PHASE_PROB-'+a+':OC_DICE')['ci95_pp'][0]>0 for a in ('NOALIGN_PROB','SOURCE_SMOOTH','SHIFT_X','SHIFT_Y','SHIFT_XY')}
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,absolute=absolute,signal_checks=checks,mechanism_primary_pass=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0,advance_to_new_protocol=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0 and all(controls.values()) and next(v['worth_followup'] for v in checks if v['arm']=='PHASE_PROB'),static_only=True,parameter_updates=0,continual_benefit_tested=False,strong_control_checks=controls,unique_inputs=32,unique_predictions=256,logical_stream_readouts=640,patient_independence=False,scope='Eight reused development contents; weighted conditions [0,1,2,3,0], two identical readout orders; no independent-patient inference'))
    for n,rr in [('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('region_transitions',regions)]:csvout(root/'public'/f'{n}.csv',rr)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',evaluation_label_files=8,unique_inputs=32,unique_metric_evaluations=256,absolute_rows=8,region_rows=64,condition_channel_rows=64,contrast_rows=len(contrasts),cpu_seconds=time.time()-started))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='worker':worker(c)
    elif mode=='score':score(c)
    elif mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS'
        sys.exit(controller(c))
    else:raise ValueError('unknown mode')
