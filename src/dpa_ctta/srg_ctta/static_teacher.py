"""E2-A: static source / mirrored source / equal-probability fusion; no updates."""
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .run import guard_assets, read, save, controller
from .paired_state import summary
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..r78_cross_patient.score import metric,csvout
from .score import unpack

ARMS=('SOURCE','TEACHER_FLIP','PROB_AVG')
PRIMARY='PROB_AVG-SOURCE:OC_DICE'


def validate(c):
    assert c['design_version']=='STATIC_TEACHER_E2A' and c['primary']==PRIMARY
    assert c['arms']==list(ARMS) and c['unique_inputs']==32 and c['expected_readouts']==96
    assert c['seed']==20261018 and c['grouping']=='CONTENT_DEVELOPMENT_USER_AUTHORIZED'
    assert c['stream_blocks']==[0,1,2,3,0] and c['orders']==['forward','reverse']
    assert c['fusion_weight']==.5 and c['parameter_updates']==0 and c['repeat_all_inputs']
    assert not any(c[k] for k in ('rdc','engram','new_training','patient_independence'))
    assert c['gpu_cap_seconds']==1200 and c['cpu_cap_seconds']==900 and c['output_cap_bytes']==2*1024**3


def outputs(z,t):
    if z.shape!=t.shape or z.ndim!=4 or z.shape[:2]!=(1,2):raise ValueError('one image, independent OD/OC')
    p,q=z.double().sigmoid(),t.double().sigmoid()
    return dict(SOURCE=z>=0,TEACHER_FLIP=t>=0,PROB_AVG=(p+q)*.5>=.5)


def region_counts(pred,truth,source,region):
    corrected=(source!=truth)&(pred==truth)&region
    spoiled=(source==truth)&(pred!=truth)&region
    disagree=(source!=pred)&region
    c,h,d,n=map(int,(corrected.sum(),spoiled.sum(),disagree.sum(),region.sum()))
    assert c+h==d
    return dict(corrected=c,spoiled=h,disagreement=d,pixels=n,net=c-h)


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    validate(c);root=Path(c['output_root']);started=time.time()
    status=dict(status='QUALIFYING',unique_inputs=0,predictions=0,parameter_updates=0)
    save(root/'RUN_STATE.json',status)
    def guard():
        if time.time()-started>c['gpu_cap_seconds']-20:raise TimeoutError('fixed GPU cap')
    try:
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=read(root/'private/panel.json');assert len(panel)==8 and len({r['group_id'] for r in panel})==8 and all('mask_path' not in r for r in panel)
        opened=guard_assets(c,panel)
        for p in (root/'scorer/panel.json',Path(c['state_prior_root'])/'public/RESULTS.json'):
            try:p.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('access guard failed')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        model,_=build_reference_model('fundus');model.load_state_dict(state);model.cuda().eval().requires_grad_(False)
        versions={n:p._version for n,p in model.named_parameters()};buffers={n:b.clone() for n,b in model.named_buffers()}
        rows=[];frozen=[];counts=0;tick=time.time()
        with torch.no_grad():
            for case,item in enumerate(panel):
                pixels=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,spec in enumerate(FINAL):
                    guard();x=normalize(perturb(pixels,spec,SEED+case)).cuda()
                    z=model(x)[0];t=model(x.flip(-1))[0].flip(-1);counts+=2
                    frozen.append((z.cpu(),t.cpu()))
                    pp=outputs(z,t)
                    if bool(((pp['PROB_AVG']!=pp['SOURCE'])&(pp['TEACHER_FLIP']==pp['SOURCE'])).any()):raise ValueError('fusion changed agreed decision')
                    for a in ARMS:
                        ix=len(rows);p=pp[a].cpu().numpy()[0]
                        with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(np.packbits(p).tobytes())
                        rows.append(dict(index=ix,case=case,condition=k,arm=a,nesting_violation_pixels=int((p[1]&~p[0]).sum())))
                    status.update(unique_inputs=case*4+k+1,predictions=len(rows))
                    if status['unique_inputs']==2:
                        avg=(time.time()-tick)/2;projection=time.time()-started+(64-2)*avg*1.35+60
                        admitted=projection<=1180
                        save(root/'QUALIFICATION.json',dict(status='PASS' if admitted else 'NOT_ADMITTED',projected_worker_seconds=projection,measured_unique_input_seconds=avg,final_label_reads=0))
                        if not admitted:raise TimeoutError('static coverage/repeat not admitted')
                        status['status']='GENERATING'
                    if status['unique_inputs']%8==0:save(root/'RUN_STATE.json',status)
            for case,item in enumerate(panel):
                pixels=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,spec in enumerate(FINAL):
                    guard();x=normalize(perturb(pixels,spec,SEED+case)).cuda();z=model(x)[0];t=model(x.flip(-1))[0].flip(-1);counts+=2
                    a,b=frozen[case*4+k]
                    if not torch.equal(z.cpu(),a) or not torch.equal(t.cpu(),b):raise ValueError('full frozen-source forward repeat differs')
            if any(p._version!=versions[n] or p.grad is not None for n,p in model.named_parameters()):raise ValueError('source parameters changed')
            if any(not torch.equal(b,buffers[n]) for n,b in model.named_buffers()):raise ValueError('BN buffer changed')
        assert len(rows)==96 and status['unique_inputs']==32 and counts==128
        save(root/'private/predictions.json',rows)
        save(root/'FULL_REPEATABILITY.json',dict(status='PASS',unique_inputs=32,logit_arrays=64,source_and_teacher_exact=True,scope='Same process full encoder rerun for all 32 inputs; no cross-process claim'))
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',unique_inputs=32,unique_readouts=96,logical_stream_readouts=240,source_parameter_updates=0,optimizer_steps=0,backwards=0,model_forwards=128,source_parameters_and_buffers_immutable=True,fusion_probability_weight=.5,fusion_decisions_between_source_teacher=True))
        save(root/'ACCESS_AUDIT.json',dict(source_image_files=len(opened),source_label_reads=0,evaluation_label_reads=0,prior_outcome_score_reads=0,new_source_training=False,future_images_used_for_update=False,patient_independence=False,previously_exposed_development_contents=True))
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
    status=read(root/'RUN_STATE.json');assert status['status']=='GPU_COMPLETE' and status['predictions']==96 and status['unique_inputs']==32
    assert read(root/'FULL_REPEATABILITY.json')['source_and_teacher_exact']
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json');assert len(panel)==8
    expected={(i,k,a) for i in range(8) for k in range(4) for a in ARMS}
    assert len(rows)==96 and {(r['case'],r['condition'],r['arm']) for r in rows}==expected
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
    primary=next(v for v in contrasts if v['comparison']==PRIMARY)
    base=absolute[0];checks=[]
    for a in ARMS[1:]:
        row=next(v for v in absolute if v['arm']==a);oc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:OC_DICE');allc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:ALL_DICE');returned=next(v for v in contrasts if v['comparison']==a+'-SOURCE:RETURN_DICE')
        key={v['region']:v for v in regions if v['arm']==a and v['channel']=='OC'}
        gates=dict(oc_mean=oc['delta_pp']>=.5,oc_ci=oc['ci95_pp'][0]>0,all_ci=allc['ci95_pp'][0]>0,worst=allc['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,oc_foreground_net=key['foreground']['net_ci_low_pp'] is not None and key['foreground']['net_ci_low_pp']>0,oc_boundary_net=key['boundary3px']['net_ci_low_pp'] is not None and key['boundary3px']['net_ci_low_pp']>0,structure=all(row[f] is not None and row[f]<=base[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')),coverage=row['surface_defined']>=base['surface_defined'] and row['vcdr_defined']>=base['vcdr_defined'],empty_nesting=row['empty_predictions']<=base['empty_predictions'] and row['nesting_violation_pixels']<=base['nesting_violation_pixels'])
        checks.append(dict(arm=a,gates=gates,worth_followup=all(gates.values())))
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,absolute=absolute,signal_checks=checks,static_only=True,parameter_updates=0,continual_benefit_tested=False,unique_inputs=32,unique_predictions=96,logical_stream_readouts=240,patient_independence=False,scope='Eight reused development contents; weighted conditions [0,1,2,3,0], two identical readout orders; no independent-patient inference'))
    for n,rr in [('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('region_transitions',regions)]:csvout(root/'public'/f'{n}.csv',rr)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',evaluation_label_files=8,unique_inputs=32,unique_metric_evaluations=96,absolute_rows=3,region_rows=24,cpu_seconds=time.time()-started))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='worker':worker(c)
    elif mode=='score':score(c)
    elif mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS'
        sys.exit(controller(c))
    else:raise ValueError('unknown mode')
