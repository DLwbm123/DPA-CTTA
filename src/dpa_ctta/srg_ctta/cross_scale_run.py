"""A0.1 offline scale-reference diagnostic; no online-learning claim."""
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .run import read,save,controller
from .single_bn import moments,output
from .cross_scale import ARMS,GROUPS,supporters,pool,anchored_output,relative_output,mean_check
from .static_teacher import summary,region_counts
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..r78_cross_patient.score import metric,csvout,interval
from .score import unpack
PRIMARY='CROSS_SCALE-SOURCE:OC_DICE'


def validate(c):
    assert c['design_version']=='CROSS_CONTENT_SCALE_A01' and c['primary']==PRIMARY and c['arms']==list(ARMS)
    assert c['bn_layer']=='res.bn1' and c['support_groups']==[list(g) for g in GROUPS]
    assert c['unique_inputs']==32 and c['expected_readouts']==128 and c['moment_correction']==0
    assert c['seed']==20261018 and c['grouping']=='CONTENT_DEVELOPMENT_USER_AUTHORIZED'
    assert c['stream_blocks']==[0,1,2,3,0] and c['orders']==['forward','reverse']
    assert c['parameter_updates']==0 and c['repeat_all_inputs'] and c['oracle_same_condition_support'] and c['same_content_original_reference']
    assert not any(c[k] for k in ('rdc','engram','new_training','patient_independence','online_ctta'))
    assert c['gpu_cap_seconds']==1200 and c['cpu_cap_seconds']==900 and c['output_cap_bytes']==2*1024**3


def guard_assets(c,panel):
    root=Path(c['output_root']).resolve();old=Path(c['a0_root']).resolve();data=Path(c['source_root']).resolve()
    images={Path(r['image_path']).resolve() for r in panel};opened=set()
    allowed={old/'private/predictions.json',*(old/'predictions'/f'{i:04d}.bits' for i in range(128))}
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root/'scorer') or p==root/'A0_ROBUSTNESS.json' or p==root/'private/a0_robustness.private.json' or (p.is_relative_to(old) and p not in allowed):raise PermissionError('labels and old outcome scores denied')
        if p.is_relative_to(data):
            if p not in images:raise PermissionError('only eight registered RGB files')
            opened.add(str(p))
    sys.addaudithook(audit);return opened


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
        for p in (root/'scorer/panel.json',Path(c['a0_root'])/'scorer/individual.private.json',root/'A0_ROBUSTNESS.json'):
            try:p.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('access guard failed')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        model,_=build_reference_model('fundus');model.load_state_dict(state);model.cuda().eval().requires_grad_(False)
        bn=dict(model.named_modules())[c['bn_layer']];assert isinstance(bn,torch.nn.BatchNorm2d) and bn.num_features==64
        before={n:t.detach().cpu().clone() for n,t in model.state_dict().items()}
        old=read(Path(c['a0_root'])/'private/predictions.json');oldmap={(r['case'],r['condition'],r['arm']):r['index'] for r in old}
        rows=[];diagnostics=[];forwards=0;parity=0
        def frame(i,k):
            pixels=read_pixels(panel[i]['image_path'],'fundus',panel[i]['image_size'])
            return normalize(perturb(pixels,FINAL[k],SEED+i)).cuda()
        def source(x):
            nonlocal forwards
            got={}
            def hook(m,args,y):got.update(h=args[0].detach().clone(),y=y.detach().clone())
            handle=bn.register_forward_hook(hook)
            try:z=model(x)[0]
            finally:handle.remove()
            forwards+=1;v,mu=moments(got['h']);return z,got['h'],got['y'],mu,v
        def collect():
            bank={};zs={}
            for i in range(8):
                for k in range(4):
                    guard();z,h,y,mu,v=source(frame(i,k));bank[i,k]=(mu,v);zs[i,k]=z.cpu()
            return bank,zs
        def generate(bank,zs,repeat=False,frozen=None):
            nonlocal forwards,parity
            result={};tick=time.time()
            for i in range(8):
                for k in range(4):
                    guard();x=frame(i,k);z,h,y,mu,v=source(x)
                    if not torch.equal(z.cpu(),zs[i,k]) or not torch.equal(mu,bank[i,k][0]) or not torch.equal(v,bank[i,k][1]):raise ValueError('reference pass coordinates changed')
                    sup=supporters(i);mh,vh,within,between=pool(torch.stack([bank[j,k][0] for j in sup]),torch.stack([bank[j,k][1] for j in sup]))
                    if i in sup:raise AssertionError('query in own support')
                    zz={'SOURCE':z};yy={}
                    for arm in ARMS[1:]:
                        def replace(m,args,base,arm=arm):
                            if not torch.equal(args[0],h):raise ValueError('upstream input changed')
                            if arm=='CURRENT_RB':val=output(m,args[0],mu,v,'RB_BN')
                            elif arm=='CROSS_SCALE':val=anchored_output(m,args[0],mu,vh)
                            else:val=relative_output(base,bank[i,0][1],v,m.eps)
                            yy[arm]=val.detach();return val
                        handle=bn.register_forward_hook(replace)
                        try:zz[arm]=model(x)[0]
                        finally:handle.remove()
                        forwards+=1
                        mean_check(y,yy[arm])
                    if k==0 and not torch.equal(zz['SOURCE'],zz['RELATIVE_SCALE']):raise ValueError('original relative identity failed')
                    for a in ARMS:
                        cur=zz[a].cpu();result[i,k,a]=cur
                        if repeat:
                            if not torch.equal(cur,frozen[i,k,a]):raise ValueError('all output repeat differs')
                        else:
                            pred=cur.numpy()[0]>=0;ix=len(rows)
                            with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(np.packbits(pred).tobytes())
                            rows.append(dict(index=ix,case=i,condition=k,arm=a,nesting_violation_pixels=int((pred[1]&~pred[0]).sum()),logit_rms=float((zz[a]-z).square().mean().sqrt()),changed_pixels=int((pred!=(z.cpu().numpy()[0]>=0)).sum())))
                            if a in ('SOURCE','CURRENT_RB'):
                                key='SOURCE' if a=='SOURCE' else 'RB_BN';j=oldmap[i,k,key]
                                if (Path(c['a0_root'])/'predictions'/f'{j:04d}.bits').read_bytes()!=np.packbits(pred).tobytes():raise ValueError('unchanged A0 output parity failed')
                                parity+=1
                    if not repeat:
                        ss=(bn.running_var.double()+bn.eps).sqrt();st=(v+bn.eps).sqrt();sh=(vh+bn.eps).sqrt();s0=(bank[i,0][1]+bn.eps).sqrt()
                        diag=dict(current_ratio_min=float((ss/st).min()),current_ratio_max=float((ss/st).max()),cross_ratio_min=float((ss/sh).min()),cross_ratio_max=float((ss/sh).max()),relative_ratio_min=float((s0/st).min()),relative_ratio_max=float((s0/st).max()),pool_within_mean=float(within.mean()),pool_between_mean=float(between.mean()),query_support_overlap=0)
                        for a in ARMS[1:]:diag[a+'_mean_max_error']=mean_check(y,yy[a])
                        diagnostics.append(diag);status.update(unique_inputs=i*4+k+1,predictions=len(rows))
                        if status['unique_inputs']==2:
                            avg=(time.time()-tick)/2;projection=time.time()-started+((64-2)*avg+32*reference_mean)*1.35+60;admitted=projection<=1180
                            save(root/'QUALIFICATION.json',dict(status='PASS' if admitted else 'NOT_ADMITTED',projected_worker_seconds=projection,measured_generation_input_seconds=avg,reference_pass_repeated=True,final_label_reads=0))
                            if not admitted:raise TimeoutError('fixed repeat coverage not admitted')
                            status['status']='GENERATING'
                        if status['unique_inputs']%8==0:save(root/'RUN_STATE.json',status)
            return result
        with torch.no_grad():
            reference_start=time.time();bank,zs=collect();reference_mean=(time.time()-reference_start)/32;frozen=generate(bank,zs)
            bank2,zs2=collect()
            if any(not torch.equal(bank[q][j],bank2[q][j]) for q in bank for j in (0,1)):raise ValueError('moment bank repeat failed')
            generate(bank2,zs2,True,frozen)
            if any(not torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items()) or any(p.grad is not None for p in model.parameters()):raise ValueError('source changed')
        assert len(rows)==128 and forwards==320 and parity==64
        save(root/'private/predictions.json',rows);save(root/'private/scale_diagnostics.json',diagnostics)
        save(root/'FULL_REPEATABILITY.json',dict(status='PASS',unique_inputs=32,logit_arrays=128,all_arms_exact=True,moment_bank_exact=True,scope='Same-process full source and intervention encoder rerun; no cross-process claim'))
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',unique_inputs=32,unique_readouts=128,logical_stream_readouts=320,source_parameter_updates=0,optimizer_steps=0,backwards=0,model_forwards=forwards,source_parameters_and_buffers_immutable=True,bn_layer=c['bn_layer'],bn_channels=64,upstream_inputs_exact=True,a0_unchanged_bit_parity=parity,query_in_support=0,support_images_per_query=4,original_relative_logit_identity=True,oracle_condition_matching=True,not_online_method=True))
        save(root/'ACCESS_AUDIT.json',dict(registered_rgb_files=len(opened),source_label_reads=0,evaluation_label_reads=0,prior_outcome_score_reads=0,posthoc_robustness_denied=True,new_source_training=False,offline_cross_content_support=True,oracle_condition_information=True,same_content_original_reference=True,causal_online_test=False,patient_independence=False))
        out=[]
        for key in diagnostics[0]:
            vals=[r[key] for r in diagnostics];out.append(dict(measure=key,mean=float(np.mean(vals)),min=float(min(vals)),max=float(max(vals))))
        for a in ARMS:
            for key in ('logit_rms','changed_pixels'):
                vals=[r[key] for r in rows if r['arm']==a];out.append(dict(measure=a+'_'+key,mean=float(np.mean(vals)),min=float(min(vals)),max=float(max(vals))))
        save(root/'public/DIAGNOSTICS.json',out)
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
    status=read(root/'RUN_STATE.json');assert status['status']=='GPU_COMPLETE' and status['predictions']==128 and status['unique_inputs']==32
    assert read(root/'FULL_REPEATABILITY.json')['all_arms_exact']
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json');assert len(panel)==8
    expected={(i,k,a) for i in range(8) for k in range(4) for a in ARMS}
    assert len(rows)==128 and {(r['case'],r['condition'],r['arm']) for r in rows}==expected
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
    contrasts.append(summary('CROSS_SCALE-CURRENT_RB:OC_DICE',cubes['CROSS_SCALE'][:,:,1]-cubes['CURRENT_RB'][:,:,1]))
    contrasts.append(summary('RELATIVE_SCALE-CURRENT_RB:OC_DICE',cubes['RELATIVE_SCALE'][:,:,1]-cubes['CURRENT_RB'][:,:,1]))
    primary=next(v for v in contrasts if v['comparison']==PRIMARY)
    base=absolute[0];checks=[]
    for a in ARMS[1:]:
        row=next(v for v in absolute if v['arm']==a);oc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:OC_DICE');allc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:ALL_DICE');returned=next(v for v in contrasts if v['comparison']==a+'-SOURCE:RETURN_DICE')
        key={v['region']:v for v in regions if v['arm']==a and v['channel']=='OC'}
        gates=dict(oc_mean=oc['delta_pp']>=.5,oc_ci=oc['ci95_pp'][0]>0,all_ci=allc['ci95_pp'][0]>0,worst=allc['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,oc_foreground_net=key['foreground']['net_ci_low_pp'] is not None and key['foreground']['net_ci_low_pp']>0,oc_boundary_net=key['boundary3px']['net_ci_low_pp'] is not None and key['boundary3px']['net_ci_low_pp']>0,structure=all(row[f] is not None and row[f]<=base[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')),coverage=row['surface_defined']>=base['surface_defined'] and row['vcdr_defined']>=base['vcdr_defined'],empty_nesting=row['empty_predictions']<=base['empty_predictions'] and row['nesting_violation_pixels']<=base['nesting_violation_pixels'])
        checks.append(dict(arm=a,gates=gates,worth_followup=all(gates.values())))
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,absolute=absolute,signal_checks=checks,primary_pass=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0,advance_to_history_protocol=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0 and next(v['worth_followup'] for v in checks if v['arm']=='CROSS_SCALE'),static_only=True,parameter_updates=0,continual_benefit_tested=False,bn_layer=c['bn_layer'],oracle_same_condition_support=True,same_content_original_reference=True,deployable_online_method=False,unique_inputs=32,unique_predictions=128,logical_stream_readouts=320,patient_independence=False,scope='Eight reused development contents; weighted conditions [0,1,2,3,0], two identical readout orders; no independent-patient inference'))
    for n,rr in [('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('region_transitions',regions)]:csvout(root/'public'/f'{n}.csv',rr)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',evaluation_label_files=8,unique_inputs=32,unique_metric_evaluations=128,absolute_rows=4,region_rows=32,cpu_seconds=time.time()-started))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='worker':worker(c)
    elif mode=='score':score(c)
    elif mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS'
        sys.exit(controller(c))
    else:raise ValueError('unknown mode')
