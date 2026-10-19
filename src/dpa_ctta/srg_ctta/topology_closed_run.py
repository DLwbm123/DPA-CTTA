"""P4 finite-domain 2D topology/integrity per-image RESET scientific qualification."""
import os,sys,time,copy,json
from pathlib import Path
import numpy as np
import torch
from .run import read,save,guard_assets
from .storage_control import controller
from .cpu_control import bytes_used
from .persistent_topology import critical_pairs
from .topology_integrity_closed import refine
from .topology_projection import project
from .static_teacher import summary,region_counts
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..r78_cross_patient.score import metric,csvout,interval
from .score import unpack
STATIC=('SOURCE','SMOOTH','FILL_SOURCE','TOPO_SOURCE','TOPO_SMOOTH')
OPTIMIZED=('TIB','PERMUTED','COR','INTEGRITY')
NATIVE=('CURRENT_BN','CW1_RESET','CW2_RESET')
ARMS=STATIC+NATIVE+OPTIMIZED
PRIMARY='TIB-PERMUTED:OC_DICE'

def validate(c):
    assert c['design_version']=='TOPOLOGY_INTEGRITY_CLOSED_P4' and c['primary']==PRIMARY
    assert c['arms']==list(ARMS) and c['steps']==90 and c['lr']==.0001 and c['weight_decay']==.01
    assert c['alpha']==10000 and c['beta']==1000 and c['dense_last_step']==30 and c['refresh_interval']==3 and c['late_weight']==.1
    assert c['unique_inputs']==32 and c['expected_readouts']==384 and c['expected_optimizer_steps']==11880
    assert c['seed']==20261018 and c['grouping']=='CONTENT_DEVELOPMENT_USER_AUTHORIZED'
    assert c['stream_blocks']==[0,1,2,3,0] and c['orders']==['forward','reverse']
    assert c['time_gpu_budgets_removed'] and c['output_cap_bytes']==2*1024**3
    assert c['empty_rule']=='STRUCTURAL_SIMILARITY_ZERO_IF_UNDEFINED_NO_EPSILON'
    assert c['history']==False and c['repeat_first_input_all_optimized']
    assert c['skeleton']=='BINARY_SOFT_SKELETON_TO_EROSION_EXHAUSTION_CROSS_SQUARE_ZERO_EXTERIOR'
    assert c['trainable']=='ALL_EXISTING_PARAMETERS_EVAL_BN_BUFFERS_FROZEN'
    assert not any(c[k] for k in ('rdc','engram','new_training','patient_independence'))

def static_outputs(z):
    from torch.nn import functional as F
    from scipy.ndimage import binary_fill_holes,generate_binary_structure
    p=z.sigmoid();axis=z.new_tensor([1,2,1])/4;kernel=(axis[:,None]*axis[None,:]).expand(2,1,3,3)
    q=F.conv2d(F.pad(p,(1,1,1,1),mode='reflect'),kernel,groups=2)
    smooth=z.clone();smooth[...,:-1,:-1]=torch.logit(q[...,:-1,:-1].clamp(1e-12,1-1e-12))
    source=(z.cpu().numpy()[0]>=0);sm=(smooth.cpu().numpy()[0]>=0)
    bg=generate_binary_structure(2,1)
    return dict(SOURCE=source,SMOOTH=sm,FILL_SOURCE=np.stack([binary_fill_holes(a,structure=bg) for a in source]),TOPO_SOURCE=np.stack([project(a) for a in source]),TOPO_SMOOTH=np.stack([project(a) for a in sm]))

def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r37_cw_orientation.method import Host as Native,candidates
    root=Path(c['output_root']);started=time.time();status=dict(status='QUALIFYING',unique_inputs=0,predictions=0,optimizer_steps=0,scientific_optimizer_steps=0)
    save(root/'RUN_STATE.json',status)
    try:
        validate(c);torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        # Strict CUDA loss/backward qualification uses synthetic probabilities only.
        from .topology_integrity_closed import objective,weighted_support
        toy=torch.full((2,9,9),.2,dtype=torch.float64,device='cuda');toy[:,2:7,2:7]=.9;toy[:,4,4]=.3;toy.requires_grad_()
        loss,_,_,_=objective(toy,toy.detach(),weighted_support(toy.detach()),0,'TIB',None);loss.backward()
        assert torch.isfinite(toy.grad).all();del toy,loss
        panel=read(root/'private/panel.json');assert len(panel)==8 and len({r['group_id'] for r in panel})==8 and all('mask_path' not in r for r in panel)
        opened=guard_assets(c,panel)
        def block_failed_prior(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
                if Path(os.fsdecode(args[0])).resolve().is_relative_to(Path(c['failed_prior_root']).resolve()):raise PermissionError('failed P3 runtime denied')
        sys.addaudithook(block_failed_prior)
        for p in (root/'scorer/panel.json',Path(c['state_prior_root'])/'public/RESULTS.json'):
            try:p.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('label/outcome guard failed')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        base,_=build_reference_model('fundus');base.load_state_dict(state);base.cuda().eval().requires_grad_(False)
        immutable={k:v.detach().cpu().clone() for k,v in base.state_dict().items()};model=copy.deepcopy(base)
        rows=[];trace=[];first={};forwards_static=0;native_backwards=0;native_forwards=0
        def model_state():return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        def exact(a,b):
            if torch.is_tensor(a):return torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(exact(a[k],b[k]) for k in a)
            return a==b
        def reset():model.load_state_dict(immutable);model.zero_grad(set_to_none=True)
        def progress(step,detail):
            status['optimizer_steps']+=1
            if status['status']!='REPEATING':status['scientific_optimizer_steps']+=1
            row=dict(total_step=status['optimizer_steps'],phase=status['status'],arm=status['arm'],case_index=status['case_index'],condition_index=status['condition_index'],**detail)
            with (root/'private/optimization_steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            status.update(arm_step=step,arm_completed_steps=step+1,worker_body_seconds=time.time()-started)
            tmp=root/'RUN_STATE.progress.tmp';tmp.write_text(json.dumps(status));tmp.replace(root/'RUN_STATE.json')
        def put(i,k,arm,p,z0,z=None):
            ix=len(rows)
            with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(np.packbits(p).tobytes())
            rows.append(dict(index=ix,case=i,condition=k,arm=arm,nesting_violation_pixels=int((p[1]&~p[0]).sum()),changed_pixels=int((p!=z0).sum()),logit_rms=float(z) if z is not None else None))
            status['predictions']=len(rows)
        for i,item in enumerate(panel):
            pixels=read_pixels(item['image_path'],'fundus',item['image_size'])
            for k,spec in enumerate(FINAL):
                status.update(case_index=i,condition_index=k,status='OPTIMIZING')
                x=normalize(perturb(pixels,spec,SEED+i)).cuda()
                with torch.no_grad():z0=base(x)[0].double();static=static_outputs(z0);forwards_static+=1
                if i==0 and k==0:
                    import subprocess
                    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True)
                    own=[v for v in raw.splitlines() if v.split(',')[0].strip()==str(os.getpid())]
                    assert len(own)==1 and c['gpu']['uuid'] in own[0] and all(t not in own[0].lower() for t in ('wangbomin','ctta','topology'))
                    save(root/'GPU_VISIBLE.private.json',dict(status='PASS',line=own[0]))
                for a in STATIC:put(i,k,a,static[a],static['SOURCE'])
                for steps in (1,2):
                    conf=copy.deepcopy(candidates()[3]);conf.setdefault('params',{})['online_steps_per_arrival']=steps
                    host=Native(state,conf,SEED,'finite_stream',device='cuda:0')
                    perturbed=perturb(pixels,spec,SEED+i)
                    if steps==1:
                        with torch.no_grad():pre=host._readonly(host.native.model,perturbed)
                        put(i,k,'CURRENT_BN',pre.detach().cpu().numpy()[0]>=0,static['SOURCE'])
                        native_forwards+=1
                    post,d=host.step(perturbed)
                    native_backwards+=sum(v['counts']['backwards'] for v in d['native']);native_forwards+=sum(v['counts']['forwards'] for v in d['native'])
                    put(i,k,'CW'+str(steps)+'_RESET',post.detach().cpu().numpy()[0]>=0,static['SOURCE'])
                    del host,post,d
                
                for a in OPTIMIZED:
                    reset();status['arm']=a;z,t,opt=refine(model,x,a,90,progress)
                    if any(not torch.equal(v.detach().cpu(),immutable[n]) for n,v in model.named_buffers()):raise ValueError('BN/source buffer changed')
                    if i==0 and k==0:first[a]=(z.detach().cpu().clone(),model_state(),opt)
                    pred=z.detach().cpu().numpy()[0]>=0;put(i,k,a,pred,static['SOURCE'],float((z-z0).square().mean().sqrt()))
                    trace.extend(dict(case=i,condition=k,arm=a,**v) for v in t)
                    save(root/'private/optimization_trace.json',trace)
                status.update(unique_inputs=i*4+k+1,arm_step=90);save(root/'RUN_STATE.json',status)
                if i==0 and k==0:save(root/'QUALIFICATION.json',dict(status='PASS',strict_cuda_backward=True,first_input_all_four_arms_90_steps=True,time_gpu_quotas_removed=True,final_label_reads=0))
        # Full end model/Adam and output repeat for all optimized arms on the first input.
        item=panel[0];pixels=read_pixels(item['image_path'],'fundus',item['image_size']);x=normalize(perturb(pixels,FINAL[0],SEED)).cuda()
        status['status']='REPEATING'
        for a in OPTIMIZED:
            reset();status['arm']=a;z,t,opt=refine(model,x,a,90,progress)
            if not torch.equal(z.cpu(),first[a][0]) or not exact(model_state(),first[a][1]) or not exact(opt,first[a][2]):raise ValueError('full first-input repeat differs')
        if any(not torch.equal(v.cpu(),immutable[n]) for n,v in base.state_dict().items()):raise ValueError('frozen source changed')
        assert len(rows)==384 and status['scientific_optimizer_steps']==11520 and status['optimizer_steps']==11880
        save(root/'private/predictions.json',rows)
        diag=[]
        for a in OPTIMIZED:
            t=[v for v in trace if v['arm']==a];rr=[v for v in rows if v['arm']==a]
            diag.append(dict(arm=a,optimizer_steps=len(t),ph_computations=sum(v['ph_refresh'] for v in t)*2,initial_objective_mean=float(np.mean([v['loss'] for v in t if v['step']==0])),last_objective_mean=float(np.mean([v['loss'] for v in t if v['step']==89])),gradient_norm_mean=float(np.mean([v['gradient_l2'] for v in t])),logit_rms_mean=float(np.mean([v['logit_rms'] for v in rr])),changed_pixels_mean=float(np.mean([v['changed_pixels'] for v in rr])),empty_current_head_steps=sum(v['empty_current_heads'] for v in t),empty_previous_head_steps=sum(v['empty_previous_heads'] for v in t),undefined_similarity_head_steps=sum(v['undefined_similarity_heads'] for v in t)))
        save(root/'public/DIAGNOSTICS.json',diag)
        save(root/'FULL_REPEATABILITY.json',dict(status='PASS',optimized_arms=4,first_input_90step_logit_arrays=4,all_arms_exact=True,final_model_exact=True,final_adam_exact=True,scope='First input all four optimized arms, full encoder and optimizer rerun, same process; not all 32 inputs/cross-process'))
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',unique_inputs=32,unique_readouts=384,scientific_optimizer_steps=11520,repeat_optimizer_steps=360,total_optimizer_steps=11880,source_parameters_and_buffers_immutable=True,bn_buffers_changed=False,training='target_only_no_source_training',reset_each_image=True,trainable='all_existing_parameters',steps_per_arm=90,skeleton_empty_rule='zero_structural_similarity_no_epsilon_no_source_fallback',strict_cuda=True,static_forwards=forwards_static,native_arrivals=64,native_backwards=native_backwards,native_forwards=native_forwards,core_model_forwards=forwards_static+132*92))
        save(root/'ACCESS_AUDIT.json',dict(registered_rgb_files=len(opened),evaluation_label_reads=0,prior_outcome_score_reads=0,new_source_training=False,future_images_used_for_update=False,patient_independence=False,previously_exposed_development_contents=True))
        status.update(status='GPU_COMPLETE',worker_body_seconds=time.time()-started,output_bytes=bytes_used(root),peak_allocated_bytes=torch.cuda.max_memory_allocated());save(root/'RUN_STATE.json',status)
    except BaseException as e:
        if 'rows' in locals():save(root/'private/predictions.json',rows)
        if 'trace' in locals():save(root/'private/partial_completed_trace.json',trace)
        status.update(status='NOT_EVALUABLE',error_type=type(e).__name__,reason=str(e),worker_body_seconds=time.time()-started);save(root/'RUN_STATE.json',status);raise

def score(c):
    from ..source_io import read_mask
    from scipy.ndimage import binary_erosion,distance_transform_edt
    validate(c);root=Path(c['output_root']);started=time.time();ret=read(root/'processes/worker.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not ret['active'] and ret['exit_code']==0
    status=read(root/'RUN_STATE.json');assert status['status']=='GPU_COMPLETE' and status['predictions']==384 and status['unique_inputs']==32
    assert read(root/'FULL_REPEATABILITY.json')['all_arms_exact']
    rows=read(root/'private/predictions.json');panel=read(root/'scorer/panel.json');assert len(panel)==8
    prior=Path(c['state_prior_root']);oldrows=read(prior/'private/predictions.json');old={(v['case'],v['condition'],v['arm']):v for v in oldrows}
    parity=0
    for row in rows:
        if row['arm'] in STATIC:
            v=old[row['case'],row['condition'],row['arm']]
            assert (root/'predictions'/f"{row['index']:04d}.bits").read_bytes()==(prior/'predictions'/f"{v['index']:04d}.bits").read_bytes()
            parity+=1
    assert parity==160
    save(root/'STATIC_PARITY.json',dict(status='PASS',bit_arrays=parity,scope='Only scorer after worker retirement reads sealed P2 bits; no metrics reused'))
    expected={(i,k,a) for i in range(8) for k in range(4) for a in ARMS}
    assert len(rows)==384 and {(r['case'],r['condition'],r['arm']) for r in rows}==expected
    labels=[read_mask(r['mask_path'],'fundus',r['image_size']).numpy()[0].astype(bool) for r in panel]
    masks={(r['case'],r['condition'],r['arm']):unpack(root/'predictions'/f"{r['index']:04d}.bits")[1] for r in rows}
    metrics={};counts={};individual=[]
    for row in rows:
        pass # no user CPU quota
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
        if control in ('SOURCE','TIB'):continue
        contrasts.append(summary('TIB-'+control+':OC_DICE',cubes['TIB'][:,:,1]-cubes[control][:,:,1]))
    primary=next(v for v in contrasts if v['comparison']==PRIMARY)
    base=absolute[0];checks=[]
    for a in ARMS[1:]:
        row=next(v for v in absolute if v['arm']==a);oc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:OC_DICE');allc=next(v for v in contrasts if v['comparison']==a+'-SOURCE:ALL_DICE');returned=next(v for v in contrasts if v['comparison']==a+'-SOURCE:RETURN_DICE')
        key={v['region']:v for v in regions if v['arm']==a and v['channel']=='OC'}
        gates=dict(oc_mean=oc['delta_pp']>=.5,oc_ci=oc['ci95_pp'][0]>0,all_ci=allc['ci95_pp'][0]>0,worst=allc['worst_cell_pp']>=-1,return_no_harm=returned['delta_pp']>=-.2,oc_foreground_net=key['foreground']['net_ci_low_pp'] is not None and key['foreground']['net_ci_low_pp']>0,oc_boundary_net=key['boundary3px']['net_ci_low_pp'] is not None and key['boundary3px']['net_ci_low_pp']>0,structure=all(row[f] is not None and row[f]<=base[f] for f in ('assd_pixel','hd95_pixel','vcdr_mae')),coverage=row['surface_defined']>=base['surface_defined'] and row['vcdr_defined']>=base['vcdr_defined'],empty_nesting=row['empty_predictions']<=base['empty_predictions'] and row['nesting_violation_pixels']<=base['nesting_violation_pixels'])
        checks.append(dict(arm=a,gates=gates,worth_followup=all(gates.values())))
    controls={a:next(v for v in contrasts if v['comparison']=='TIB-'+a+':OC_DICE')['ci95_pp'][0]>0 for a in ('PERMUTED','INTEGRITY','COR','SMOOTH','FILL_SOURCE','TOPO_SOURCE','TOPO_SMOOTH','CURRENT_BN','CW1_RESET','CW2_RESET')}
    save(root/'public/RESULTS.json',dict(status='COMPLETE',primary=primary,contrasts=contrasts,absolute=absolute,signal_checks=checks,mechanism_primary_pass=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0,advance_to_new_protocol=primary['delta_pp']>=.5 and primary['ci95_pp'][0]>0 and all(controls.values()) and next(v['worth_followup'] for v in checks if v['arm']=='TIB'),static_only=True,parameter_updates=c['expected_optimizer_steps'],continual_benefit_tested=False,strong_control_checks=controls,unique_inputs=32,unique_predictions=384,logical_stream_readouts=960,patient_independence=False,scope='Per-image RESET target optimization, not CTTA; native CW1/CW2 use current-image BN semantics; eight exposed contents, no independent patients'))
    for n,rr in [('absolute',absolute),('condition_channel',channels),('contrasts',contrasts),('region_transitions',regions)]:csvout(root/'public'/f'{n}.csv',rr)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',evaluation_label_files=8,unique_inputs=32,unique_metric_evaluations=384,absolute_rows=12,region_rows=96,condition_channel_rows=96,contrast_rows=len(contrasts),cpu_seconds=time.time()-started))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);validate(c);mode=os.environ['DIAG_MODE']
    if mode=='worker':worker(c)
    elif mode=='score':score(c)
    elif mode=='controller':
        assert read(Path(c['output_root'])/'CPU_CHECK.json')['status']=='PASS'
        sys.exit(controller(c))
    else:raise ValueError('unknown mode')
