"""Finite label-free continual adaptation, with prequential and post-update readouts."""
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .method import ARMS,NORMALIZED,Host,intensity_views
from ..r78_cross_patient.objective import BNState,Field,reference,direct_logits
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
from ..source_correction.method import correction
from ..boundary_amplitude.run import controller
from ..r51_same_state.run import read,save

LEARNERS=ARMS+('CW1','CW2')
POLICIES=('RESET','RETAIN')
STAGES=('PRE','POST')
REFERENCES=('SOURCE_EVAL','CURRENT_BN','D_OLD','R_OLD','R_SHUFFLE_OLD')
BLOCKS=(0,1,2,3,0)
EXPECTED=3600


def schedule(order):
    return [dict(case=case,block=block,condition=condition,arrival=block*8+pos)
            for block,condition in enumerate(BLOCKS)
            for pos,case in enumerate(range(8) if order==0 else reversed(range(8)))]


def guard_assets(c,panel):
    root=Path(c['output_root']).resolve();data=Path(c['source_root']).resolve()
    images={Path(r['image_path']).resolve() for r in panel};opened=set()
    old=[Path(c[k]).resolve() for k in ('old_head_root','prior_root')]
    state_prior=Path(c['state_prior_root']).resolve() if c.get('reset_optimizer_per_arrival') else None
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if state_prior is not None and p.is_relative_to(state_prior):
            raise PermissionError('previous state experiment is denied to adaptation')
        if p.is_relative_to(root/'scorer') or any(p.is_relative_to(d/f) for d in old for f in ('scorer','public')):
            raise PermissionError('labels and previous outcome scores are denied')
        if p.is_relative_to(data):
            if p not in images:raise PermissionError('only eight registered stream RGB files allowed')
            opened.add(str(p))
    sys.addaudithook(audit);return opened


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r37_cw_orientation.method import Host as Native,candidates
    root=Path(c['output_root']);started=time.time();status=dict(status='QUALIFYING',started=started,predictions=0,arrivals=0,legacy_parity=0)
    save(root/'RUN_STATE.json',status)
    def guard():
        if time.time()-started>c['gpu_cap_seconds']-20:raise TimeoutError('fixed GPU worker cap')
    try:
        if c['rdc'] or c['engram'] or c['new_training'] or c['grouping']!='CONTENT_DEVELOPMENT_USER_AUTHORIZED':raise ValueError('scope mismatch')
        ledger=read(Path(c['old_head_root'])/'RESOURCE_LEDGER.json')
        if not ledger['all_workers_retired'] or any(a['exit_code'] for a in ledger['attempts'][-2:]):raise ValueError('previous run must be complete')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=read(root/'private/panel.json')
        if len(panel)!=8 or len({r['group_id'] for r in panel})!=8 or any('mask_path' in r for r in panel):raise ValueError('eight content-only stream records required')
        opened=guard_assets(c,panel)
        for denied in (root/'scorer/panel.json',Path(c['old_head_root'])/'public/RESULTS.json'):
            try:denied.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('access guard did not reject forbidden file')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        base,_=build_reference_model('fundus');base.load_state_dict(state);base.cuda().eval().requires_grad_(False)
        legacy=BNState(copy.deepcopy(base));headroot=Path(c['old_head_root']);prior=Path(c['prior_root'])
        heads={}
        for arm,folder,key,epoch in (('D',prior,'D',150),('R',headroot,'R',100)):
            payload=torch.load(folder/'private'/f'head_{key}.pt',map_location='cpu',weights_only=True)
            if payload['selected_episode']!=epoch:raise ValueError('frozen source head changed')
            head=Field().cuda().eval().requires_grad_(False);head.load_state_dict(payload['state_dict']);heads[arm]=head
        before={n:p._version for n,p in base.named_parameters()};buffers={n:b.clone() for n,b in base.named_buffers()}
        images=[read_pixels(r['image_path'],'fundus',r['image_size']) for r in panel]
        oldrows=read(headroot/'private/predictions.json');lookup={(r['pair'],r['condition'],r['arm']):r['index'] for r in oldrows if r['mode']=='DIRECT'}
        legacy_names=dict(CURRENT_BN='B0',D_OLD='D_OLD',R_OLD='R',R_SHUFFLE_OLD='R_SHUFFLE')
        cache={};diags={};records=[];step_rows=[];reset_predictions={};reset_mismatches=0
        def prediction(z):return z.detach().cpu().numpy()[0]>=0
        def diagnostics(z,z0):
            dz=z-z0;dp=z.sigmoid()-z0.sigmoid()
            return dict(logit_rms=float(dz.square().mean().sqrt()),probability_abs_mean=float(dp.abs().mean()),changed_pixels=int((prediction(z)!=prediction(z0)).sum()))
        def append(row,pred,d):
            ix=len(records)
            with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f:f.write(np.packbits(pred).tobytes())
            records.append(dict(index=ix,**row,diagnostics=d,nesting_violation_pixels=int((pred[1]&~pred[0]).sum())))
            status['predictions']=len(records)
        # Frozen predictors are evaluated once per unique image/condition, then reused
        # at repeated arrivals; the adapting learner never consumes this cache.
        for case in range(8):
            for k,spec in enumerate(FINAL):
                guard();pixels=perturb(images[case],spec,SEED+case);x=normalize(pixels).cuda()
                with torch.no_grad():
                    ze=base(x)[0];zc=legacy(x,legacy.zero());ref=reference(x,zc)
                    outputs=dict(SOURCE_EVAL=ze,CURRENT_BN=zc,D_OLD=direct_logits(zc,ref,heads['D'](ref.inputs)),R_OLD=correction(zc,ref,heads['R'],'R'),R_SHUFFLE_OLD=correction(zc,ref,heads['R'],'R_SHUFFLE'))
                    for arm,z in outputs.items():
                        pred=prediction(z);cache[case,k,arm]=pred;diags[case,k,arm]=diagnostics(z,ze)
                        if arm in legacy_names:
                            ix=lookup[case,k,legacy_names[arm]];old=(headroot/'predictions'/f'{ix:04d}.bits').read_bytes()
                            if np.packbits(pred).tobytes()!=old:raise ValueError('legacy reference parity failed: '+arm)
                            status['legacy_parity']+=1
        for order in range(2):
            for arrival in schedule(order):
                for arm in REFERENCES:
                    key=arrival['case'],arrival['condition'],arm
                    append(dict(**arrival,order=order,arm=arm,policy='REFERENCE',stage='STATIC'),cache[key],diags[key])
        save(root/'RUN_STATE.json',status)
        normalization_inventory=None;norm_checks=[];core_forwards=0;native_forwards=0;native_backwards=0
        trajectories=[]
        run_arms=('STRAT','RANDOM','ALL','GLOBAL','NOANCHOR','ENT_NORM','RG','ENT')
        for arm in run_arms:
            for policy in POLICIES:
                for order in range(2):
                    host=Host(base,arm)
                    inventory=(len(host.names),sum(p.numel() for p in host.params))
                    if inventory!=(82,19136):raise ValueError('fixed affine inventory differs')
                    normalization_inventory=inventory
                    qualification_start=time.time()
                    for pos,arrival in enumerate(schedule(order)):
                        guard();case=arrival['case'];k=arrival['condition']
                        if policy=='RESET':host.reset()
                        elif c.get('reset_optimizer_per_arrival'):host.reset_optimizer()
                        pixels=perturb(images[case],FINAL[k],SEED+case)
                        w,s=intensity_views(pixels,SEED+10000+case*10+k)
                        x,xw,xs=[normalize(p).cuda() for p in (pixels,w,s)]
                        with torch.no_grad():z0=base(x)[0]
                        pre,post,gate,gate_pred,d=host.step(x,xw,xs,z0,SEED+20000+case*10+k)
                        core_forwards+=d['forwards']+1
                        for stage,z in (('PRE',pre),('POST',post)):
                            pred=prediction(z)
                            if policy=='RESET' and arrival['block'] in (0,4):
                                key=(arm,order,case,stage)
                                if arrival['block']==0:reset_predictions[key]=pred.copy()
                                else:reset_mismatches+=int((pred!=reset_predictions[key]).sum())
                            append(dict(**arrival,order=order,arm=arm,policy=policy,stage=stage),pred,dict(**diagnostics(z,z0),total_affine_norm=d['total_affine_norm'],optimizer_age=d['optimizer_age']))
                        gi=len(step_rows)
                        with (root/'gates'/f'{gi:04d}.bits').open('xb') as f:f.write(np.packbits(gate.cpu().numpy()[0]).tobytes())
                        with (root/'gate_predictions'/f'{gi:04d}.bits').open('xb') as f:f.write(np.packbits(prediction(gate_pred)).tobytes())
                        step_rows.append(dict(index=gi,**arrival,order=order,arm=arm,policy=policy,**d))
                        norm_checks.extend(dict(arm=arm,**v) for v in d['steps'])
                        status['arrivals']+=1
                        if arm=='STRAT' and policy=='RESET' and order==0 and pos==1:
                            avg=(time.time()-qualification_start)/2
                            projection=time.time()-started+(1280-2)*avg*1.8+320*.25+60
                            admitted=projection<=c['gpu_cap_seconds']-20
                            save(root/'QUALIFICATION.json',dict(status='PASS' if admitted else 'NOT_ADMITTED',projected_worker_seconds=projection,measured_arrival_seconds=avg,affine_parameters=inventory[1],affine_tensors=inventory[0],first_two_arrivals_in_budget=True,legacy_parity=status['legacy_parity'],final_label_reads=0))
                            if not admitted:raise TimeoutError('full frozen matrix not admitted')
                            status['status']='GENERATING'
                        if pos%8==7:status.update(current_arm=arm,policy=policy,order=order,position=pos+1);save(root/'RUN_STATE.json',status)
                    trajectories.append(dict(arm=arm,policy=policy,order=order,arrivals=40,optimizer_steps=host.total_steps))
                    del host
        for steps in (1,2):
            arm='CW'+str(steps)
            for policy in POLICIES:
                for order in range(2):
                    conf=copy.deepcopy(candidates()[3]);conf.setdefault('params',{})['online_steps_per_arrival']=steps
                    host=Native(state,conf,SEED,'finite_stream',device='cuda:0');initial=host.snapshot()
                    try:
                        for pos,arrival in enumerate(schedule(order)):
                            guard();case=arrival['case'];k=arrival['condition']
                            if policy=='RESET':host.restore(initial)
                            pixels=perturb(images[case],FINAL[k],SEED+case);x=normalize(pixels).cuda()
                            with torch.no_grad():z0=base(x)[0];pre=host._readonly(host.native.model,pixels)
                            post,d=host.step(pixels);post=post.cuda()
                            native_forwards+=1+sum(v['counts']['forwards'] for v in d['native']);native_backwards+=sum(v['counts']['backwards'] for v in d['native'])
                            for stage,z in (('PRE',pre),('POST',post)):
                                pred=prediction(z)
                                if policy=='RESET' and arrival['block'] in (0,4):
                                    key=(arm,order,case,stage)
                                    if arrival['block']==0:reset_predictions[key]=pred.copy()
                                    else:reset_mismatches+=int((pred!=reset_predictions[key]).sum())
                                append(dict(**arrival,order=order,arm=arm,policy=policy,stage=stage),pred,dict(**diagnostics(z,z0),optimizer_age=host.native.steps))
                            status['arrivals']+=1
                            if pos%8==7:status.update(current_arm=arm,policy=policy,order=order,position=pos+1);save(root/'RUN_STATE.json',status)
                        trajectories.append(dict(arm=arm,policy=policy,order=order,arrivals=40,optimizer_steps=40*steps))
                    finally:host.close()
        if len(records)!=EXPECTED or len(step_rows)!=1280 or status['legacy_parity']!=128 or status['arrivals']!=1600:raise ValueError('fixed matrix incomplete')
        if any(p._version!=before[n] or p.grad is not None for n,p in base.named_parameters()):raise ValueError('source anchor changed')
        if any(not torch.equal(b,buffers[n]) for n,b in base.named_buffers()):raise ValueError('source anchor buffers changed')
        save(root/'private/predictions.json',records);save(root/'private/updates.json',step_rows)
        matched=[v for v in norm_checks if v['arm'] in NORMALIZED]
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',trajectories=trajectories,stream_arrivals=1600,core_update_events=2560,normalized_update_events=len(matched),normalized_events_matched=sum(v['norm_matched'] for v in matched),zero_proposals=sum(v['zero_proposal'] for v in norm_checks),reset_return_mask_differing_pixels=reset_mismatches,legacy_parity=128,core_adaptation_forwards=core_forwards,native_forwards=native_forwards,native_backwards=native_backwards,core_loss_backward_calls=2560,core_diagnostic_anchor_gradient_calls=2560,source_affine_inventory=normalization_inventory,source_anchor_immutable=True,reset_optimizer_per_arrival=bool(c.get('reset_optimizer_per_arrival')),retained_core_state='AFFINE_ONLY' if c.get('reset_optimizer_per_arrival') else 'AFFINE_AND_ADAM'))
        save(root/'ACCESS_AUDIT.json',dict(source_image_files=len(opened),source_label_reads=0,evaluation_label_reads=0,prior_outcome_score_reads=0,new_source_training=False,method_inputs='Current image and source checkpoint only; retained core state: '+('affine parameters only' if c.get('reset_optimizer_per_arrival') else 'affine parameters and optimizer moments'),future_images_used_for_update=False,patient_independence=False,previously_exposed_development_contents=True))
        used=sum(p.stat().st_size for folder in ('private','predictions','gates','gate_predictions','logs') for p in (root/folder).rglob('*') if p.is_file())
        if used>c['output_cap_bytes']:raise ValueError('output cap')
        status.update(status='GPU_COMPLETE',output_bytes=used,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        status.update(status='NOT_EVALUABLE',error_type=type(error).__name__,reason=str(error));raise
    finally:status.update(ended=time.time(),gpu_worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);mode=os.environ['DIAG_MODE']
    if mode=='controller':sys.exit(controller(c))
    elif mode=='worker':worker(c)
    elif mode=='score':
        from .score import score
        score(c)
    else:raise ValueError('unknown mode')
