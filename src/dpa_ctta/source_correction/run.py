"""Finite source-supervised correction comparison; final labels are scorer-only."""
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch
from .method import correction, match_logit_rms
from ..r78_cross_patient.objective import BNState, Field, ParameterPredictor, reference
from ..r78_cross_patient.run import FINAL, VALIDATION, SEED, episodes, normalize, perturb, seed, supervision, dice, adapt, direct, access_guard
from ..boundary_amplitude.method import match
from ..boundary_amplitude.run import controller
from ..r51_same_state.run import read, save

TRAIN=('D_RETRAIN','R','R_BAND','R_GEOM')
DIRECT=('B0','D_OLD','D_RETRAIN','R','R_BAND','R_GEOM','R_BUDGET','R_NEG','R_SHUFFLE')
ARMS=DIRECT+('F','X_MATCH','P_MATCH','C_FULL')
MODES=('SELF','CROSS_MATCH','CROSS_MISMATCH')


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels, read_mask
    from ..r37_cw_orientation.method import Host as Native, candidates
    root=Path(c['output_root']);prior=Path(c['prior_root']);amp=Path(c['amplitude_root']);started=time.time()
    status=dict(status='QUALIFYING',started=started,training_episodes=0,predictions=0,baseline_parity=0)
    save(root/'RUN_STATE.json',status);native=None
    def guard():
        if time.time()-started>c['gpu_cap_seconds']-20: raise TimeoutError('fixed GPU worker budget')
    try:
        if c['rdc'] or c['engram'] or c['grouping']!='CONTENT_DEVELOPMENT_USER_AUTHORIZED':
            raise ValueError('scope mismatch')
        for old in (prior,amp):
            ledger=read(old/'RESOURCE_LEDGER.json')
            if not ledger['all_workers_retired'] or any(a['exit_code'] for a in ledger['attempts']):
                raise ValueError('prior worker retirement required')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.backends.cudnn.benchmark=False
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=read(root/'private/panel.json')
        if [len(panel[k]) for k in ('train','validation','support','query')]!=[16,8,8,8]: raise ValueError('fixed panel')
        if len({r['group_id'] for rows in panel.values() for r in rows})!=40: raise ValueError('content overlap')
        opened=access_guard(c,[r['mask_path'] for role in ('train','validation') for r in panel[role]])
        def prior_guard(event,args):
            if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
                path=Path(os.fsdecode(args[0])).resolve()
                if any(path.is_relative_to(d/f) for d in (prior,amp) for f in ('public','scorer')):
                    raise PermissionError('prior outcomes and final labels denied')
        sys.addaudithook(prior_guard)
        for denied in (root/'scorer/panel.json',prior/'scorer/panel.json',amp/'public/RESULTS.json'):
            try: denied.read_bytes()
            except PermissionError: pass
            else: raise AssertionError('label/outcome guard failed')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        model,_=build_reference_model('fundus');model.load_state_dict(state);model.cuda();host=BNState(model)
        before={n:p._version for n,p in model.named_parameters()};counts=dict(backbone_forwards=0)
        hook=model.register_forward_hook(lambda *a:counts.__setitem__('backbone_forwards',counts['backbone_forwards']+1))
        images={role:[read_pixels(r['image_path'],'fundus',r['image_size']) for r in rows] for role,rows in panel.items()}
        labels={role:[read_mask(r['mask_path'],'fundus',r['image_size']).cuda() for r in panel[role]] for role in ('train','validation')}
        heads={};opts={}
        for arm in TRAIN:
            seed();heads[arm]=Field().cuda();opts[arm]=torch.optim.Adam(heads[arm].parameters(),lr=.001,betas=(.9,.999),eps=1e-8,weight_decay=0)
        history={a:[] for a in TRAIN};best={a:(-float('inf'),None,None) for a in TRAIN}
        schedule=episodes();save(root/'private/episodes.json',schedule)
        def base(pixels):
            with torch.no_grad():
                x=normalize(pixels).cuda();z=host(x,host.zero());ref=reference(x,z)
            return z,ref
        training_started=time.time()
        for i,spec in enumerate(schedule):
            guard()
            for opt in opts.values(): opt.zero_grad(set_to_none=True)
            losses={a:0. for a in TRAIN}
            for index,noise in zip(spec['pair'],spec['noise_seeds']):
                z,ref=base(perturb(images['train'][index],spec['appearance'],noise))
                for arm in TRAIN:
                    loss=supervision(correction(z,ref,heads[arm],arm),labels['train'][index])/2
                    if not torch.isfinite(loss): raise ValueError('nonfinite source loss')
                    loss.backward();losses[arm]+=float(loss.detach())
            for arm in TRAIN:
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in heads[arm].parameters()):
                    raise ValueError('missing/nonfinite source gradient')
                norm=sum(float(p.grad.square().sum()) for p in heads[arm].parameters())**.5
                opts[arm].step();history[arm].append(dict(loss=losses[arm],gradient_norm=norm))
            status['training_episodes']+=4
            if i==1:
                projected=time.time()-started+198*(time.time()-training_started)/2*2+300
                if projected>c['gpu_cap_seconds']-20: raise TimeoutError('training not admitted by measured first episodes')
                save(root/'QUALIFICATION.json',dict(status='PASS',projected_worker_seconds=projected,head_parameters={a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()},backbone_frozen=True,first_two_episodes_in_budget=True,final_label_reads=0))
                status['status']='TRAINING'
            if (i+1)%50==0:
                validation={a:[] for a in TRAIN}
                for pair in range(4):
                    for k,s in enumerate(VALIDATION):
                        z,ref=base(perturb(images['validation'][pair+4],s,SEED+2000+pair))
                        with torch.no_grad():
                            for arm in TRAIN: validation[arm].append(float(dice(correction(z,ref,heads[arm],arm),labels['validation'][pair+4])))
                for arm in TRAIN:
                    value=float(np.mean(validation[arm]))
                    if value>best[arm][0]: best[arm]=(value,i+1,copy.deepcopy(heads[arm].state_dict()))
                    save(root/'private'/f'training_{arm}.json',dict(steps=history[arm],validation_episode=i+1,validation_dice=value,selected_episode=best[arm][1]))
                status['current_episode']=i+1;save(root/'RUN_STATE.json',status)
        for arm in TRAIN:
            heads[arm].load_state_dict(best[arm][2]);heads[arm].eval().requires_grad_(False)
            torch.save(dict(state_dict=best[arm][2],selected_episode=best[arm][1]),root/'private'/f'head_{arm}.pt')
        save(root/'TRAINING_SUMMARY.json',dict(arms={a:dict(episodes=200,supervised_label_visits=400,validation_readouts=64,selected_episode=best[a][1],validation_dice=best[a][0],first_loss=history[a][0]['loss'],last_loss=history[a][-1]['loss'],first_gradient_norm=history[a][0]['gradient_norm'],last_gradient_norm=history[a][-1]['gradient_norm']) for a in TRAIN},total_training_episodes=800,total_supervised_label_visits=1600,parameter_count_per_head=3649))
        frozen={}
        for arm,epoch in dict(X=100,P=100,F=100,D=150).items():
            payload=torch.load(prior/'private'/f'head_{arm}.pt',map_location='cpu',weights_only=True)
            if payload['selected_episode']!=epoch: raise ValueError('frozen head binding mismatch')
            head=ParameterPredictor(512) if arm=='F' else Field();head.load_state_dict(payload['state_dict']);head.cuda().eval().requires_grad_(False);frozen[arm]=head
        previous=read(amp/'private/predictions.json');lookup={(r['pair'],r['condition'],r['arm'],r['mode']):r['index'] for r in previous}
        parity=dict(B0='B0',D_OLD='D',F='F',X_MATCH='X_MATCH',P_MATCH='P_MATCH',C_FULL='C_FULL')
        native=Native(state,candidates()[3],SEED,'source_episode_reset',device='cuda:0');initial=native.snapshot()
        records=[];checks=[];budget_checks=[]
        def record(pair,k,arm,mode,z,z0,detail):
            pred=z.detach().cpu().numpy()[0]>=0
            if arm in parity:
                ix=lookup[pair,k,parity[arm],mode]
                raw=(amp/'predictions'/f'{ix:04d}.bits').read_bytes()
                if np.packbits(pred).tobytes()!=raw: raise ValueError('strong baseline parity failed: '+arm)
                status['baseline_parity']+=1
            dz=z-z0;dp=z.sigmoid()-z0.sigmoid()
            detail.update(logit_rms=float(dz.square().mean().sqrt()),probability_abs_mean=float(dp.abs().mean()),changed_pixels=int((pred!=(z0.detach().cpu().numpy()[0]>=0)).sum()),maximum_logit_change=float(dz.abs().max()),nesting_violation_pixels=int((pred[1]&~pred[0]).sum()))
            ix=len(records)
            with (root/'predictions'/f'{ix:04d}.bits').open('xb') as f: f.write(np.packbits(pred).tobytes())
            records.append(dict(index=ix,pair=pair,condition=k,arm=arm,mode=mode,diagnostics=detail))
            status['predictions']=len(records)
        status['status']='GENERATING';save(root/'RUN_STATE.json',status)
        for pair in range(8):
            for k,spec in enumerate(FINAL):
                guard();pixels=perturb(images['query'][pair],spec,SEED+pair);q=normalize(pixels).cuda()
                with torch.no_grad():
                    z0=host(q,host.zero());ref=reference(q,z0)
                    values=dict(B0=z0,D_OLD=direct(host,frozen['D'],q))
                    for arm in TRAIN: values[arm]=correction(z0,ref,heads[arm],arm)
                    values['R_NEG']=correction(z0,ref,heads['R'],'R_NEG')
                    values['R_SHUFFLE']=correction(z0,ref,heads['R'],'R_SHUFFLE')
                    values['R_BUDGET'],bd=match_logit_rms(values['R'],z0,values['D_RETRAIN']);budget_checks.append(bd)
                    for arm in DIRECT:
                        if arm.startswith('R') and (values[arm]-z0).abs().max()>1.000001: raise ValueError('bounded correction violated')
                        record(pair,k,arm,'DIRECT',values[arm],z0,dict(bd) if arm=='R_BUDGET' else {})
                for mode in MODES:
                    guard();condition=(k+1)%4 if mode=='CROSS_MISMATCH' else k
                    p=pixels if mode=='SELF' else perturb(images['support'][pair],FINAL[condition],SEED+pair+100)
                    x=normalize(p).cuda();af,df=adapt(host,frozen['F'],'F',x)
                    states=dict(F=(af,df))
                    for arm in ('X','P'):
                        raw,_=adapt(host,frozen[arm],arm,x);a,d=match(raw,af);checks.append(d);states[arm+'_MATCH']=(a,d)
                    for arm,(a,d) in states.items():
                        with torch.no_grad(): z=host(q,a)
                        record(pair,k,arm,mode,z,z0,dict(d))
                    native.restore(initial);_,d=native.step(p)
                    z=native._readonly(native.native.model,pixels);record(pair,k,'C_FULL',mode,z,z0,d['diagnostics'])
                save(root/'RUN_STATE.json',status)
        if len(records)!=672 or status['baseline_parity']!=448 or status['training_episodes']!=800: raise ValueError('incomplete fixed matrix')
        if any(p._version!=before[n] or p.grad is not None for n,p in model.named_parameters()): raise ValueError('backbone changed')
        hook.remove();save(root/'private/predictions.json',records)
        save(root/'MECHANISM_AUDIT.json',dict(status='PASS',baseline_parity=448,matched_parameter_states=len(checks),maximum_parameter_norm_error=max(d['norm_absolute_error'] for d in checks),function_budget_exact=sum(d['rms_match_exact'] for d in budget_checks),function_budget_total=32,function_budget_downscale_only=True,all_correction_bounds_pass=True,**counts))
        save(root/'ACCESS_AUDIT.json',dict(source_image_files=len([p for p in opened if '/image/' in p]),train_validation_label_files=len([p for p in opened if '/mask/' in p]),final_label_reads=0,prior_final_score_reads=0,target_data_reads=0,backbone_unchanged=True,patient_independence=False,previously_exposed_development_contents=True))
        used=sum(p.stat().st_size for d in ('private','predictions','logs') for p in (root/d).rglob('*') if p.is_file())
        if used>c['output_cap_bytes']: raise ValueError('output cap')
        status.update(status='GPU_COMPLETE',output_bytes=used,peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        status.update(status='NOT_EVALUABLE',error_type=type(error).__name__,reason=str(error));raise
    finally:
        if native is not None: native.close()
        status.update(ended=time.time(),gpu_worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);mode=os.environ['DIAG_MODE']
    if mode=='controller': sys.exit(controller(c))
    elif mode=='worker': worker(c)
    elif mode=='score':
        from .score import score
        score(c)
    else: raise ValueError('unknown mode')
