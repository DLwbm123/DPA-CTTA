"""Label-free, bounded diagnostic of repeated identical adaptation trajectories."""
import copy
import os
from pathlib import Path
import sys
import time
import torch
from .run import guard_assets,read,save,controller
from .method import Host,intensity_views
from ..r78_cross_patient.run import SEED,FINAL,seed,normalize,perturb


def compare_outputs(first,second):
    if len(first)!=len(second):raise ValueError('repeat trajectory coverage differs')
    changed_logits=changed_pixels=0;maximum=0.
    for a,b in zip(first,second):
        if a.shape!=b.shape:raise ValueError('repeat shape differs')
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('nonfinite repeat output')
        changed_logits+=int((a!=b).sum())
        changed_pixels+=int(((a>=0)!=(b>=0)).sum())
        maximum=max(maximum,float((a-b).abs().max()))
    return dict(readouts=len(first),changed_logit_elements=changed_logits,changed_prediction_pixels=changed_pixels,max_abs_logit_difference=maximum)


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r37_cw_orientation.method import Host as Native,candidates
    root=Path(c['output_root']);started=time.time();status=dict(status='QUALIFYING',arrivals=0,started=started)
    save(root/'RUN_STATE.json',status)
    try:
        assert not any(c[k] for k in ('rdc','engram','new_training'))
        torch.set_num_threads(2);torch.set_num_interop_threads(2)
        torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        panel=read(root/'private/panel.json');assert len(panel)==8
        opened=guard_assets(c,panel)
        for forbidden in (root/'scorer/panel.json',Path(c['state_prior_root'])/'private/predictions.json'):
            try:forbidden.read_bytes()
            except PermissionError:pass
            else:raise AssertionError('label/prior access was not denied')
        state=torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True)
        seed();base,_=build_reference_model('fundus');base.load_state_dict(state);base.cuda().eval().requires_grad_(False)
        images=[read_pixels(r['image_path'],'fundus',r['image_size']) for r in panel[:2]]
        schedule=((0,0),(1,1),(0,2),(1,0));rows=[]
        for mode in ('DEFAULT','STRICT'):
            strict=mode=='STRICT'
            torch.use_deterministic_algorithms(strict,warn_only=False);torch.backends.cudnn.deterministic=strict
            for arm in ('STRAT','RANDOM','CW1','CW2'):
                for policy in ('RESET','RETAIN'):
                    outputs=[];states=[]
                    for repeat in range(2):
                        seed()
                        native=arm.startswith('CW')
                        if native:
                            conf=copy.deepcopy(candidates()[3]);conf.setdefault('params',{})['online_steps_per_arrival']=int(arm[-1])
                            h=Native(state,conf,SEED,'finite_stream',device='cuda:0');initial=h.snapshot()
                        else:h=Host(base,arm)
                        trajectory=[]
                        try:
                            for case,k in schedule:
                                if time.time()-started>c['gpu_cap_seconds']-10:raise TimeoutError('repeatability GPU cap')
                                if policy=='RESET':
                                    if native:h.restore(initial)
                                    else:h.reset()
                                pixels=perturb(images[case],FINAL[k],SEED+case);x=normalize(pixels).cuda()
                                if native:
                                    with torch.no_grad():pre=h._readonly(h.native.model,pixels).detach().cpu()
                                    post,_=h.step(pixels);post=post.detach().cpu()
                                else:
                                    w,s=intensity_views(pixels,SEED+10000+case*10+k)
                                    with torch.no_grad():z0=base(x)[0]
                                    pre,post,_,_,_=h.step(x,normalize(w).cuda(),normalize(s).cuda(),z0,SEED+20000+case*10+k)
                                    pre=pre.detach().cpu();post=post.detach().cpu()
                                trajectory.extend((pre.clone(),post.clone()));status['arrivals']+=1
                            model=h.native.model if native else h.model
                            states.append({n:v.detach().cpu().clone() for n,v in model.state_dict().items()})
                            outputs.append(trajectory)
                        finally:
                            if native:h.close()
                            del h
                    row=dict(mode=mode,arm=arm,policy=policy,**compare_outputs(*outputs))
                    row['state_changed_elements']=sum(int((states[0][n]!=states[1][n]).sum()) for n in states[0])
                    row['exact']=row['changed_logit_elements']==0 and row['state_changed_elements']==0
                    rows.append(row);save(root/'REPEATABILITY.json',dict(status='PARTIAL',rows=rows))
                    status.update(status='GENERATING',mode=mode,arm=arm,policy=policy);save(root/'RUN_STATE.json',status)
                    if strict and not row['exact']:raise ValueError('strict deterministic repeatability failed')
        assert status['arrivals']==128 and len(rows)==16
        assert len(opened)==2
        save(root/'REPEATABILITY.json',dict(status='PASS',rows=rows,arrivals=128,paired_readouts=128,strict_exact=all(r['exact'] for r in rows if r['mode']=='STRICT'),source_image_files=len(opened),label_reads=0,previous_predictions_read=0,scientific_effect_estimate=False,scope='Two reused contents, four arrivals, within-process repeated trajectories only; does not establish cross-process or full-stream reproducibility'))
        status.update(status='GPU_COMPLETE',peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        status.update(status='NOT_EVALUABLE',error_type=type(error).__name__,reason=str(error));raise
    finally:
        status.update(ended=time.time(),gpu_worker_seconds=time.time()-started);save(root/'RUN_STATE.json',status)


def score(c):
    root=Path(c['output_root']);w=read(root/'processes/worker.json');r=read(root/'REPEATABILITY.json')
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and not w['active'] and w['exit_code']==0
    assert r['status']=='PASS' and r['arrivals']==128 and r['paired_readouts']==128 and r['strict_exact']
    save(root/'public/REPEATABILITY.json',r)
    save(root/'SCORE_COMPLETE.json',dict(status='COMPLETE',engineering_only=True,evaluation_label_files=0,arrivals=128,paired_readouts=128))


if __name__=='__main__':
    c=read(os.environ['DIAG_CONFIG']);mode=os.environ['DIAG_MODE']
    if mode=='controller':sys.exit(controller(c))
    elif mode=='worker':worker(c)
    elif mode=='score':score(c)
    else:raise ValueError('unknown mode')
