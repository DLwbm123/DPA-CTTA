"""Frozen existing-model local re-encoding; no labels or installed edits."""
import os,time,subprocess
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .signal_common import read,save,unpack
from .signal_math import score_frame as legacy
from .e8_math import window,partitions,stratify,NAMES
from .signal_access import install

def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
    root=Path(c['output_root']);start=time.time();panel=read(root/'private/panel.json')
    assert len(panel)==8 and all('mask_path' not in r for r in panel) and c['scores']==list(NAMES)
    torch.set_num_threads(2);torch.set_num_interop_threads(2);seed();torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    opened=install(c,panel)
    for p in (root/'scorer/panel.json',Path(__file__).with_name('e8_score.py'),Path(c['denied_prior_roots'][0])/'public/RESULTS.json'):
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access isolation')
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    before={n:t.cpu().clone() for n,t in model.state_dict().items()};rows=[];counts=[];nf=nc=nh=0;first_crops=0
    save(root/'RUN_STATE.json',dict(status='GENERATING',unique_inputs=0,model_forwards=0))
    down=lambda a:F.interpolate(a,size=(256,256),mode='bilinear',align_corners=False)
    arr=lambda a:a[0].cpu().double().numpy().transpose(1,2,0)
    with torch.no_grad():
        for repeat in (False,True):
            for i,item in enumerate(panel):
                pix=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,appearance in enumerate(FINAL):
                    x=normalize(perturb(pix,appearance,SEED+i)).cuda();out=model(x);z,f=out[0],out[2];nf+=1
                    assert tuple(z.shape)==(1,2,512,512) and tuple(f.shape)==(1,32,512,512) and torch.equal(model.seg_head(f),z);nh+=1
                    s=unpack(root/'inputs'/f'{i}_{k}_SOURCE.bits');b=unpack(root/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits');assert np.array_equal(z[0].cpu().numpy()>=0,s)
                    rr,_=legacy(arr(f),arr(z),s,b)
                    if nf==1:
                        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True).splitlines()
                        own=[v for v in lines if v.split(',')[0].strip()==str(os.getpid())];assert len(own)==1 and c['gpu']['uuid'] in own[0] and not any(t in own[0].lower() for t in ('wangbomin','ctta','signal'))
                        save(root/'GPU_VISIBLE.private.json',dict(status='PASS',line=own[0]))
                    for r in rr:
                        old=r['scores'];r['scores']={n:None for n in NAMES}
                        if not r['common']:continue
                        (y0,x0),fits=window(r['query']);r['crop_start']=[y0,x0]
                        if not fits:r.update(common=False,reason='CROP_OUTSIDE');continue
                        xx=F.interpolate(x[:,:,y0:y0+256,x0:x0+256],size=(512,512),mode='bilinear',align_corners=False)
                        lo=model(xx);lf,lz=lo[2],lo[0];nf+=1;nc+=1
                        assert torch.equal(model.seg_head(lf),lz);nh+=1
                        global_f=f[:,:,y0:y0+256,x0:x0+256];global_z=z[:,:,y0:y0+256,x0:x0+256]
                        crop=arr(down(lf));resample=arr(down(F.interpolate(global_f,size=(512,512),mode='bilinear',align_corners=False)))
                        reps=dict(CROP_Q=crop,GLOBAL_Q=arr(global_f),RESAMPLE_Q=resample,LOGIT_Q=arr(global_z),CROP_LOGIT_Q=arr(down(lz)),CROP_Q_SHIFT=np.roll(crop,129,axis=1))
                        bb=b[1,y0:y0+256,x0:x0+256];ss=s[1,y0:y0+256,x0:x0+256];edit=np.zeros((256,256),bool);q=np.asarray(r['query'])-[y0,x0];edit[q[:,0],q[:,1]]=True
                        scores,reason,nvalid=partitions(reps,bb,ss,edit);r['valid_direction_pixels']=nvalid
                        if scores is None:r.update(common=False,reason=reason);continue
                        r['scores'].update(scores,E6_FEATURE=old['FEATURE'],RAW_MARGIN=old['MARGIN_DIRECT'],GEOMETRY_ONLY=old['GEOMETRY_ONLY'],AREA_DELTA=float((ss[edit].astype(int)-bb[edit].astype(int)).sum()/65536));r['reason']='COMMON'
                    ct=stratify(rr)
                    if repeat:assert rr==rows[i*4+k]['blocks'] and ct==counts[i*4+k]
                    else:
                        rows.append(dict(case=i,condition=k,blocks=rr));counts.append(ct);first_crops=nc
                        save(root/'RUN_STATE.json',dict(status='GENERATING',unique_inputs=len(rows),model_forwards=nf,crop_forwards=nc,common_blocks=sum(v['common'] for v in counts)))
        assert len(rows)==32 and nc==2*first_crops and nf==64+nc and nh==nf
        assert all(torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
    save(root/'private/scores.json',rows)
    save(root/'FULL_REPEATABILITY.json',dict(status='PASS',source_frames=32,model_forwards=nf,crop_forwards=nc,all_scores_anchors_uids_and_coverage_exact=True,same_process=True,head_reconstruction_checks=nh,source_bits_parity_checks=64))
    save(root/'MECHANISM_AUDIT.json',dict(status='PASS',source_parameters_buffers_immutable=True,model_forwards=nf,first_pass_crop_forwards=first_crops,updates=0,backwards=0,selector_applied=False,history_used=False,crop_pixels=256))
    save(root/'ACCESS_AUDIT.json',dict(registered_rgb_files=len(opened),evaluation_label_reads=0,prior_scores_oracle_reads=0,utility_module_read_denied=True,dense_features_logits_probabilities_exported=False))
    save(root/'RUN_STATE.json',dict(status='SCORES_COMPLETE',source_frames=32,model_forwards=nf,worker_seconds=time.time()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))

if __name__=='__main__':worker(read(os.environ['DIAG_CONFIG']))
