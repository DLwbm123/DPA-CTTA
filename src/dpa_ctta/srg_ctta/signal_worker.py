"""Frozen E6-B scores, no labels, utility scorer, or dense feature export."""
import os,sys,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .signal_common import read,save,unpack
from .signal_math import score_frame,NAMES
from .signal_access import install

def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed
    root=Path(c['output_root']);started=time.time();panel=read(root/'private/panel.json')
    assert len(panel)==8 and all('mask_path' not in r for r in panel)
    assert c['scores']==list(NAMES) and c['head_feature_channels']==32
    torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
    torch.use_deterministic_algorithms(True,warn_only=False);torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    checkpoint=Path(c['checkpoint_path']).resolve()
    utility_file=Path(__file__).with_name('signal_score.py').resolve()
    opened=install(c,panel)
    for p in (root/'scorer/panel.json',utility_file,Path(c['denied_prior_roots'][0])/'public/RESULTS.json'):
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access isolation failure')
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    before={n:t.cpu().clone() for n,t in model.state_dict().items()};rows=[];all_counts=[];forwards=head_checks=0
    status=dict(status='GENERATING',unique_inputs=0,model_forwards=0);save(root/'RUN_STATE.json',status)
    with torch.no_grad():
        for repeat in (False,True):
            for i,item in enumerate(panel):
                pixels=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,appearance in enumerate(FINAL):
                    x=normalize(perturb(pixels,appearance,SEED+i)).cuda();out=model(x);z,f=out[0],out[2];forwards+=1
                    assert tuple(z.shape)==(1,2,512,512) and tuple(f.shape)==(1,32,512,512)
                    assert torch.equal(model.seg_head(f),z);head_checks+=1
                    s=unpack(root/'inputs'/f'{i}_{k}_SOURCE.bits');b=unpack(root/'inputs'/f'{i}_{k}_TOPO_SMOOTH.bits')
                    assert np.array_equal(z[0].cpu().numpy()>=0,s)
                    rr,counts=score_frame(f[0].cpu().double().numpy().transpose(1,2,0),z[0].cpu().double().numpy().transpose(1,2,0),s,b)
                    if forwards==1:
                        line=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True)
                        own=[v for v in line.splitlines() if v.split(',')[0].strip()==str(os.getpid())]
                        assert len(own)==1 and c['gpu']['uuid'] in own[0] and not any(t in own[0].lower() for t in ('wangbomin','ctta','signal'))
                        save(root/'GPU_VISIBLE.private.json',dict(status='PASS',line=own[0]))
                    if repeat:
                        assert rr==rows[i*4+k]['blocks'] and counts==all_counts[i*4+k]
                    else:
                        rows.append(dict(case=i,condition=k,blocks=rr));all_counts.append(counts)
                        status.update(unique_inputs=len(rows),model_forwards=forwards,common_blocks=sum(v['common'] for v in all_counts));save(root/'RUN_STATE.json',status)
        assert forwards==64 and head_checks==64 and len(rows)==32
        assert all(torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
    save(root/'private/scores.json',rows)
    save(root/'FULL_REPEATABILITY.json',dict(status='PASS',source_frames=32,model_forwards=64,all_scores_anchors_uids_and_coverage_exact=True,same_process=True,head_reconstruction_checks=64,source_bits_parity_checks=64))
    save(root/'MECHANISM_AUDIT.json',dict(status='PASS',source_parameters_buffers_immutable=True,head_input_channels=32,native_feature_grid=[512,512],model_forwards=64,updates=0,backwards=0,selector_applied=False,history_used=False))
    save(root/'ACCESS_AUDIT.json',dict(registered_rgb_files=len(opened),evaluation_label_reads=0,prior_scores_oracle_reads=0,utility_module_read_denied=True,dense_features_logits_probabilities_exported=False))
    status.update(status='SCORES_COMPLETE',model_forwards=64,source_frames=32,worker_seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated());save(root/'RUN_STATE.json',status)

if __name__=='__main__':worker(read(os.environ['DIAG_CONFIG']))
