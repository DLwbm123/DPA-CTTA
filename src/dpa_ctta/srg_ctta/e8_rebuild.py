"""Explicit migration reconstruction, not a claim of old endpoint bit parity."""
import os,time,subprocess
from pathlib import Path
import numpy as np
import torch
from .signal_common import read,save
from .signal_access import install
from .signal_math import components
from .translation_phase import outputs as phase
from .topology_projection import outputs as topology
from ..r78_cross_patient.run import FINAL,SEED,normalize,perturb,seed

def rebuild(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    root=Path(c['output_root']);panel=read(root/'private/panel.json');assert len(panel)==8
    torch.set_num_threads(2);torch.set_num_interop_threads(2);seed();torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    opened=install(c,panel)
    model,_=build_reference_model('fundus');model.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True));model.cuda().eval().requires_grad_(False)
    before={n:t.cpu().clone() for n,t in model.state_dict().items()};counts=[0,0,0];forwards=0
    with torch.no_grad():
        for repeat in (False,True):
            for i,item in enumerate(panel):
                pix=read_pixels(item['image_path'],'fundus',item['image_size'])
                for k,spec in enumerate(FINAL):
                    zz,_=phase(model,normalize(perturb(pix,spec,SEED+i)).cuda());forwards+=4
                    s=zz['SOURCE'][0].cpu().numpy()>=0;sm=zz['SOURCE_SMOOTH'][0].cpu().numpy()>=0;b=topology(s,sm)['TOPO_SMOOTH']
                    for arm,bit in [('SOURCE',s),('TOPO_SMOOTH',b)]:
                        p=root/'inputs'/f'{i}_{k}_{arm}.bits';packed=np.packbits(bit).tobytes()
                        if repeat:assert p.read_bytes()==packed
                        else:
                            with p.open('xb') as f:f.write(packed)
                    if not repeat:
                        _,sizes,od,oc=components(s,b);valid=sizes[(od==0)&(oc>0)];counts[0]+=int((valid==1).sum());counts[1]+=int(((valid>=2)&(valid<=4)).sum());counts[2]+=int((valid>=5).sum())
                    save(root/'REBUILD_STATE.json',dict(repeat=repeat,inputs=i*4+k+1,encoder_forwards=forwards))
                    if forwards==4:
                        lines=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid','--format=csv,noheader'],text=True).splitlines();own=[v for v in lines if v.split(',')[0].strip()==str(os.getpid())]
                        assert len(own)==1 and c['gpu']['uuid'] in own[0] and not any(t in own[0].lower() for t in ('wangbomin','ctta','signal'))
                        save(root/'REBUILD_GPU_VISIBLE.private.json',dict(status='PASS',line=own[0]))
    assert forwards==256 and all(torch.equal(t.cpu(),before[n]) for n,t in model.state_dict().items()) and all(p.grad is None for p in model.parameters())
    save(root/'REBUILD_AUDIT.json',dict(endpoint_arrays=64,repeat_arrays=64,exact_replay=True,encoder_forwards=forwards,oc_block_size_counts=counts,registered_rgb_files=len(opened),evaluation_label_reads=0,updates=0,original_sealed_bit_parity_verified=False))
    assert counts==[1588,1034,406], 'reconstructed action space differs from frozen aggregate'
    save(root/'REBUILD_COMPLETE.json',dict(status='PASS',scope='MIGRATION_RECONSTRUCTION_NOT_ORIGINAL_SEALED_BIT_PARITY',historical_block_count_gate=True))
