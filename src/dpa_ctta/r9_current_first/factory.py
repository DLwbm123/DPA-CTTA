"""Build registered R9 hosts from verified prepared artifacts and the pinned backbone."""
import io
import json
from pathlib import Path
import random
import numpy as np
import torch
from ..b1_host import Host as GraTaHost,GRATA_COMMIT
from ..hosts.vptta import VPTTAHost
from ..source_pilot import SourceOnlyHost
from ..integrations.ctta_suite import build_reference_model
from ..r7_source_prep.registry import verified
from ..r7_shared.context import tensor_digest
from ..r8_ba.native_host import NativeHost
from ..r8_ba.protocol import PROTOCOL_SHA256 as R8_SHA
from ..r8_ba.context import capture
from ..r8_ba.methods import build,CurrentMLP
from ..r8_ba.target_factory import load_deployed
from .deployment import Segmenter,Host,prepare
from .gradient import GradientHost
from .protocol import SPEC_SHA
from .storage import load_torch


def construct(resolved,config,source_root,source_lock,lr_selection):
    b=config['bindings'];seed=resolved['seed'] or 20260907;random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    raw=verified(b['checkpoint_path'],b['checkpoint_sha256'],256*1024**2)
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);arm=resolved['arm']
    if arm in ('N_SOURCE_EVAL','VPTTA_NATIVE','C_CTTA_FIXED_LR','G_CTTA_RELEASE_TRANSFER'):
        if arm=='N_SOURCE_EVAL':native=SourceOnlyHost('fundus',state,device='cuda:0')
        elif arm=='VPTTA_NATIVE':native=VPTTAHost('fundus',source_state=state,device='cuda:0')
        else:native=GraTaHost(arm[0],state=state,device='cuda:0')
        identity=dict(code_sha=config['code_sha'],protocol_sha256=R8_SHA,checkpoint_sha256=b['checkpoint_sha256'],registration_sha256=b['refs']['target']['sha256'],seed=resolved['seed'])
        host=NativeHost(native,arm,identity)
        # Add R9 outer binding without changing the frozen native algorithm.
        host.context['payload']['r9_spec_sha256']=SPEC_SHA
        host.context['payload']['r9_source_lock_sha256']=source_lock['sha256']
        from ..r7_shared.context import json_digest
        host.context['sha256']=json_digest(host.context['payload']);return host,host.close
    model,_=build_reference_model('fundus');model.load_state_dict(state)
    s=Segmenter(model,.3,device='cuda:0',alpha=resolved['alpha']);method=None
    route='A' if arm.startswith('A_') else 'B';candidate=b['configs'][route]
    if resolved['source_job']:
        if resolved['source_origin']=='screen24':
            row=b['screen24_index']['source_jobs'][resolved['source_job']]
            method=load_deployed(row,candidate,resolved['seed'],'FULL')
        else:
            job=resolved['source_job'];artifact=resolved['artifact'];p=Path(source_root)/job/artifact['file']
            saved=load_torch(p,artifact['sha256']);mlp=arm.startswith('MLP')
            weights=saved['fit']['method'] if mlp else saved['method']
            static=job.endswith(f"STATIC_{resolved['seed']}")
            method=CurrentMLP(weights['basis'],.3,candidate['observer']) if mlp else build(candidate,weights['basis'],static,resolved['seed'])
            method.load_state_dict(weights,strict=True);method.freeze()
        method=prepare(method,resolved['diagnostic'])
    source=dict(checkpoint_sha256=b['checkpoint_sha256'],source_manifest_sha256=b['refs']['manifest']['sha256'],source_split_sha256=b['refs']['split']['sha256'],source_oracle_sha256=b['oracle_receipt_sha256'],basis_sha256=b['bases_receipt_sha256'],spec_sha256=SPEC_SHA,protocol_sha256=R8_SHA)
    c=dict(candidate,id=arm,r9_spec_sha256=SPEC_SHA,output_alpha=resolved['alpha'],diagnostic=resolved['diagnostic'],r9_source_lock_sha256=source_lock['sha256'],source_job=resolved['source_job'])
    if resolved['gradient']:
        gradient=resolved['gradient'];values=lr_selection['selected_lr'][gradient]
        lr=values.get(str(resolved['seed']),values.get('global'))
        scale=torch.tensor(b['gradient_scale'],dtype=torch.float64)
        c.update(gradient_arm=gradient,gradient_lr=lr,grata_commit=GRATA_COMMIT,scale_sha256=tensor_digest([('scale',scale)]))
        if gradient=='BN_RESET_G1':c['bn_affine_names']=sorted(n+'.'+pn for n,m in s.model.named_modules() if isinstance(m,torch.nn.BatchNorm2d) for pn,_ in m.named_parameters(recurse=False))
        host=GradientHost(s,method,c,source,capture(s,method,c,source),gradient,lr,scale)
    else:host=Host(s,method,c,source,capture(s,method,c,source),resolved['diagnostic'])
    return host,s.close
