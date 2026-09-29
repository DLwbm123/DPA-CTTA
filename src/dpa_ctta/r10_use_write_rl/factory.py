"""R10 policies and unchanged registered native baselines on bound artifacts."""
import io,copy,random
from pathlib import Path
import numpy as np
import torch
from .protocol import SPEC_SHA,digest
from .controller import Actor,Controller,Host
from ..r9_current_first.deployment import Segmenter
from ..r9_current_first.storage import load_torch
from ..r7_source_prep.registry import verified
from ..integrations.ctta_suite import build_reference_model
from ..r8_ba.target_factory import load_deployed
from ..r8_ba.host import OnlineHost
from ..r8_ba.context import capture
from ..r8_ba.native_host import NativeHost
from ..r8_ba.protocol import PROTOCOL_SHA256
from ..r7_shared.context import json_digest


def resolve(slot,selection,receipts):
    arm=slot['arm'];seed=slot['seed'];method=arm;job=None;artifact=None
    if arm.startswith('SELECTED_SUP'):method=selection['selected']['SUP']
    if arm.startswith('SELECTED_GR'):method=selection['selected']['GR']
    if method is None:raise ValueError('family unavailable')
    if method in ('SUP_STATIC','SUP_SEQ','SUP_RET','GR_CUR','GR_SEQ','GR_RET','GR_RET_EMA'):
        family='SUP' if method.startswith('SUP') else 'GR'
        job=f'FIT_{method}_{seed}' if seed in (20260924,20260925) else f'FIT_SELECTED_{family}_{seed}'
        receipt=receipts[job]
        if receipt['method']!=method:raise ValueError('selected recipe artifact mismatch')
        artifact=receipt['points']['4000']['artifact'] if slot.get('checkpoint')=='round_4000' else receipt['selection']['artifact']
    elif arm=='WARM_STATIC':job=f'WARM_{seed}';artifact=receipts[job]['artifact']
    return dict(slot,method=method,source_job=job,artifact=artifact)


def construct(resolved,config,root,lock):
    b=config['bindings'];arm={'C0':'C0_CURRENT_STATS','C_CTTA':'C_CTTA_FIXED_LR','G_CTTA':'G_CTTA_RELEASE_TRANSFER'}.get(resolved['arm'],resolved['arm']);seed=resolved['seed'] or 20260907
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    weights=torch.load(io.BytesIO(verified(b['checkpoint_path'],b['checkpoint_sha256'],256*1024**2)),map_location='cpu',weights_only=True)
    identity=dict(code_sha=config['code_sha'],spec_sha256=SPEC_SHA,assets=digest(b),source_lock=lock['sha256'],resolved=resolved)
    nativearms=('N_SOURCE_EVAL','VPTTA_NATIVE','C_CTTA_FIXED_LR','G_CTTA_RELEASE_TRANSFER')
    if arm in nativearms:
        if arm=='N_SOURCE_EVAL':
            from ..source_pilot import SourceOnlyHost
            native=SourceOnlyHost('fundus',weights,device='cuda:0')
        elif arm=='VPTTA_NATIVE':
            from ..hosts.vptta import VPTTAHost
            native=VPTTAHost('fundus',source_state=weights,device='cuda:0')
        else:
            from ..b1_host import Host as Native
            native=Native(arm[0],state=weights,device='cuda:0')
        h=NativeHost(native,arm,dict(code_sha=config['code_sha'],protocol_sha256=PROTOCOL_SHA256,checkpoint_sha256=b['checkpoint_sha256'],registration_sha256=b['refs']['target']['sha256'],seed=resolved['seed']))
        h.context['payload']['r10']=identity;h.context['sha256']=json_digest(h.context['payload']);return h,h.close
    model,_=build_reference_model('fundus');model.load_state_dict(weights);seg=Segmenter(model,.3,device='cuda:0')
    if resolved['source_job']:
        p=resolved['artifact'];payload=load_torch(Path(root)/'source'/resolved['source_job']/p['file'],p['sha256'])
        actor=Actor(seed);actor.load_state_dict(payload['actor']);actor.requires_grad_(False)
        parent=load_deployed(b['screen24_index']['source_jobs']['B_FULL_20260924'],b['configs']['B'],20260924,'FULL')
        diagnostic=resolved.get('diagnostic')
        if arm=='WARM_STATIC' or resolved['method']=='SUP_STATIC':diagnostic='WARM_STATIC' if diagnostic is None else diagnostic
        host=Host(seg,Controller(parent,b['gradient_scale']),actor,identity,diagnostic)
    else:
        parent=None;ablation=None
        if arm!='C0_CURRENT_STATS':
            mode='STATIC' if 'STATIC' in arm else 'FULL'
            parent=load_deployed(b['screen24_index']['source_jobs'][f'B_{mode}_20260924'],b['configs']['B'],20260924,mode)
            if 'RESET' in arm:ablation='RESET_HISTORY'
        c=dict(b['configs']['B'],r10=identity)
        source=dict(checkpoint_sha256=b['checkpoint_sha256'],source_manifest_sha256=b['refs']['manifest']['sha256'],source_split_sha256=b['refs']['split']['sha256'],source_oracle_sha256=b['oracle_receipt_sha256'],basis_sha256=b['bases_receipt_sha256'],spec_sha256=SPEC_SHA,protocol_sha256=PROTOCOL_SHA256)
        host=OnlineHost(seg,parent,c,source,capture(seg,parent,c,source),ablation)
    return host,seg.close
