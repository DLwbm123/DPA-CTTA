"""Versioned, content-based CPU inference identity; no real-asset reads.

Expected contexts are supplied by a trusted preparation artifact, never minted
from the candidate segmenter by the fresh loader. Digests are not signatures.
"""
import copy
import hashlib
import json
import re
import sys
import time
from collections import Counter
from functools import lru_cache
from pathlib import Path
import torch
from torch import nn

HASH_COST=Counter()
SCHEMA='R7_INFERENCE_CONTEXT_V1'
STATE_SCHEMA='R7_ONLINE_STATE_V2'
POLICY=dict(schema='R7_SEGMENTER_POLICY_V1',preprocessing='pinned_fundus_minmax_float32_batch1_512_v1',BN='current_batch_spatial_no_running_stats_frozen_affine_v1',FiLM='up1_up3_256_each_gamma_beta_expm1_0.1_tanh_v1',observer='RGB_conv1_population134_up3_std_clamp1e-6_pool8x8_QR256x64_LN1e-6_v1',ctta_source='dbff0d985c6c95345d9fb78f5b1daef57b392564')
SCIENCE={
 'COMMON_PROTOCOL.json':'a18f51317b8547067f8383ebd1125eca1697b24ce6245ca5e2733e2daa28f42d',
 'A_PSF.json':'5e8f62ab8a39806ef5aef2dbb5b0206255d9e49c33fa3bcd68c82e78e88410bf',
 'B_RCA.json':'9895dba31443881e50bb87d4df94f83598c0069931b35d6d000e6c2b1e61fc61',
 'C_RBE.json':'4151c86ebec9073bca570e696c45bbecc4f10e0505698989a0be41cbe2032e86'}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf8')
def json_digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def tensor_digest(named):
    """Sorted names; length-framed JSON(name,dtype,shape,nbytes) and C-order LE bytes.

    No pickle/device/address participates. Detach and CPU staging preserve dtype
    and values. Supported types are the real/bool types used by this backend.
    """
    started=time.perf_counter_ns();cpu=time.process_time_ns();size=0
    h=hashlib.sha256(b'R7_NAMED_TENSORS_LE_V1\0');seen=set()
    try:
        for name,t in sorted(named,key=lambda x:x[0]):
            if not isinstance(name,str) or name in seen:raise ValueError('unique named tensors required')
            seen.add(name)
            if t.layout!=torch.strided or t.is_complex() or t.is_quantized:raise ValueError('unsupported tensor representation')
            v=t.detach().cpu().contiguous();raw=v.reshape(-1).view(torch.uint8).numpy().tobytes(order='C')
            if sys.byteorder!='little' and v.element_size()>1:
                n=v.element_size();raw=b''.join(raw[i:i+n][::-1] for i in range(0,len(raw),n))
            header=canonical(dict(name=name,dtype=str(v.dtype),shape=list(v.shape),nbytes=len(raw)))
            h.update(len(header).to_bytes(8,'little'));h.update(header)
            h.update(len(raw).to_bytes(8,'little'));h.update(raw);size+=len(raw)
        return h.hexdigest()
    finally:
        HASH_COST.update(tensor_digest_calls=1,tensor_bytes=size,wall_ns=time.perf_counter_ns()-started,cpu_ns=time.process_time_ns()-cpu)

def tensors(module):
    # Include nonpersistent buffers too: effective forward state, not just a
    # checkpoint's serializable subset. None buffers are covered in the layout.
    return list(module.named_parameters())+list(module.named_buffers())

def layout(module):
    rows=[]
    for name,m in module.named_modules():
        row=dict(name=name,kind=type(m).__module__+'.'+type(m).__qualname__,training=m.training,configuration=m.extra_repr())
        if isinstance(m,nn.BatchNorm2d):
            row['BN']=dict(eps=m.eps,momentum=m.momentum,affine=m.affine,track_running_stats=m.track_running_stats,num_features=m.num_features,none_buffers=sorted(k for k,v in m._buffers.items() if v is None))
        rows.append(row)
    return rows

@lru_cache(maxsize=1)
def design_identity():
    root=Path(__file__).resolve().parents[3]/'docs/review/r7/input'
    actual={name:hashlib.sha256((root/'specs'/name).read_bytes()).hexdigest() for name in SCIENCE}
    if actual!=SCIENCE:raise ValueError('original R7 science changed')
    return dict(design_manifest_sha256=hashlib.sha256((root/'MANIFEST.json').read_bytes()).hexdigest(),science_sha256=actual)

def provenance():
    return dict(schema='R7_SOURCE_PROVENANCE_V1',**copy.deepcopy(design_identity()),source_binding_status='PENDING',checkpoint_file_sha256=None,source_manifest_sha256=None,source_split_sha256=None,training_asset_file_sha256=None)

def validate_provenance(value):
    template=provenance()
    if not isinstance(value,dict) or set(value)!=set(template):raise ValueError('versioned source provenance required')
    if value['schema']!=template['schema'] or any(value[k]!=template[k] for k in ('design_manifest_sha256','science_sha256')):raise ValueError('source training specification mismatch')
    if value['source_binding_status'] not in ('PENDING','PROCEDURAL_CPU','BOUND'):raise ValueError('source status')
    keys=('checkpoint_file_sha256','source_manifest_sha256','source_split_sha256','training_asset_file_sha256')
    for key in keys:
        if value[key] is not None and not re.fullmatch('[0-9a-f]{64}',str(value[key])):raise ValueError('source file SHA256')
    if value['source_binding_status']=='BOUND' and any(value[k] is None for k in keys[:3]):raise ValueError('incomplete claimed source binding')

def environment(segmenter):
    if any(p.requires_grad for p in segmenter.model.parameters()) or segmenter.projection.requires_grad:raise ValueError('frozen segmenter/projection required')
    if segmenter.inference_policy!=POLICY:raise ValueError('unsupported preprocessing/BN/FiLM policy')
    for m in segmenter.model.modules():
        if m.training:raise ValueError('frozen eval backbone required')
        if isinstance(m,nn.BatchNorm2d) and (m.track_running_stats or m.running_mean is not None or m.running_var is not None):raise ValueError('current-statistics BN required')
    return dict(effective_state_sha256=tensor_digest(tensors(segmenter.model)),observer_projection_sha256=tensor_digest([('projection',segmenter.projection)]),model_layout=layout(segmenter.model),policy=copy.deepcopy(segmenter.inference_policy))

def make_context(segmenter,method=None,ablation=None,source=None,*,expected_environment=None):
    started=time.perf_counter_ns();cpu=time.process_time_ns()
    try:
        source=copy.deepcopy(provenance() if source is None else source);validate_provenance(source)
        env=environment(segmenter)
        if expected_environment is not None and env!=expected_environment:raise ValueError('source segmenter changed during preparation')
        payload=dict(environment=env,method=dict(schema='R7_METHOD_DIGEST_V1',group='C0' if method is None else method.group,mode='ZERO' if method is None else 'STATIC' if method.static else 'FULL',weights_sha256=None if method is None else method.digest()),ablation=ablation,source=source)
        return dict(schema=SCHEMA,payload=payload,sha256=json_digest(payload))
    finally:
        HASH_COST.update(context_build_calls=1,context_wall_ns=time.perf_counter_ns()-started,context_cpu_ns=time.process_time_ns()-cpu)

def validate_context(context):
    if not isinstance(context,dict) or set(context)!= {'schema','payload','sha256'} or context['schema']!=SCHEMA:raise ValueError('legacy/incomplete inference context unsupported')
    if context['sha256']!=json_digest(context['payload']):raise ValueError('context digest mismatch')
    payload=context['payload']
    if set(payload)!= {'environment','method','ablation','source'}:raise ValueError('context payload schema')
    validate_provenance(payload['source'])

def require_context(actual,expected):
    validate_context(expected)
    if actual!=expected:raise ValueError('incompatible inference context')

def stamp(segmenter,method,ablation):
    """In-process version/replacement guard; never used as cross-instance identity.

    Ordinary in-place mutation/replacement, trainability and policy changes fail
    before a forward. Deliberate .data/storage bypass is unsupported; boundaries
    rehash content. No tensor bytes or full-model SHA is scanned per visit.
    """
    modules=[('segmenter',segmenter.model)] + ([] if method is None else [('method',method)])
    entries=[('projection',segmenter.projection)]
    for prefix,m in modules:entries.extend((prefix+'.'+n,t) for n,t in tensors(m))
    identity=tuple((n,id(t),t._version,t.requires_grad,str(t.dtype),tuple(t.shape),str(t.device)) for n,t in entries)
    policy=canonical(dict(policy=segmenter.inference_policy,layouts=[layout(m) for _,m in modules],group=None if method is None else method.group,static=None if method is None else method.static,stage=None if method is None else method.stage,ablation=ablation))
    return identity,policy


def deployment_context(prepared,ablation):
    """Explicit derivation on trusted preparation metadata; no segmenter input."""
    validate_context(prepared)
    allowed={'A':'A_ISO_OBS','B':'B_PRED_ONLY','C':'C_CONST_R'}
    m=prepared['payload']['method']
    if m['mode']!='FULL' or allowed.get(m['group'])!=ablation or prepared['payload']['ablation'] is not None:raise ValueError('FULL deployment ablation context')
    result=copy.deepcopy(prepared);result['payload']['ablation']=ablation;result['sha256']=json_digest(result['payload']);return result
