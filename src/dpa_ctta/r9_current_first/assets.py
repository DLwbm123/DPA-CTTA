"""Explicitly bound R8 read-only preparations and private source inputs for R9."""
import contextlib
import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import torch
from ..r8_ba.inputs import bind_metadata
from ..r8_ba.oracle_journal import load_oracles
from ..r8_ba.preparation import load_bases
from ..r8_ba.scaler_journal import load_scaler
from ..r8_ba.methods import build,CurrentMLP
from ..r7_source_prep.registry import Reader,verified
from ..integrations.ctta_suite import build_reference_model
from .deployment import Segmenter
from .protocol import parse_recipe,digest,SPEC_SHA,SPEC
from .storage import load_torch


class ContinuationUnavailable(Exception):pass


def gpu_policy(assignment):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(assignment['physical_id']) or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('R9 assigned GPU/deterministic environment')
    output=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    ids={int(i.strip()):u.strip() for i,u in (r.split(',') for r in output.splitlines())}
    if ids.get(assignment['physical_id'])!=assignment['uuid']:raise ValueError('R9 physical GPU UUID mismatch')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('one bound CUDA device required')


def candidates(binding):
    rows=binding['configs']
    a,b=rows['A'],rows['B']
    if (a.get('rank')!=32 or b.get('rank')!=64 or a.get('film_amplitude')!=.3 or b.get('film_amplitude')!=.3 or
        b.get('observer')!='global' or b.get('aux_multiplier')!=1.):raise ValueError('Screen24 fixed capacity/config')
    return rows


@contextlib.contextmanager
def open_source(binding,assignment,guard):
    gpu_policy(assignment);validate_metadata(binding);bound=bind_metadata(binding['refs']);meta=bound['docs']
    checkpoint=meta['manifest']['checkpoint'];segmenter=None
    if checkpoint['sha256']!=binding['checkpoint_sha256']:raise ValueError('checkpoint metadata binding')
    from collections import Counter
    reader=Reader(meta['manifest'],meta['split'],meta['target'],Path(binding['source_root']),Counter(),256*1024**2)
    try:
        guard();data=reader.data();guard()
        raw=verified(binding['checkpoint_path'],checkpoint['sha256'],256*1024**2)
        if len(raw)!=checkpoint['bytes']:raise ValueError('checkpoint size')
        model,_=build_reference_model('fundus');model.load_state_dict(torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True))
        segmenter=Segmenter(model,.3,device='cuda:0')
        old=binding['r8_assets']
        oracles=load_oracles(Path(old['oracle_root']),data,old['oracle_binding'],.3)
        bases=load_bases(Path(old['bases_root']),old['bases_identity'])
        scaler=load_scaler(Path(old['scaler_root']),data,old['scaler_binding'])
        if not torch.equal(bases['gradient_scale'][64],torch.tensor(binding['gradient_scale'],dtype=torch.float64)):raise ValueError('gradient scale differs from verified fit asset')
        yield data,segmenter,oracles,bases,scaler
    finally:
        if segmenter:segmenter.close()
        reader.after_check()


def make_method(recipe,seed,binding,bases,scaler,selected=None):
    route,kind,mode=parse_recipe(recipe,selected);config=candidates(binding)['B' if route=='MLP' else route]
    basis=bases['A_basis' if route=='A' else 'B_basis'][config['rank']]
    if route=='MLP':
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);method=CurrentMLP(basis,.3,config['observer'])
    else:method=build(config,basis,static=mode=='STATIC',seed=seed)
    method.observer.fit_scaler(scaler,'fit');return method,kind


def legacy_snapshot(binding,job):
    route,kind,mode=parse_recipe(job['recipe'])
    if kind!='LEGACY':raise ValueError('legacy snapshot requested by new recipe')
    name=f"CURRENT_MLP_None_{job['seed']}" if route=='MLP' else f"{route}_{mode}_{job['seed']}"
    ref=binding['legacy_snapshots'].get(name)
    if ref is None:raise ContinuationUnavailable('registered full snapshot unavailable')
    path=Path(ref['path'])
    if not path.exists():raise ContinuationUnavailable('full pre-cal source snapshot absent')
    marker=json.loads(verified(ref['marker']['path'],ref['marker']['sha256'],1024**2))
    fit=json.loads(verified(ref['fit_receipt']['path'],ref['fit_receipt']['sha256'],1024**2))
    identity=marker['binding_payload']
    if (marker.get('schema')!='R8_SOURCE_JOB_WORK_COMPLETE_V1' or marker.get('binding')!=ref['binding'] or
        hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()!=ref['binding'] or
        identity['code_sha']!=SPEC['scientific_artifact_sha_from_screen24'] or
        identity['config']!=binding['configs']['B' if route=='MLP' else route] or
        identity['oracle_receipt_sha256']!=binding['oracle_receipt_sha256'] or
        identity['bases_receipt_sha256']!=binding['bases_receipt_sha256'] or
        identity['scaler_receipt_sha256']!=binding['scaler_receipt_sha256'] or
        marker['fit_receipt_sha256']!=ref['fit_receipt']['sha256'] or fit['binding']!=ref['binding'] or
        fit['selected_sha256'].get('4000')!=ref['sha256']):raise ValueError('legacy source receipt chain')
    payload=load_torch(path,ref['sha256'])
    if payload.get('schema')!='R8_SOURCE_JOURNAL_V1':raise ContinuationUnavailable('not a full fit journal')
    snap=payload['snapshot']
    if snap.get('binding')!=ref['binding'] or snap.get('source_seed')!=job['seed'] or snap.get('steps')!=4000:
        raise ValueError('old source snapshot identity mismatch')
    if not snap.get('optimizer',{}).get('state') or 'rng' not in snap or 'episode_state' not in snap:
        raise ContinuationUnavailable('optimizer/RNG/episode snapshot unavailable')
    return snap,ref


def available_memory(assignment,peak_bytes):
    if type(peak_bytes) is not int or peak_bytes<=0:raise ValueError('measured peak VRAM required')
    text=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    rows={int(i.strip()):(u.strip(),int(m.strip())*1024**2) for i,u,m in (line.split(',') for line in text.splitlines())}
    uuid,free=rows[assignment['physical_id']]
    if uuid!=assignment['uuid']:raise ValueError('GPU UUID changed')
    if free<max(int(1.2*peak_bytes),peak_bytes+512*1024**2):raise RuntimeError('insufficient free GPU memory for measured peak plus margin')
    return free


def validate_metadata(binding):
    """Validate receipt chains before opening private images or weights."""
    from ..r8_ba.protocol import PROTOCOL_SHA256
    candidates(binding);old=binding['r8_assets']
    common=dict(code_sha=SPEC['scientific_artifact_sha_from_screen24'],protocol_sha256=PROTOCOL_SHA256,
                refs=binding['refs'],checkpoint_sha256=binding['checkpoint_sha256'])
    canonical=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
    oracle_identity=dict(**common,amplitude=.3)
    if old['oracle_binding']!=canonical(oracle_identity) or old['scaler_binding']!=canonical(dict(**common,fold='fit',zero_film=True)):
        raise ValueError('prepared data/code/checkpoint identity')
    if old['bases_identity']!=dict(**oracle_identity,oracle_receipt_sha256=binding['oracle_receipt_sha256']):raise ValueError('basis identity')
    for kind in ('oracle','bases','scaler'):
        path=Path(old[kind+'_root'])/(kind+'_complete.json')
        verified(path,binding[kind+'_receipt_sha256'],1024**2)
    ref=binding['screen24_index_ref'];index=json.loads(verified(ref['path'],ref['sha256'],16*1024**2))
    if index!=binding['screen24_index']:raise ValueError('Screen24 index binding')
    for route in ('A','B'):
        for seed in (20260924,20260925):
            if index['source_jobs'][f'{route}_FULL_{seed}']['config']!=binding['configs'][route]:raise ValueError('Screen24 actual config mismatch')
    return dict(spec_sha256=SPEC_SHA,bindings_sha256=digest(binding))
