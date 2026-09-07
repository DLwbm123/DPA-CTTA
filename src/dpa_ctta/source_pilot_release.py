"""Single authorized source-pilot release; private receipts, no scheduler or resume."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .integrations.ctta_suite import REFERENCE_COMMIT, checkout_root
from .hosts.vptta import VPTTAHost, model_input_from_pixels
from .source_io import SOURCES, registered_source, source_proxy, read_pixels, read_mask
from .source_pilot import (ARMS, DEFAULT_CONFIG, ArmFailure, _run_arm, assemble,
                           evaluate_after_step, expected_visits, summarize, validate_arm, validate_config)

CONFIG_SHA = 'e1aeb2454899d024e00934e1f012b5a0fe8cb3bb656f5dd20f73468fdd7a3faf'
AUTHORIZATION = 'direct_user_source_pilot_20260907_gpu4_7_coexist'
ROOT = Path(__file__).resolve().parents[2]
TOLERANCE = dict(rtol=1e-4, atol=1e-5)  # Fixed before GPU smoke; never adjusted using scores.


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): result.update(block)
    return result.hexdigest()


def write_json(path, value):
    data = json.dumps(value, indent=2, allow_nan=False) + '\n'
    if len(data.encode()) > 16*1024**2: raise ValueError('private JSON budget exceeded')
    with Path(path).open('x') as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def validate_receipt(receipt, stage):
    # Check explicit task authority before science config, registrations or any real inputs.
    if receipt.get('authorization') != AUTHORIZATION or receipt.get('stage') != stage:
        raise ValueError('UNAUTHORIZED_SOURCE_PILOT')
    if (receipt.get('config_sha256') != CONFIG_SHA or receipt.get('reference_commit') != REFERENCE_COMMIT
            or receipt.get('tasks') != ['fundus','polyp'] or receipt.get('physical_gpu') not in range(4,8)
            or not str(receipt.get('gpu_uuid','')).startswith('GPU-')):
        raise ValueError('EXECUTION_RECEIPT_MISMATCH')
    actual = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if receipt.get('commit') != actual or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():
        raise ValueError('EXECUTION_COMMIT_NOT_CLEAN_OR_MATCHED')
    if digest(DEFAULT_CONFIG) != CONFIG_SHA: raise ValueError('SCIENCE_CONFIG_MISMATCH')
    validate_config(json.loads(DEFAULT_CONFIG.read_text()))
    checkout_root()
    if digest(receipt['source_spec_path']) != receipt.get('source_spec_sha256'):
        raise ValueError('SOURCE_RECEIPT_SUMMARY_MISMATCH')
    out = Path(receipt['output_dir']).resolve()
    if out.is_relative_to(ROOT): raise ValueError('private outputs must be outside source checkout')
    if stage != 'register' and digest(out/'registration.json') != receipt.get('registration_sha256'):
        raise ValueError('REGISTERED_INPUT_SUMMARY_MISMATCH')
    if stage in ('smoke','pilot') and os.environ.get('CUDA_VISIBLE_DEVICES') != receipt['gpu_uuid']:
        raise ValueError('GPU_UUID_ENV_MISMATCH')
    inventory = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
    devices = {int(line.split(',')[0]):line.split(',')[1].strip() for line in inventory.splitlines()}
    if devices.get(receipt['physical_gpu']) != receipt['gpu_uuid']: raise ValueError('PHYSICAL_GPU_UUID_MISMATCH')
    return out


def file_identity(path, expected):
    path = Path(path)
    before = path.stat()
    if digest(path) != expected: raise ValueError('BLOCKED_INPUT_IDENTITY: digest mismatch')
    after = path.stat()
    if (before.st_size,before.st_mtime_ns) != (after.st_size,after.st_mtime_ns):
        raise ValueError('BLOCKED_INPUT_IDENTITY: file changed during registration')
    return dict(path=str(path),sha256=expected,bytes=after.st_size,mtime_ns=after.st_mtime_ns)


def register_inputs(receipt, out):
    spec = json.loads(Path(receipt['source_spec_path']).read_text())
    if set(spec['tasks']) != set(SOURCES): raise ValueError('BLOCKED_INPUT_IDENTITY: task registration mismatch')
    config = json.loads(DEFAULT_CONFIG.read_text())
    registration = dict(tasks={},identities=[],checkpoint_training_membership='UNKNOWN')
    for task in receipt['tasks']:
        entry = spec['tasks'][task]
        if entry['source'] != SOURCES[task]: raise ValueError('BLOCKED_INPUT_IDENTITY: wrong source')
        selected = registered_source(entry['manifest'],entry['split'],config['tasks'][task]['csv_specs'],entry['data_root'],task,config['seed'])
        # Identity stage hashes selected raw bytes only; it does not decode query labels.
        for row in selected['proxy']+selected['query']:
            for field in ('image','mask'):
                registration['identities'].append(file_identity(row[field+'_path'],row[field+'_sha256']))
        checkpoint = file_identity(entry['checkpoint'],entry['sealed_checkpoint_sha256'])
        registration['identities'].append(checkpoint)
        registration['tasks'][task] = dict(selected=selected,checkpoint=checkpoint,
            expected_visits=expected_visits(selected['query']),source=SOURCES[task],
            checkpoint_record=entry['checkpoint_record'],checkpoint_training_membership='UNKNOWN')
    write_json(out/'registration.json',registration)
    return registration


def check_registered_files(registration):
    # Registration already compared exact selected bytes. Cheap drift check before stages.
    for identity in registration['identities']:
        stat = Path(identity['path']).stat()
        if (stat.st_size,stat.st_mtime_ns) != (identity['bytes'],identity['mtime_ns']):
            raise ValueError('BLOCKED_INPUT_IDENTITY: registered file changed')


def environment():
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return dict(python=sys.version.split()[0],torch=torch.__version__,cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(),device_name=torch.cuda.get_device_name(0),
                tf32_matmul=False,tf32_cudnn=False,cudnn_benchmark=False,cudnn_deterministic=True,
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),seed=20260907,
                gpu_same_device_tolerance=TOLERANCE,seed_is_not_bitwise_determinism_proof=True)


def source_unchanged(host, state):
    for model in (host.model, getattr(host,'_proxy_model',host.model)):
        own = model.state_dict()
        if own.keys() != state.keys() or any(not torch.equal(v,state[k].to(v.device)) for k,v in own.items()):
            raise ValueError('source weights/buffers changed')


def execute_task(task, registered, state, proxy, out, *, factory=assemble, pixel_reader=None, evaluator=None):
    """Persist complete arms and failure prefixes. Fixture collaborators are test-only."""
    results = {}
    for arm in ARMS:
        host = None
        try:
            host = factory(task,arm,state,proxy,device='cuda:0',seed=20260907)
            with (out/f'{task}_{arm}.jsonl').open('x') as handle:
                def sink(record):
                    line = json.dumps(record,allow_nan=False)+'\n'
                    if handle.tell()+len(line.encode()) > 16*1024**2: raise ValueError('arm JSONL budget exceeded')
                    handle.write(line);handle.flush();os.fsync(handle.fileno())
                records = _run_arm(host,registered['selected']['query'],
                    pixel_reader or (lambda r:read_pixels(r['image_path'],task,r['image_size'])),
                    evaluator or (lambda pred,r:evaluate_after_step(pred,read_mask(r['mask_path'],task,r['image_size']),task)),
                    record_sink=sink,task=task,arm=arm)
            validate_arm(records,registered['expected_visits'],task,arm)
            source_unchanged(host,state)
            write_json(out/f'{task}_{arm}.completion.json',dict(status='COMPLETE',task=task,arm=arm,visits=len(records),source_unchanged=True))
            results[arm] = records
        except Exception as error:
            details = error.details if isinstance(error,ArmFailure) else dict(task=task,arm=arm,visit=None,segment=None,stage='assembly_or_completion',exception_type=type(error).__name__,private_reason=str(error),completed_records=0)
            if not isinstance(error,ArmFailure):
                p=out/f'{task}_{arm}.jsonl'
                details['completed_records']=sum(1 for _ in p.open()) if p.exists() else 0
            write_json(out/'failure.json',dict(status='INCOMPLETE',**details))
            raise
        finally:
            del host
            gc.collect()  # Release graphs; no empty_cache peak manipulation.
    return results


def smoke(registration, out):
    # Reuse the already reviewed procedural full-grid fixture and native snapshot helper.
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels, native_snapshot
    evidence = {}
    def close(a,b):
        if isinstance(a,torch.Tensor): torch.testing.assert_close(a,b,**TOLERANCE)
        elif isinstance(a,np.ndarray): np.testing.assert_allclose(a,b,**TOLERANCE)
        elif isinstance(a,dict):
            if a.keys()!=b.keys(): raise ValueError('native snapshot keys differ')
            for k in a: close(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            if len(a)!=len(b): raise ValueError('native snapshot length differs')
            for x,y in zip(a,b): close(x,y)
        elif a!=b: raise ValueError('native scalar snapshot differs')
    for task, registered in registration['tasks'].items():
        state=torch.load(registered['checkpoint']['path'],weights_only=True,map_location='cpu')
        proxy=source_proxy(registered['selected']['proxy'],task)
        direct=assemble(task,'A',state,device='cuda:0'); wrapped=assemble(task,'A',state,device='cuda:0')
        torch.cuda.reset_peak_memory_stats(0); start=time.perf_counter(); differences=[];retrieval=[]
        original_get=wrapped.memory_bank.get_neighbours
        def get(*args,**kwargs):
            retrieval.append(index+1)
            return original_get(*args,**kwargs)
        wrapped.memory_bank.get_neighbours=get
        for index in range(17):
            x=pixels(task,index)
            cpu_rng=torch.get_rng_state(); gpu_rng=torch.cuda.get_rng_state()
            a=direct.native_step(direct,model_input_from_pixels(x,task).to('cuda:0'))
            cpu_after=torch.get_rng_state();gpu_after=torch.cuda.get_rng_state()
            torch.set_rng_state(cpu_rng);torch.cuda.set_rng_state(gpu_rng)
            b=wrapped.step(x)
            if not torch.equal(cpu_after,torch.get_rng_state()) or not torch.equal(gpu_after,torch.cuda.get_rng_state()): raise ValueError('native RNG mismatch')
            close(a,b); differences.append(float((a-b).abs().max()))
            sa,sb=native_snapshot(direct),native_snapshot(wrapped);sb['memory'].pop('get_neighbours',None);close(sa,sb)
        if retrieval != [17]: raise ValueError('first retrieval not exercised')
        historical=next(iter(wrapped.memory_bank.memory.values()));saved=historical.copy()
        with torch.no_grad(): wrapped.prompt.data_prompt.add_(.001)
        if not np.array_equal(saved,historical): raise ValueError('GPU memory is not a historical CPU snapshot')
        source_unchanged(direct,state);source_unchanged(wrapped,state)
        evidence[task]=dict(native_steps=34,first_retrieval=17,max_logit_difference=max(differences),tolerance=TOLERANCE,
                            gpu_memory_history_snapshot=True,native_elapsed_seconds=time.perf_counter()-start,
                            native_peak_bytes=torch.cuda.max_memory_allocated(0),arms={})
        del direct,wrapped,sa,sb,a,b;gc.collect()
        for arm in ('B','C'):
            host=assemble(task,arm,state,proxy,device='cuda:0');host.audit_initial_state()
            for tensor in (host._proxy_input,host.proxy.mask,host.proxy.signed_distance):
                if tensor.device != torch.device('cuda:0'): raise ValueError('proxy device mismatch')
            if len(host._proxy_input)!=4: raise ValueError('proxy K changed')
            # Exactly one native joint step, no query GT is decoded for mechanical smoke.
            torch.cuda.reset_peak_memory_stats(0);start=time.perf_counter();host.step(pixels(task,35));torch.cuda.synchronize()
            p=host.prompt.data_prompt
            if not torch.isfinite(p.grad).all() or int(host.optimizer.state[p]['step'])!=1 or {m.sample_num for m in host.model.modules() if isinstance(m,host.adabn)}!={1} or host.memory_bank.get_size()!=1:
                raise ValueError('joint smoke lifecycle/gradient mismatch')
            source_unchanged(host,state)
            if any(p.requires_grad or p.grad is not None for p in host._proxy_model.parameters()): raise ValueError('clone gradient mutation')
            evidence[task]['arms'][arm]=dict(native_steps=1,proxy_K=4,prompt_gradient_norm=float(p.grad.norm()),elapsed_seconds=time.perf_counter()-start,peak_bytes=torch.cuda.max_memory_allocated(0),source_unchanged=True)
            del host,p;gc.collect()
        del state,proxy;gc.collect()
    write_json(out/'smoke.json',dict(status='PASS',native_steps=72,evidence=evidence))


def recompute(registration,out):
    result={}
    for task,reg in registration['tasks'].items():
        arms={}
        for arm in ARMS:
            complete=json.loads((out/f'{task}_{arm}.completion.json').read_text())
            if complete['status']!='COMPLETE' or complete['source_unchanged'] is not True: raise ValueError('arm completion missing')
            arms[arm]=[json.loads(line) for line in (out/f'{task}_{arm}.jsonl').read_text().splitlines()]
        result[task]=dict(summary=summarize(arms,reg['expected_visits'],task),observations={})
        for arm,rows in arms.items():
            result[task]['observations'][arm]=dict(visits=len(rows),adaptation_steps=sum(r['optimizer_steps_this_visit'] for r in rows),
                **{k:dict(mean=float(np.mean([r[k] for r in rows])),maximum=float(max(r[k] for r in rows))) for k in
                   ('prompt_state_delta_norm','optimizer_update_norm','pipeline_elapsed_seconds','host_step_elapsed_seconds')},
                peak_cuda_bytes=max(r['peak_cuda_bytes'] for r in rows))
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt',required=True,type=Path)
    parser.add_argument('--stage',required=True,choices=('register','smoke','pilot'))
    args=parser.parse_args(argv)
    receipt=json.loads(args.receipt.read_text())
    try:
        out=validate_receipt(receipt,args.stage)
    except Exception as error:
        print(json.dumps(dict(status='BLOCKED_RECEIPT',reason=type(error).__name__)))
        return 2
    try:
        if args.stage=='register':
            register_inputs(receipt,out)
        else:
            registration=json.loads((out/'registration.json').read_text());check_registered_files(registration)
            env=environment();write_json(out/f'{args.stage}_environment.json',env)
            if args.stage=='smoke':
                smoke(registration,out)
            else:
                if json.loads((out/'smoke.json').read_text())['status']!='PASS': raise ValueError('smoke required')
                for task,registered in registration['tasks'].items():
                    state=torch.load(registered['checkpoint']['path'],weights_only=True,map_location='cpu')
                    proxy=source_proxy(registered['selected']['proxy'],task)
                    execute_task(task,registered,state,proxy,out)
                    del state,proxy;gc.collect()
                summary=recompute(registration,out)
                write_json(out/'summary.private.json',summary)
                formal=sum(v['visits'] for task in summary.values() for v in task['observations'].values())
                updates=sum(v['adaptation_steps'] for task in summary.values() for v in task['observations'].values())
                if (formal,updates)!=(832,624): raise ValueError('formal aggregate coverage mismatch')
                if sum(p.stat().st_size for p in out.iterdir() if p.is_file()) > 256*1024**2: raise ValueError('private output disk budget exceeded')
                write_json(out/'pilot.completion.json',dict(status='SOURCE_PILOT_COMPLETE',formal_visits=formal,adaptation_steps=updates,smoke_native_steps=72,exit_code=0))
        print(json.dumps(dict(status='SOURCE_PILOT_COMPLETE' if args.stage=='pilot' else 'PASS',stage=args.stage,exit_code=0)))
        return 0
    except Exception as error:
        status='BLOCKED_INPUT_IDENTITY' if args.stage=='register' or 'BLOCKED_INPUT_IDENTITY' in str(error) else 'INCOMPLETE'
        write_json(out/f'{args.stage}.failure.json',dict(status=status,stage=args.stage,exception_type=type(error).__name__,private_reason=str(error),exit_code=1))
        print(json.dumps(dict(status=status,stage=args.stage,exit_code=1)))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
