"""One B2 registration, 20-call smoke, detached finite stream and CPU closeout."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from .b1_host import Host as CHost,official,GRATA_COMMIT
from .b2_host import Host
from .b2_analysis import old_records,evaluate,recompute
from .p2_data import expected
from .p2_run import P2Budget
from .source_pilot import seed_all,SourceOnlyHost
from .source_pilot_release import digest,check_registered_files,environment
from .source_io import read_pixels,read_mask
from .host_diagnostic_run import private_json,append
from .host_diagnostic import rng,restore,close
from .offline.adaptation_dd import cpu_tree
from .m2_run import deterministic_smoke_pair
from .m1_run import new_log
from .p1_run import sync
from .hosts.vptta import model_input_from_pixels
from .integrations.ctta_suite import checkout_root,REFERENCE_COMMIT

ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/b2_interval_consistency_v1.json'
BASELINE='fd40597ba05d6333a3fb4f41cebf466eeb9ee063';P2_EXEC='d3ee6901379be293f47abd1687808caaf5b04266'
BUDGET=dict(records=11706,forwards=93648,backwards=11706,base_adam=11706,perturb=0,restore=0)


def metadata(path):
    p=Path(path);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def register(b1,out):
    b1=Path(b1).resolve();receipt=json.loads((b1/'receipt.run.json').read_text())
    if receipt['commit']!='029a79340057b58b8074ccf86c7b5671b5fdbd1c' or digest(b1/'registration.json')!=receipt['registration_sha256']:raise ValueError('B1 registration binding')
    if json.loads((b1/'verification.json').read_text())['status']!='B1_MATCHED_BASELINE_COMPARISON_COMPLETE':raise ValueError('B1 incomplete')
    reg=json.loads((b1/'registration.json').read_text());reg.update(b1_directory=str(b1),b1_registration_sha256=receipt['registration_sha256'],formal_budget=BUDGET)
    reg['identities'] += [metadata(b1/name) for name in ['registration.json','receipt.run.json','verification.json','execution_audit.json','run.environment.json']]
    reg['identities'] += [metadata(p) for p in sorted(b1.glob('fundus_*.jsonl'))]
    check_registered_files(reg)
    for order in (0,1):old_records(reg,order)
    private_json(out/'registration.json',reg)
    private_json(out/'registration.completion.json',dict(status='B2_REGISTERED',groups=1951,counts=reg['tasks']['fundus']['counts'],old_controls_CPU_validated=True,formal_budget=BUDGET,exit_code=0))
    return reg


def smoke(receipt,out,reg):
    import sys
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels
    budget=P2Budget(out);calls=0;ev={};state=torch.load(reg['tasks']['fundus']['checkpoint']['path'],map_location='cpu',weights_only=True)
    saved=[];physical=dict(forwards=0,backwards=0,base_adam=0,perturb=0,restore=0)
    host=None;completed_trajectories=physical.copy()
    try:
        with new_log(out/'smoke.calls.jsonl') as call_log,deterministic_smoke_pair():
            for arm in ('C','zero','U','S','I'):
                budget.check();seed_all(20260907);host=CHost('C',state,'cuda:0') if arm=='C' else Host(arm,state,'cuda:0')
                if len(host.names)!=82 or sum(p.numel() for p in host.params)!=19136:raise ValueError('BN registration')
                def count_call(optimizer,args,kwargs):
                    nonlocal calls
                    calls+=1;append(call_log,dict(arm=arm,visit=i+1,completed_base_adam=calls,current_counts=host.counts.copy()))
                count_hook=host.base.register_step_post_hook(count_call);torch.cuda.reset_peak_memory_stats();e=[];active_total=0
                for i in range(4):
                    logits,meta=host.step(pixels('fundus',i))
                    for key in physical:physical[key]+=meta['counts'][key]
                    snapshot=dict(logits=logits.cpu(),affine=[p.detach().cpu().clone() for p in host.params],adam=cpu_tree(copy.deepcopy(host.base.state_dict())),rng=rng())
                    if arm=='C':saved.append(snapshot)
                    elif arm=='zero':
                        close(snapshot['rng'],saved[i]['rng'],exact=True)
                        for key in ('logits','affine','adam'):close(snapshot[key],saved[i][key])
                    if arm!='C':active_total+=sum(v['active'] for c in meta['diagnostics']['channels'] for v in c['partitions'].values())
                    e.append(dict(visit=i+1,counts=meta['counts'],diagnostics=meta['diagnostics'],zero_C_state_RNG_match=True if arm=='zero' else None,max_logits_difference=float((snapshot['logits']-saved[i]['logits']).abs().max()) if arm=='zero' else None))
                    del logits,snapshot
                if arm not in ('C','zero') and active_total==0:raise ValueError('procedural smoke has no active pixels')
                count_hook.remove();host.finish(state)
                ev[arm]=dict(visits=e,peak_allocated_bytes=torch.cuda.max_memory_allocated(),active_total=active_total)
                completed_trajectories=physical.copy();host=None;gc.collect();print(json.dumps(dict(stage='smoke',arm=arm,status='PASS',base_adam=calls)),flush=True)
    except Exception as e:
        actual={k:completed_trajectories[k]+(host.counts[k] if host else 0) for k in physical}
        private_json(out/'smoke.failure.json',dict(status='INCOMPLETE',reason=str(e),physical=actual,base_adam=calls,gpu_seconds=budget.seconds(),exit_code=1));raise

    if calls!=20 or physical!=dict(forwards=160,backwards=20,base_adam=20,perturb=0,restore=0):raise ValueError('smoke budget')
    private_json(out/'smoke.completion.json',dict(status='B2_SMOKE_PASS',base_adam=calls,physical=physical,evidence=ev,gpu_seconds=budget.seconds(),exit_code=0,**{k:receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']}))


def run(receipt,out,reg):
    budget=P2Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress={k:0 for k in BUDGET};where={};host=None
    state=torch.load(reg['tasks']['fundus']['checkpoint']['path'],map_location='cpu',weights_only=True)
    private_json(out/'run.start.json',dict(status='B2_RUNNING',commit=receipt['commit']))
    try:
        for order in [0,1]:
            stream=expected(reg,'fundus',order)
            for arm in ['U','S','I']:
                seed_all(20260907);host=Host(arm,state,'cuda:0');torch.cuda.reset_peak_memory_stats()
                if len(host.names)!=82 or sum(p.numel() for p in host.params)!=19136:raise ValueError('BN registration')
                trainable=dict(names=host.names,layers=[n[:-7] for n in host.names if n.endswith('.weight')],parameters=sum(p.numel() for p in host.params),parameter_tensors=len(host.params),state_mapping='identity: strict full P2 segmentation state; no auxiliary heads')
                if not (out/'trainable.json').exists():private_json(out/'trainable.json',trainable)
                elif json.loads((out/'trainable.json').read_text())!=trainable:raise ValueError('BN list drift')
                with new_log(out/f'fundus_{order}_{arm}.jsonl') as f:
                    for i,row in enumerate(stream):
                        where=dict(order=order,arm=arm,visit=i+1);budget.check();start=time.monotonic();x=read_pixels(row['image_path'],'fundus',row['image_size'])
                        sync();hoststart=time.monotonic();logits,m=host.step(x);sync();hostseconds=time.monotonic()-hoststart
                        for k,v in m['counts'].items():progress[k]+=v
                        probability=logits.sigmoid();mask=read_mask(row['mask_path'],'fundus',row['image_size']);metrics=evaluate(probability,mask,'fundus')
                        record=dict(task='fundus',**where,**{k:row[k] for k in ['domain','subset','sample_id','group_id']},**m,metrics=metrics,prediction_fixed_before_label=True,frozen_parameters_checked=True,host_seconds=hostseconds,pipeline_seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                        append(f,record);progress['records']+=1;del logits,probability,mask,x
                        if (i+1)%64==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
                host.finish(state);host=None;gc.collect()
        if progress!=BUDGET:raise ValueError('B2 counts')
        check_registered_files(reg)
        private_json(out/'run.completion.json',dict(status='B2_RUN_COMPLETE',progress=progress,trainable=trainable,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        private_json(out/'run.failure.json',dict(status='INCOMPLETE',reason=str(e),where=where,progress=progress,current_counts=host.counts if host else None,gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(r,stage):
    if r['authorization']!='direct_user_B2_INTERVAL_CONSISTENCY':raise ValueError('new B2 authorization required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=r['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('B2 execution checkout')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('old scientific code changed')
    if digest(CONFIG)!=r['config_sha256'] or r['reference_commit']!=REFERENCE_COMMIT or r['grata_commit']!=GRATA_COMMIT:raise ValueError('B2 code bindings')
    checkout_root();out=Path(r['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_mode&0o077 or out.stat().st_uid!=os.getuid():raise ValueError('private output')
    if digest(out/'registration.json')!=r['registration_sha256']:raise ValueError('registration identity')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg)
    if reg['formal_budget']!=BUDGET or json.loads(CONFIG.read_text())['formal_budget']!=BUDGET:raise ValueError('B2 frozen budget')
    cpu=json.loads((out/'CPU_validation.json').read_text())
    if cpu['exit_code']!=0 or cpu['tests']!=9 or cpu['commit']!=r['commit']:raise ValueError('B2 CPU acceptance gate')
    if stage!='recompute':
        official();devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(p[0]):p[1].strip() for p in [s.split(',') for s in devices.splitlines()]}
        if r['physical_gpu'] not in [4,5,6,7] or mapping[r['physical_gpu']]!=r['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']:raise ValueError('B2 GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='B2_SMOKE_PASS' or s['base_adam']!=20 or digest(out/'smoke.completion.json')!=r['smoke_sha256'] or any(s[k]!=r[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('B2 smoke gate')
    return out,reg


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();os.umask(0o077)
    if a.stage=='smoke':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    r=json.loads(a.receipt.read_text());out=Path(r['output_directory'])
    try:
        out,reg=validate(r,a.stage)
        if a.stage!='recompute':
            env=environment();private_json(out/(a.stage+'.environment.json'),env)
            if env!=json.loads((Path(reg['b1_directory'])/'run.environment.json').read_text()):raise ValueError('B1 environment drift')
        if a.stage=='recompute':recompute(out,reg,r)
        else:{'smoke':smoke,'run':run}[a.stage](r,out,reg)
    except Exception as e:
        f=out/(a.stage+'.failure.json')
        if not f.exists():private_json(f,dict(status='INCOMPLETE',stage=a.stage,reason=str(e),exit_code=1))
        raise
