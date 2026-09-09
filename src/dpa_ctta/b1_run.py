"""One B1 registration, 16-call smoke, detached finite stream and CPU closeout."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from .b1_host import Host,official,configure,reference_step,GRATA_COMMIT
from .b1_analysis import old_records,evaluate,recompute
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

ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/b1_grata_matched_baselines_v1.json'
BASELINE='2fcbc0a69645e34da42b97830935f0e290e488aa';P2_EXEC='d3ee6901379be293f47abd1687808caaf5b04266'
BUDGET=dict(records=7804,forwards=66334,backwards=11706,base_adam=7804,perturb=3902,restore=3902)


def metadata(path):
    p=Path(path);s=p.stat();return dict(path=str(p),bytes=s.st_size,mtime_ns=s.st_mtime_ns)


def register(p2,out):
    p2=Path(p2).resolve();receipt=json.loads((p2/'receipt.run.json').read_text())
    if receipt['commit']!=P2_EXEC or digest(p2/'registration.json')!=receipt['registration_sha256']:raise ValueError('P2 registration binding')
    if json.loads((p2/'verification.json').read_text())['status']!='P2_FROZEN_FULL_STREAM_COMPLETE':raise ValueError('P2 incomplete')
    p=json.loads((p2/'registration.json').read_text());f=p['tasks']['fundus'];paths={f['checkpoint']['path']}|{r[k] for r in f['target'] for k in ['image_path','mask_path']}
    reg=dict(p2_directory=str(p2),tasks={'fundus':f},identities=[i for i in p['identities'] if i['path'] in paths],formal_budget=BUDGET,limitations=p['limitations'],p2_registration_sha256=receipt['registration_sha256'])
    if len(f['target'])!=1951 or sum(v['remaining_dev'] for v in f['counts'].values())!=1695:raise ValueError('Fundus frozen pool')
    if paths!={i['path'] for i in reg['identities']}:raise ValueError('incomplete inherited asset bindings')
    check_registered_files(reg)
    for order in [0,1]:old_records(reg,order)
    reg['identities'] += [metadata(p2/name) for name in ['registration.json','receipt.run.json','verification.json','execution_audit.json','run.environment.json']]
    reg['identities'] += [metadata(p) for p in sorted(p2.glob('fundus_*.jsonl'))]
    private_json(out/'registration.json',reg)
    private_json(out/'registration.completion.json',dict(status='B1_REGISTERED',groups=1951,counts=f['counts'],reused_P2_targets=1951,old_controls_CPU_validated=True,formal_budget=BUDGET,exit_code=0))
    return reg


def smoke(receipt,out,reg):
    import sys
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels
    budget=P2Budget(out);calls=0;ev={};state=torch.load(reg['tasks']['fundus']['checkpoint']['path'],map_location='cpu',weights_only=True)
    call_log=new_log(out/'smoke.calls.jsonl')
    with deterministic_smoke_pair():
        seed_all(20260907);n=SourceOnlyHost('fundus',state,'cuda:0');candidate=SourceOnlyHost('fundus',state,'cuda:0');x=pixels('fundus',0)
        q=n.step(x);z=candidate.model(model_input_from_pixels(x,'fundus').to('cuda:0'))[0];close(q,z)
        parity=float((q-z).abs().max());del n,candidate,q,z;gc.collect()
        for arm in ['C','G']:
            saved=[];e=[];peak=0
            for path in ['reference','new']:
                budget.check();seed_all(20260907)
                if path=='reference':
                    model=SourceOnlyHost('fundus',state,'cuda:0').model;names,params=configure(model)
                    base=torch.optim.Adam(params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0);opt=official().GraTa(params,base,model,device='cuda:0')
                else:
                    host=Host(arm,state,'cuda:0');model,params,base,opt=host.model,host.params,host.base,host.opt
                def count_call(optimizer,args,kwargs):
                    nonlocal calls
                    calls+=1;append(call_log,dict(arm=arm,path=path,visit=i+1,completed_base_adam=calls))
                count_hook=base.register_step_post_hook(count_call)
                torch.cuda.reset_peak_memory_stats()
                for i in range(4):
                    x=pixels('fundus',i)
                    if path=='reference':logits=reference_step(model,opt,arm,model_input_from_pixels(x,'fundus'))
                    else:logits,meta=host.step(x)
                    snapshot=dict(logits=logits.cpu(),affine=[p.detach().cpu().clone() for p in params],adam=cpu_tree(copy.deepcopy(base.state_dict())),aux_state=cpu_tree(copy.deepcopy(opt.state_dict())),rng=rng())
                    if path=='reference':saved.append(snapshot)
                    else:
                        close(snapshot['rng'],saved[i]['rng'],exact=True)
                        for key in ['logits','affine','adam','aux_state']:close(snapshot[key],saved[i][key])
                        e.append(dict(visit=i+1,max_logits_difference=float((snapshot['logits']-saved[i]['logits']).abs().max()),counts=meta['counts'],lr=meta['lr'],state_and_RNG_match=True))
                    del logits,snapshot
                peak=max(peak,torch.cuda.max_memory_allocated());count_hook.remove()
                if path=='new':host.finish(state);del host
                else:
                    for key,value in model.state_dict().items():
                        if key not in names and not torch.equal(value.cpu(),state[key]):raise ValueError('reference changed frozen state')
                del model,params,base,opt;gc.collect()
            ev[arm]=dict(visits=e,peak_allocated_bytes=peak);print(json.dumps(dict(stage='smoke',arm=arm,status='PASS',base_adam=calls)),flush=True)
    call_log.close()
    if calls!=16:raise ValueError('smoke budget')
    private_json(out/'smoke.completion.json',dict(status='B1_SMOKE_PASS',base_adam=calls,evidence=ev,gpu_seconds=budget.seconds(),source_parity=parity,extra_source_parity_forwards=2,exit_code=0,**{k:receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']}))


def run(receipt,out,reg):
    budget=P2Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress={k:0 for k in BUDGET};where={};host=None
    state=torch.load(reg['tasks']['fundus']['checkpoint']['path'],map_location='cpu',weights_only=True)
    private_json(out/'run.start.json',dict(status='B1_RUNNING',commit=receipt['commit']))
    try:
        for order in [0,1]:
            stream=expected(reg,'fundus',order)
            for arm in ['C','G']:
                seed_all(20260907);host=Host(arm,state,'cuda:0');torch.cuda.reset_peak_memory_stats()
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
        if progress!=BUDGET:raise ValueError('B1 counts')
        check_registered_files(reg)
        private_json(out/'run.completion.json',dict(status='B1_RUN_COMPLETE',progress=progress,trainable=trainable,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        private_json(out/'run.failure.json',dict(status='INCOMPLETE',reason=str(e),where=where,progress=progress,current_counts=host.counts if host else None,gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(r,stage):
    if r['authorization']!='direct_user_B1_GRATA_MATCHED_BASELINES':raise ValueError('new B1 authorization required')
    if subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()!=r['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('B1 execution checkout')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('old scientific code changed')
    if digest(CONFIG)!=r['config_sha256'] or r['reference_commit']!=REFERENCE_COMMIT or r['grata_commit']!=GRATA_COMMIT:raise ValueError('B1 code bindings')
    checkout_root();out=Path(r['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_mode&0o077 or out.stat().st_uid!=os.getuid():raise ValueError('private output')
    if digest(out/'registration.json')!=r['registration_sha256']:raise ValueError('registration identity')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg)
    if reg['formal_budget']!=BUDGET or json.loads(CONFIG.read_text())['formal_budget']!=BUDGET:raise ValueError('B1 frozen budget')
    if stage!='recompute':
        official();devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(p[0]):p[1].strip() for p in [s.split(',') for s in devices.splitlines()]}
        if r['physical_gpu'] not in [4,5,6,7] or mapping[r['physical_gpu']]!=r['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=r['gpu_uuid']:raise ValueError('B1 GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='B1_SMOKE_PASS' or s['base_adam']!=16 or digest(out/'smoke.completion.json')!=r['smoke_sha256'] or any(s[k]!=r[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('B1 smoke gate')
    return out,reg


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();os.umask(0o077)
    if a.stage=='smoke':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    r=json.loads(a.receipt.read_text());out=Path(r['output_directory'])
    try:
        out,reg=validate(r,a.stage)
        if a.stage!='recompute':
            env=environment();private_json(out/(a.stage+'.environment.json'),env)
            if env!=json.loads((Path(reg['p2_directory'])/'run.environment.json').read_text()):raise ValueError('P2 environment drift')
        if a.stage=='recompute':recompute(out,reg,r)
        else:{'smoke':smoke,'run':run}[a.stage](r,out,reg)
    except Exception as e:
        f=out/(a.stage+'.failure.json')
        if not f.exists():private_json(f,dict(status='INCOMPLETE',stage=a.stage,reason=str(e),exit_code=1))
        raise
