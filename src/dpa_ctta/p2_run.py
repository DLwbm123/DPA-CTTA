"""P2 fixed shared-teacher schedule; unchanged P1/native adapting parents."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from .p2_data import ARMS,PARENTS,VIEWS,expected,load_proxy
from .p2_analysis import evaluate,pixel_cells
from .p1_run import update,teacher,sync
from .p1_host import ensemble,check_probability
from .m1_run import Observed,make_host,new_log,Budget
from .m2_run import deterministic_smoke_pair
from .m2_registration import inspect_history
from .offline.adaptation_dd import restore_history,cpu_tree
from .host_diagnostic import snapshot,rng,restore,close
from .host_diagnostic_run import private_json,append
from .source_pilot import seed_all
from .source_io import read_pixels,read_mask
from .source_pilot_release import digest,environment,check_registered_files,source_unchanged
from .integrations.ctta_suite import checkout_root,REFERENCE_COMMIT
ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/p2_frozen_factorial_full_stream_v1.json';BASELINE='18b8f32186678e8930a6c540f96941365fd511b9'

class P2Budget(Budget):
    def check(self):
        super().check()
        if sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())>=1024**3:raise RuntimeError('P2 1 GiB cap')


def parent_visit(host,watch,x,state):
    restore(state);p,m=update(host,watch,x);return p,m,rng()


class Bundle:
    def __init__(self,task,state,reg,device='cuda:0'):
        self.n=teacher(task,state,device);self.tw=Observed(self.n);self.hosts={};self.watches={};self.rngs={}
        for a in PARENTS:
            seed_all(20260907);h=make_host(task,'A' if a=='A' else 'R',state,None if a=='A' else load_proxy(reg,task,a),device=device)
            self.hosts[a]=h;self.watches[a]=Observed(h);self.rngs[a]=rng()
    def step(self,x):
        sync();start=time.monotonic();q=self.n.step(x).detach().sigmoid();sync();seconds=time.monotonic()-start
        self.tw.verify();check_probability(q);ps={'N':q};ms={}
        for a in PARENTS:
            ps[a],ms[a],self.rngs[a]=parent_visit(self.hosts[a],self.watches[a],x,self.rngs[a])
        zero={k:0 for k in ms['A']['counts']}
        ms['N']=dict(counts=dict(zero,model_forwards=1,teacher_forwards=1),host_step_elapsed_seconds=0.,parent_adam_step=0,parent_counters=[],parent_memory_size=0,anchor=None)
        for e,b in VIEWS.items():ps[e]=ensemble(q,ps[b]);ms[e]=dict(ms[b],counts=dict(zero))
        for a,m in ms.items():
            m['teacher_seconds']=seconds if a=='N' or a in VIEWS else 0.
            m['deployment_step_seconds']=m['host_step_elapsed_seconds']+m['teacher_seconds']
        return ps,ms
    def finish(self,state):
        for h in [self.n,*self.hosts.values()]:source_unchanged(h,state)
        if any(p.grad is not None for p in self.n.model.parameters()):raise ValueError('teacher gradient')
        for w in [self.tw,*self.watches.values()]:w.release()


def make_record(task,order,arm,i,row,p,meta,metrics,cells,elapsed,peak):
    b=VIEWS.get(arm,arm)
    return dict(task=task,order=order,arm=arm,visit=i+1,sample_id=row['sample_id'],group_id=row['group_id'],domain=row['domain'],subset=row['subset'],parent=b,prediction_id=f'{task}:{order}:{i+1}:{arm}',parent_prediction_ids=[f'{task}:{order}:{i+1}:N',f'{task}:{order}:{i+1}:{b}'] if arm in VIEWS else [],formula='.5*q0+.5*p_parent' if arm in VIEWS else None,parent_state_tag=f'{task}:{order}:{b}:{i+1}',probability_mean=float(p.mean()),source_unchanged=True,prediction_fixed_before_label=True,pipeline_elapsed_seconds=elapsed,peak_allocated_bytes=peak,metrics=metrics,pixel_cells=cells,**meta)


def smoke(receipt,out,reg):
    import sys
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels
    budget=P2Budget(out);completed=0;evidence={};watch=None
    try:
        with deterministic_smoke_pair():
            for task,r in reg['tasks'].items():
                budget.check();torch.cuda.reset_peak_memory_stats();state=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True)
                n=teacher(task,state,'cuda:0');nw=Observed(n);x=pixels(task,0);q=n.step(x).detach().sigmoid();nw.verify();history=inspect_history(r['history']['path'],task)[16];ev={}
                for a in PARENTS:
                    reference=None
                    for mode in ['native','p2']:
                        seed_all(20260907);h=make_host(task,'A' if a=='A' else 'R',state,None if a=='A' else load_proxy(r,task,a));h._initial_hook.remove();h._started=True;restore_history(h,history);watch=Observed(h)
                        captured=[]
                        def capture(model,args,output):
                            if not torch.is_grad_enabled():captured.append((output[0] if isinstance(output,(tuple,list)) else output).detach().cpu())
                        handle=h.model.register_forward_hook(capture);before_rng=rng()
                        if mode=='native':
                            logits=h.step(x);watch.verify();p=logits.detach().sigmoid();after_rng=rng()
                        else:p,_,after_rng=parent_visit(h,watch,x,before_rng)
                        completed+=1
                        if len(captured)!=1:raise ValueError('final logits capture')
                        state_after=cpu_tree(snapshot(h));actual=dict(logits=captured[0],state=state_after,rng=after_rng)
                        if mode=='native':reference=actual
                        else:
                            close(actual,reference)
                            ev[a]=dict(max_logits_difference=float((actual['logits']-reference['logits']).abs().max()),counts=watch.counts.copy(),native_state_and_rng_equal=True)
                        fused=ensemble(q,p);close(state_after,cpu_tree(snapshot(h)),exact=True)
                        # Procedural labels only, after frozen predictions; no extra step.
                        mask=torch.zeros_like(fused).cpu();evaluate(fused,mask,task);pixel_cells(mask,q,p,fused)
                        source_unchanged(h,state);handle.remove();watch.release();watch=None;del h,p,fused,actual,state_after,captured;gc.collect()
                    del reference
                source_unchanged(n,state);nw.release();ev['teacher_frozen']=True;ev['peak_allocated_bytes']=torch.cuda.max_memory_allocated();evidence[task]=ev
                del n,nw,q,state,history;gc.collect();print(json.dumps(dict(stage='smoke',task=task,status='PASS',online=completed)),flush=True)
        if completed!=12:raise ValueError('smoke count')
        private_json(out/'smoke.completion.json',dict(status='P2_SMOKE_PASS',online=completed,evidence=evidence,gpu_seconds=budget.seconds(),exit_code=0,**{k:receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']}))
    except Exception as e:
        private_json(out/'smoke.failure.json',dict(status='P2_PARTIAL',online_completed=completed,current_counts=watch.counts if watch else None,evidence=evidence,reason=str(e),gpu_seconds=budget.seconds(),exit_code=1));raise


def run(receipt,out,reg):
    budget=P2Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress=dict(records=0,online=0,teacher_forwards=0,outer=0,inner=0);where={};bundle=None
    private_json(out/'run.start.json',dict(status='P2_RUNNING',commit=receipt['commit']))
    try:
        for task,r in reg['tasks'].items():
            state=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True)
            for order in [0,1]:
                stream=expected(reg,task,order);bundle=Bundle(task,state,r);torch.cuda.reset_peak_memory_stats()
                with contextlib.ExitStack() as stack:
                    logs={a:stack.enter_context(new_log(out/f'{task}_{order}_{a}.jsonl')) for a in ARMS}
                    for i,row in enumerate(stream):
                        where=dict(task=task,order=order,visit=i+1);budget.check();start=time.monotonic();x=read_pixels(row['image_path'],task,row['image_size'])
                        ps,ms=bundle.step(x);progress['online']+=3;progress['teacher_forwards']+=1
                        # All seven probabilities fixed; only now may the evaluator read GT.
                        mask=read_mask(row['mask_path'],task,row['image_size'])
                        for a in ARMS:
                            metrics=evaluate(ps[a],mask,task);cells=pixel_cells(mask,ps['N'],ps[VIEWS[a]],ps[a]) if a in VIEWS else None
                            rec=make_record(task,order,a,i,row,ps[a],ms[a],metrics,cells,time.monotonic()-start,torch.cuda.max_memory_allocated());append(logs[a],rec);progress['records']+=1
                        del x,mask,ps,ms
                        if (i+1)%64==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
                bundle.finish(state);bundle=None;gc.collect()
            del state;gc.collect()
        if progress!={k:reg['budget'][k] for k in progress}:raise ValueError('formal counts')
        private_json(out/'run.completion.json',dict(status='P2_RUN_COMPLETE',progress=progress,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        private_json(out/'run.failure.json',dict(status='P2_PARTIAL',progress=progress,where=where,current_counts={a:w.counts for a,w in bundle.watches.items()} if bundle else None,reason=str(e),gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(receipt,stage):
    if receipt['authorization']!='direct_user_P2_FROZEN_FULL_STREAM':raise ValueError('authorization')
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if head!=receipt['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('execution commit')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('old science changed')
    if digest(CONFIG)!=receipt['config_sha256'] or receipt['reference_commit']!=REFERENCE_COMMIT:raise ValueError('config/reference')
    checkout_root();out=Path(receipt['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_uid!=os.getuid() or out.stat().st_mode&0o077:raise ValueError('private output')
    if digest(out/'registration.json')!=receipt['registration_sha256']:raise ValueError('registration identity')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg);cfg=json.loads(CONFIG.read_text())
    if reg['budget']!=cfg['expected_budget'] or reg['scheduler']!=cfg['scheduler']:raise ValueError('budget/schedule')
    if stage!='recompute':
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True);devices={int(p[0]):p[1].strip() for p in [line.split(',') for line in lines.splitlines()]}
        if receipt['physical_gpu'] not in range(3,8) or devices[receipt['physical_gpu']]!=receipt['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=receipt['gpu_uuid']:raise ValueError('GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='P2_SMOKE_PASS' or s['online']!=12 or digest(out/'smoke.completion.json')!=receipt['smoke_sha256'] or any(s[k]!=receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('smoke gate')
    return out,reg


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();os.umask(0o077)
    if a.stage=='smoke':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    r=json.loads(a.receipt.read_text());out,reg=validate(r,a.stage)
    if a.stage=='recompute':
        from .p2_analysis import recompute
        recompute(out,reg,r)
    else:
        env=environment();private_json(out/(a.stage+'.environment.json'),env)
        if env!=json.loads((Path(reg['p1_directory'])/'run.environment.json').read_text()):raise ValueError('P1 environment drift')
        (smoke if a.stage=='smoke' else run)(r,out,reg)
