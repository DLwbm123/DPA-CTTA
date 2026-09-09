"""One P1 smoke and fixed five-trajectory/eight-output deployment evaluation."""
import argparse
import contextlib
import gc
import json
import os
from pathlib import Path
import subprocess
import time
import torch
from .p1_host import SourceAnchorHost,ensemble,check_probability
from .p1_data import ARMS,PARENTS,expected,load_proxy
from .p1_analysis import evaluate,validate_rows
from .m1_run import make_host,Observed,new_log,Budget
from .m2_run import deterministic_smoke_pair
from .m2_registration import inspect_history
from .offline.adaptation_dd import restore_history,cpu_tree
from .host_diagnostic import rng,restore,snapshot,close
from .host_diagnostic_run import private_json,append
from .source_pilot import seed_all
from .source_pilot_release import digest,check_registered_files,environment,source_unchanged
from .source_io import read_pixels,read_mask
from .integrations.ctta_suite import checkout_root,REFERENCE_COMMIT
ROOT=Path(__file__).resolve().parents[2];CONFIG=ROOT/'configs/p1_no_dd_current_image_v1.json'
BASELINE='fab4ce4f7733e54258d577fe5e22cea8a562b6d6'


class P1Budget(Budget):
    def check(self):
        super().check()
        if sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())>=256*1024**2:raise RuntimeError('P1 256 MiB output cap')


def sync():
    if torch.cuda.is_initialized():torch.cuda.synchronize()


def teacher(task,state,device):
    h=make_host(task,'N',state,device=device)
    if any(p.requires_grad or p.grad is not None for p in h.model.parameters()):raise ValueError('teacher not frozen')
    return h


def update(h,watch,x,q=None):
    before=dict(watch.counts);sync();start=time.monotonic()
    logits=h.step(x) if q is None else h.step(x,q);sync();elapsed=time.monotonic()-start
    watch.verify();check_probability(logits.sigmoid())
    for v in h.optimizer.state[h.prompt.data_prompt].values():
        if isinstance(v,torch.Tensor) and not torch.isfinite(v).all():raise ValueError('nonfinite Adam')
    counts={k:v-before[k] for k,v in watch.counts.items()};counts['teacher_forwards']=0
    if counts['online_adam']!=1 or counts['memory_pushes']!=1 or counts['backward_calls']!=1:raise ValueError('native step count')
    return logits.detach().sigmoid(),dict(counts=counts,host_step_elapsed_seconds=elapsed,parent_adam_step=int(h.optimizer.state[h.prompt.data_prompt]['step']),parent_counters=sorted({m.sample_num for m in h.model.modules() if isinstance(m,h.adabn)}),parent_memory_size=len(h.memory_bank.memory),anchor=getattr(h,'last_anchor',None))


class Pair:
    """Current-image-only algorithm. All five outputs fixed before evaluator access."""
    def __init__(self,task,state,device='cuda:0'):
        self.n=teacher(task,state,device);self.tw=Observed(self.n);self.hosts={};self.watches={};self.rngs={}
        for a in ['A','SA']:
            seed_all(20260907)
            self.hosts[a]=make_host(task,'A',state,device=device) if a=='A' else SourceAnchorHost(task,state,device)
            self.watches[a]=Observed(self.hosts[a]);self.rngs[a]=rng()
    def step(self,x):
        sync();start=time.monotonic();q=self.n.step(x).detach().sigmoid();sync();teacher_seconds=time.monotonic()-start
        self.tw.verify();check_probability(q)
        predictions={'N':q};meta={}
        for a in ['A','SA']:
            restore(self.rngs[a]);p,m=update(self.hosts[a],self.watches[a],x,q if a=='SA' else None);self.rngs[a]=rng()
            predictions[a]=p;predictions['ENS_'+a]=ensemble(q,p);meta[a]=m
        zero={k:0 for k in meta['A']['counts']};meta['N']=dict(counts=dict(zero,model_forwards=1,teacher_forwards=1),host_step_elapsed_seconds=0.,parent_adam_step=0,parent_counters=[],parent_memory_size=0,anchor=None)
        for a in ['ENS_A','ENS_SA']:meta[a]=dict(meta[PARENTS[a]],counts=dict(zero))
        for a,m in meta.items():
            m['teacher_seconds']=teacher_seconds if a!='A' else 0.
            m['deployment_step_seconds']=m['host_step_elapsed_seconds']+m['teacher_seconds']
        return predictions,meta
    def finish(self,state):
        for h in [self.n,*self.hosts.values()]:source_unchanged(h,state)
        if any(p.grad is not None for p in self.n.model.parameters()):raise ValueError('teacher grad')
        for w in [self.tw,*self.watches.values()]:w.release()


def smoke(receipt,out,reg):
    import sys
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels
    budget=P1Budget(out);updates=0;evidence={};watch=None
    try:
        with deterministic_smoke_pair():
            for task,r in reg['tasks'].items():
                budget.check();torch.cuda.reset_peak_memory_stats();state=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True)
                t=teacher(task,state,'cuda:0');tw=Observed(t);x=pixels(task,0);q=t.step(x).sigmoid();tw.verify()
                history=inspect_history(r['history']['path'],task)[16];reference=None;ev={}
                for arm in ['A','SA0','SA','O2','D4','L4']:
                    seed_all(20260907)
                    h=SourceAnchorHost(task,state,'cuda:0',weight=0 if arm=='SA0' else .1) if arm in ['SA0','SA'] else make_host(task,'A' if arm=='A' else 'R',state,None if arm=='A' else load_proxy(r,task,arm))
                    h._initial_hook.remove();h._started=True;restore_history(h,history);watch=Observed(h)
                    p,m=update(h,watch,x,q if arm.startswith('SA') else None);updates+=1
                    captured=cpu_tree(snapshot(h))
                    if arm=='A':reference=(p.cpu(),captured)
                    if arm=='SA0':close(p.cpu(),reference[0]);close(captured,reference[1])
                    before=cpu_tree(snapshot(h));fused=ensemble(q,p);close(before,cpu_tree(snapshot(h)),exact=True)
                    source_unchanged(h,state);ev[arm]=dict(counts=m['counts'],anchor=m['anchor'],finite=True)
                    if arm=='SA0':ev[arm]['max_native_prediction_difference']=float((p.cpu()-reference[0]).abs().max())
                    watch.release();watch=None;del h,p,fused,captured,before;gc.collect()
                source_unchanged(t,state);tw.release();ev['teacher_frozen']=True;ev['peak_allocated_bytes']=torch.cuda.max_memory_allocated();evidence[task]=ev
                del t,tw,q,state,history,reference;gc.collect();print(json.dumps(dict(stage='smoke',task=task,status='PASS',online=updates)),flush=True)
        if updates!=12:raise ValueError('smoke update coverage')
        private_json(out/'smoke.completion.json',dict(status='P1_SMOKE_PASS',online=updates,evidence=evidence,gpu_seconds=budget.seconds(),exit_code=0,**{k:receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']}))
    except Exception as e:
        private_json(out/'smoke.failure.json',dict(status='P1_PARTIAL',online_completed=updates,current_counts=watch.counts if watch else None,reason=str(e),evidence=evidence,gpu_seconds=budget.seconds(),exit_code=1));raise


def record(task,order,arm,i,row,p,m,pipeline,peak):
    parent=PARENTS[arm];ens=arm.startswith('ENS_')
    return dict(task=task,order=order,arm=arm,visit=i+1,domain=row['domain'],sample_id=row['sample_id'],group_id=row['group_id'],subset=row['subset'],parent=parent,
        prediction_id=f'{task}:{order}:{i+1}:{arm}',parent_prediction_ids=[f'{task}:{order}:{i+1}:N',f'{task}:{order}:{i+1}:{parent}'] if ens else [],formula='.5*q0+.5*p_parent' if ens else None,
        parent_state_tag=f'{task}:{order}:{parent}:{i+1}',probability_mean=float(p.mean()),source_unchanged=True,prediction_fixed_before_label=True,pipeline_elapsed_seconds=pipeline,peak_allocated_bytes=peak,**m)


def run(receipt,out,reg):
    budget=P1Budget(out,json.loads((out/'smoke.completion.json').read_text())['gpu_seconds']);progress=dict(records=0,online=0,outer=0,inner=0,teacher_forwards=0);where={};active=None
    private_json(out/'run.start.json',dict(status='P1_RUNNING',commit=receipt['commit']))
    try:
        for task,r in reg['tasks'].items():
            state=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True)
            for order in [0,1]:
                stream=expected(reg,task,order);pair=Pair(task,state);active=pair;torch.cuda.reset_peak_memory_stats()
                with contextlib.ExitStack() as stack:
                    logs={a:stack.enter_context(new_log(out/f'{task}_{order}_{a}.jsonl')) for a in ['N','A','ENS_A','SA','ENS_SA']}
                    for i,row in enumerate(stream):
                        where=dict(task=task,order=order,trajectory='A+SA',visit=i+1);budget.check();start=time.monotonic()
                        x=read_pixels(row['image_path'],task,row['image_size']);ps,ms=pair.step(x);progress['online']+=2;progress['teacher_forwards']+=1
                        # All five predictions, including both ensembles, are fixed here.
                        mask=read_mask(row['mask_path'],task,row['image_size'])
                        for a in logs:
                            metrics=evaluate(ps[a],mask,task);rec=record(task,order,a,i,row,ps[a],ms[a],time.monotonic()-start,torch.cuda.max_memory_allocated());rec['metrics']=metrics
                            append(logs[a],rec);progress['records']+=1
                        del x,mask,ps,ms
                        if (i+1)%32==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
                pair.finish(state);active=None;del pair;gc.collect()
                for arm in ['O2','D4','L4']:
                    h=make_host(task,'R',state,load_proxy(r,task,arm));watch=Observed(h);active=watch;torch.cuda.reset_peak_memory_stats()
                    with new_log(out/f'{task}_{order}_{arm}.jsonl') as f:
                        for i,row in enumerate(stream):
                            where=dict(task=task,order=order,trajectory=arm,visit=i+1);budget.check();start=time.monotonic()
                            x=read_pixels(row['image_path'],task,row['image_size']);p,m=update(h,watch,x);progress['online']+=1
                            m.update(teacher_seconds=0.,deployment_step_seconds=m['host_step_elapsed_seconds'])
                            mask=read_mask(row['mask_path'],task,row['image_size']);metrics=evaluate(p,mask,task)
                            rec=record(task,order,arm,i,row,p,m,time.monotonic()-start,torch.cuda.max_memory_allocated());rec['metrics']=metrics;append(f,rec);progress['records']+=1
                            del p,x,mask
                            if (i+1)%32==0:print(json.dumps(dict(**where,progress=progress,gpu_seconds=budget.seconds())),flush=True)
                    source_unchanged(h,state);watch.release();active=None;del h,watch;gc.collect()
            del state;gc.collect()
        b=reg['budget']
        if progress!=dict(records=b['records'],online=b['online'],outer=0,inner=0,teacher_forwards=2*b['groups']):raise ValueError('formal coverage')
        private_json(out/'run.completion.json',dict(status='P1_RUN_COMPLETE',progress=progress,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as e:
        counts={a:w.counts for a,w in active.watches.items()} if isinstance(active,Pair) else active.counts if active else None
        private_json(out/'run.failure.json',dict(status='P1_PARTIAL',progress=progress,where=where,current_counts=counts,reason=str(e),gpu_seconds=budget.seconds(),exit_code=1));raise


def validate(receipt,stage):
    if receipt['authorization']!='direct_user_P1_NO_DD_CURRENT_IMAGE':raise ValueError('P1 authorization')
    head=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if head!=receipt['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('dirty/wrong execution commit')
    changes=subprocess.check_output(['git','-C',str(ROOT),'diff','--name-status',BASELINE,'HEAD','--','src','configs'],text=True)
    if any(not line.startswith('A\t') for line in changes.splitlines()):raise ValueError('inherited scientific file changed')
    if digest(CONFIG)!=receipt['config_sha256'] or receipt['reference_commit']!=REFERENCE_COMMIT:raise ValueError('config/reference')
    checkout_root();out=Path(receipt['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_uid!=os.getuid() or out.stat().st_mode&0o077:raise ValueError('private output directory')
    if digest(out/'registration.json')!=receipt['registration_sha256']:raise ValueError('registration drift')
    reg=json.loads((out/'registration.json').read_text());check_registered_files(reg)
    if stage!='recompute':
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(p[0]):p[1].strip() for p in [s.split(',') for s in devices.splitlines()]}
        if receipt['physical_gpu'] not in range(3,8) or mapping[receipt['physical_gpu']]!=receipt['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=receipt['gpu_uuid']:raise ValueError('GPU binding')
    if stage!='smoke':
        s=json.loads((out/'smoke.completion.json').read_text())
        if s['status']!='P1_SMOKE_PASS' or s['online']!=12 or digest(out/'smoke.completion.json')!=receipt['smoke_sha256'] or any(s[k]!=receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('smoke gate')
    return out,reg


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args();os.umask(0o077)
    if a.stage=='smoke':os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    r=json.loads(a.receipt.read_text());out,reg=validate(r,a.stage)
    if a.stage=='recompute':
        from .p1_analysis import recompute
        recompute(out,reg,r)
    else:
        env=environment();prior=json.loads((Path(reg['m4_directory'])/'run.environment.json').read_text());private_json(out/(a.stage+'.environment.json'),env)
        if env!=prior:raise ValueError('formal environment differs from M4')
        (smoke if a.stage=='smoke' else run)(r,out,reg)
