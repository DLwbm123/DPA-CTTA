"""Finite M1 execution: single smoke, shared history, four DD fits, frozen five-arm scoring."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from .host_diagnostic import close, snapshot, rng, restore
from .host_diagnostic_run import private_json, append
from .integrations.ctta_suite import checkout_root, REFERENCE_COMMIT
from .offline.adaptation_dd import OfflineEpisode, history_state
from .proxy_loss import FixedProxy, ProxyProvenance
from .source_io import source_proxy, read_pixels, read_mask
from .source_pilot import assemble, model_input_from_pixels, evaluate_after_step, validate_arm, transform_pixels
from .source_pilot_release import digest, check_registered_files, environment, source_unchanged, CONFIG_SHA

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/m1_adaptation_dd_validation_v1.json'
ARMS=('N','A','R','D','O')


class BudgetExhausted(RuntimeError): pass


class Budget:
    def __init__(self,out,previous_seconds=0):
        self.out=out;self.start=time.monotonic();self.previous=previous_seconds
    def check(self):
        if self.seconds()>=21600: raise BudgetExhausted('GPU active-stage wall budget 6 hours')
        if sum(p.stat().st_size for p in self.out.iterdir() if p.is_file())>=2*1024**3: raise BudgetExhausted('private output 2 GiB budget')
    def seconds(self): return self.previous+time.monotonic()-self.start


def tensor_save(path,value):
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'wb') as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())


def new_log(path): return os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600),'w')


def make_host(task,arm,state,proxy=None,device='cuda:0'):
    return assemble(task,'B' if arm in ('R','D','O') else arm,state,proxy,device=device)


class Observed:
    """Per-host scalar hooks; original forwards, Adam, source .grad and memory retained."""
    def __init__(self,h):
        self.h=h;self.counts=dict(model_forwards=0,prompt_forwards=0,proxy_forwards=0,proxy_images=0,
            online_adam=0,backward_calls=0,memory_pushes=0,retrievals=0)
        self.handles=[];self.update_norm=0.;self.loss=None;self.init=None
        self.versions={n:(v,v._version) for n,v in h.model.state_dict(keep_vars=True).items()}
        def model(m,args): self.counts['model_forwards']+=1
        self.handles.append(h.model.register_forward_pre_hook(model))
        if hasattr(h,'prompt'):
            def prompt(m,args): self.counts['prompt_forwards']+=1
            def grad(g): self.counts['backward_calls']+=1;return g
            self.handles.extend([h.prompt.register_forward_pre_hook(prompt),h.prompt.data_prompt.register_hook(grad)])
            def before(opt,args,kw):
                self.init=h.prompt.data_prompt.detach().clone()
                scope=h.model.resnet if h.task=='polyp' else h.model
                losses=[m.bn_loss for m in scope.modules() if isinstance(m,h.adabn)]
                self.loss=float((sum(losses)/len(losses)).detach())
            def after(opt,args,kw):
                self.counts['online_adam']+=1
                self.update_norm=float((h.prompt.data_prompt.detach()-self.init).norm())
            self.handles.extend([h.optimizer.register_step_pre_hook(before),h.optimizer.register_step_post_hook(after)])
            original=h.memory_bank.push
            def push(*a,**kw):
                value=original(*a,**kw);self.counts['memory_pushes']+=1;return value
            h.memory_bank.push=push
            original_get=h.memory_bank.get_neighbours
            def get(*a,**kw):
                value=original_get(*a,**kw);self.counts['retrievals']+=1;return value
            h.memory_bank.get_neighbours=get
        if hasattr(h,'_proxy_model'):
            def proxy(m,args):
                if len(args[0])!=4: raise ValueError('K=4 full batch mismatch')
                self.counts['proxy_forwards']+=1;self.counts['proxy_images']+=4
            self.handles.append(h._proxy_model.register_forward_pre_hook(proxy))
    def verify(self):
        current=self.h.model.state_dict(keep_vars=True)
        if current.keys()!=self.versions.keys() or any(current[n] is not t or t._version!=v for n,(t,v) in self.versions.items()):
            raise ValueError('source state mutation')
    def release(self):
        for handle in self.handles: handle.remove()


def validate(receipt,stage):
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if receipt['authorization']!='direct_user_M1_ADAPTATION_ORIENTED_DD_VALIDATION': raise ValueError('wrong M1 authorization')
    if receipt['commit']!=commit or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip(): raise ValueError('dirty or mismatched execution commit')
    if digest(CONFIG)!=receipt['config_sha256'] or digest(ROOT/'configs/source_pilot_v0.json')!=CONFIG_SHA: raise ValueError('config mismatch')
    checkout_root()
    if receipt['reference_commit']!=REFERENCE_COMMIT: raise ValueError('dependency mismatch')
    out=Path(receipt['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_uid!=os.getuid() or out.stat().st_mode&0o077: raise ValueError('private output ownership/location/mode')
    if digest(out/'registration.json')!=receipt['registration_sha256']: raise ValueError('registration drift')
    registration=json.loads((out/'registration.json').read_text());check_registered_files(registration)
    if stage!='recompute':
        devices=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        mapping={int(s.split(',')[0]):s.split(',')[1].strip() for s in devices.splitlines()}
        if receipt['physical_gpu'] not in range(4,8) or mapping.get(receipt['physical_gpu'])!=receipt['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=receipt['gpu_uuid']: raise ValueError('GPU scope/UUID mismatch')
    if stage in ('run','recompute'):
        smoke=json.loads((out/'smoke.completion.json').read_text())
        if digest(out/'smoke.completion.json')!=receipt['smoke_sha256']: raise ValueError('smoke digest mismatch')
        if any(smoke[k]!=receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')) or smoke['status']!='M1_SMOKE_PASS': raise ValueError('smoke identity mismatch')
        if smoke['updates']!={'online':74,'outer':4,'inner':2}: raise ValueError('smoke coverage mismatch')
    return out,registration


def smoke(receipt,out,registration):
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels, proxy as fixture
    budget=Budget(out);updates=dict(online=0,outer=0,inner=0);evidence={};task=None
    try:
        for task,reg in registration['tasks'].items():
            state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True)
            real=source_proxy(reg['proxy'],task);torch.cuda.reset_peak_memory_stats()
            ref=make_host(task,'A',state);base=make_host(task,'A',state)
            observations=[Observed(ref),Observed(base)];diffs=[]
            for i in range(17):
                budget.check();x=pixels(task,i);before=rng()
                a=ref.native_step(ref,model_input_from_pixels(x,task).to('cuda:0'));updates['online']+=1;after=rng()
                restore(before);b=base.step(x);updates['online']+=1
                close(after,rng(),exact=True);close(a,b)
                # Callable observation wrappers are intentionally excluded by snapshot().
                close(snapshot(ref),snapshot(base))
                for pa,pb in zip(ref.model.parameters(),base.model.parameters()): close(pa.grad,pb.grad)
                diffs.append(float((a-b).abs().max()))
            for h,o in zip((ref,base),observations):
                source_unchanged(h,state);o.verify()
                if o.counts['online_adam']!=17 or o.counts['memory_pushes']!=17 or o.counts['retrievals']!=1: raise ValueError('Base smoke lifecycle')
                o.release()
            task_ev={'base_max_error':max(diffs),'base_counts':[o.counts for o in observations]}
            del ref,base,observations,a,b;gc.collect()
            reference_prediction=None;reference_snapshot=None
            for arm in ('R','D','O'):
                h=make_host(task,arm,state,real);watch=Observed(h);pred=h.step(pixels(task));updates['online']+=1
                snap=snapshot(h)
                if reference_prediction is None: reference_prediction=pred.detach().cpu();reference_snapshot=snap
                else: close(reference_prediction,pred.detach().cpu());close(reference_snapshot,snap)
                source_unchanged(h,state);watch.verify();task_ev[arm+'_counts']=dict(watch.counts);watch.release()
                del h,watch,pred,snap;gc.collect()
            del reference_snapshot,reference_prediction
            n=make_host(task,'N',state);nwatch=Observed(n);pred=n.step(pixels(task))
            if not torch.isfinite(pred).all(): raise ValueError('N smoke nonfinite')
            source_unchanged(n,state);task_ev['N_counts']=dict(nwatch.counts);nwatch.release();del n,nwatch,pred;gc.collect()
            ep=OfflineEpisode(task,state,'cuda:0');empty=history_state(ep.host)
            procedural=torch.cat([pixels(task,i+3) for i in range(4)]).to('cuda:0')
            masks=fixture(task).mask.expand(4,-1,-1,-1).clone()
            for method in ('D','O'):
                budget.check();S=torch.nn.Parameter(procedural.clone());opt=torch.optim.Adam([S],lr=.01)
                loss=ep.objective(method,S,masks,pixels(task),masks[:1],empty)
                gradient,=torch.autograd.grad(loss,S)
                if not torch.isfinite(gradient).all() or not gradient.norm()>0: raise ValueError('broken/nonfinite image meta-gradient')
                before=S.detach().clone();S.grad=gradient;opt.step();updates['outer']+=1
                if method=='O': updates['inner']+=1
                with torch.no_grad(): S.clamp_(0,1)
                if torch.equal(before,S): raise ValueError('image optimizer made no change')
                task_ev[method+'_meta']=dict(loss=float(loss.detach()),image_gradient_norm=float(gradient.norm()),image_update_norm=float((S-before).norm()))
                ep.clear_graphs();del S,opt,loss,gradient,before;gc.collect()
            source_unchanged(ep.host,state)
            if any(p.grad is not None for p in ep.host.model.parameters()): raise ValueError('offline source gradients populated')
            task_ev['offline_counts']=dict(ep.counts);task_ev['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            evidence[task]=task_ev;del ep,state,real,procedural,masks;gc.collect()
            print(json.dumps(dict(stage='smoke',task=task,status='PASS',updates=updates)),flush=True)
        if updates!=dict(online=74,outer=4,inner=2): raise ValueError('smoke budget mismatch')
        private_json(out/'smoke.completion.json',dict(status='M1_SMOKE_PASS',updates=updates,evidence=evidence,
            **{k:receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')},gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as error:
        private_json(out/'smoke.failure.json',dict(status='M1_PARTIAL',task=task,updates=updates,evidence=evidence,reason=str(error),exception=type(error).__name__,exit_code=1,gpu_seconds=budget.seconds()))
        raise


def train_all(out,registration,budget,progress):
    from .offline.adaptation_dd import cpu_tree
    for task,reg in registration['tasks'].items():
        progress.update(stage='history',task=task,method=None,episode=0,current_counts=None)
        state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True)
        real=source_proxy(reg['proxy'],task);h=make_host(task,'A',state);watch=Observed(h);library=[]
        progress['current_counts']=watch.counts
        with new_log(out/f'history_{task}.jsonl') as f:
            for i,row in enumerate(reg['history']):
                budget.check();library.append(history_state(h))
                prediction=h.step(read_pixels(row['image_path'],task,row['image_size']))
                progress['online']+=1;watch.verify()
                if not torch.isfinite(prediction).all(): raise ValueError('history prediction nonfinite')
                append(f,dict(visit=i+1,group_id=row['group_id'],counts=dict(watch.counts)))
        source_unchanged(h,state)
        if watch.counts['online_adam']!=32 or watch.counts['memory_pushes']!=32: raise ValueError('history count mismatch')
        tensor_save(out/f'history_{task}.pt',library)
        private_json(out/f'history_{task}.completion.json',dict(status='COMPLETE',counts=watch.counts,source_unchanged=True))
        watch.release();del h,watch,prediction;gc.collect()
        for method in ('D','O'):
            progress.update(stage='training',method=method,episode=0)
            ep=OfflineEpisode(task,state,'cuda:0');S=torch.nn.Parameter(real.pixel_rgb.to('cuda:0').clone())
            progress['current_counts']=ep.counts
            optimizer=torch.optim.Adam([S],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
            def save(index):
                tensor_save(out/f'{task}_{method}_{index}.pt',dict(pixels=S.detach().cpu(),optimizer=cpu_tree(optimizer.state_dict()),episode=progress['episode'],seed=20260907))
            save(0);torch.cuda.reset_peak_memory_stats();started=time.monotonic()
            with new_log(out/f'train_{task}_{method}.jsonl') as f:
                for episode in reg['episodes']:
                    try: budget.check()
                    except BudgetExhausted:
                        save('budget_partial');raise
                    row=reg['train_query'][episode['query_index']]
                    x=transform_pixels(read_pixels(row['image_path'],task,row['image_size']),episode['transform'])
                    y=read_mask(row['mask_path'],task,row['image_size'])
                    before=S.detach().clone();optimizer.zero_grad(set_to_none=True)
                    loss=ep.objective(method,S,real.mask,x,y,library[episode['state_index']])
                    g,=torch.autograd.grad(loss,S)
                    if not torch.isfinite(g).all(): raise ValueError('nonfinite image meta-gradient')
                    S.grad=g;optimizer.step();progress['outer']+=1
                    if method=='O': progress['inner']+=1
                    with torch.no_grad(): S.clamp_(0,1)
                    if not torch.isfinite(S).all(): raise ValueError('nonfinite trained proxy')
                    progress['episode']=episode['episode']
                    append(f,dict(**episode,loss=float(loss.detach()),gradient_norm=float(g.norm()),image_delta_norm=float((S-before).norm()),
                        image_adam_step=int(optimizer.state[S]['step']),counts=dict(ep.counts),elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    ep.clear_graphs();del loss,g,before,x,y
                    if episode['episode'] in (100,300,600): save(episode['episode'])
                    if episode['episode']%25==0: print(json.dumps(dict(stage='training',task=task,method=method,episode=episode['episode'],gpu_seconds=budget.seconds())),flush=True)
            source_unchanged(ep.host,state)
            for name,v in ep.clone.state_dict().items():
                if not torch.equal(v.cpu(),state[name]): raise ValueError('offline clone state changed')
            artifact=out/f'{task}_{method}_600.pt'
            private_json(out/f'train_{task}_{method}.completion.json',dict(status='COMPLETE',episodes=600,final_sha256=digest(artifact),source_unchanged=True,
                elapsed_seconds=time.monotonic()-started,counts=ep.counts,peak_allocated_bytes=torch.cuda.max_memory_allocated(),artifact_bytes=artifact.stat().st_size))
            del ep,S,optimizer;gc.collect()
        del library,state,real;gc.collect()
    # Freeze both tasks and both methods before any source/target scoring.
    finals={t:{m:json.loads((out/f'train_{t}_{m}.completion.json').read_text())['final_sha256'] for m in ('D','O')} for t in registration['tasks']}
    private_json(out/'final_artifacts.frozen.json',finals)


def evaluate_all(out,registration,budget,progress):
    for task,reg in registration['tasks'].items():
        state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task)
        proxies={'R':real}
        for method in ('D','O'):
            path=out/f'{task}_{method}_600.pt'
            if digest(path)!=json.loads((out/'final_artifacts.frozen.json').read_text())[task][method]: raise ValueError('final proxy drift')
            artifact=torch.load(path,map_location='cpu',weights_only=True)
            if artifact['episode']!=600: raise ValueError('only step600 can be evaluated')
            proxies[method]=FixedProxy(artifact['pixels'],real.mask,real.signed_distance,ProxyProvenance.SOURCE)
        for stage in ('source','target'):
            selected=reg['source_query'] if stage=='source' else reg['target']
            if not selected: continue
            for arm in ARMS:
                progress.update(stage=stage,task=task,method=arm,visit=0)
                h=make_host(task,arm,state,proxies.get(arm));watch=Observed(h);torch.cuda.reset_peak_memory_stats()
                progress['current_counts']=watch.counts
                with new_log(out/f'{stage}_{task}_{arm}.jsonl') as f:
                    for i,row in enumerate(selected):
                        budget.check();progress['visit']=i+1;torch.cuda.synchronize();start=time.monotonic()
                        image=read_pixels(row['image_path'],task,row['image_size'])
                        p0=h.prompt.data_prompt.detach().clone() if hasattr(h,'prompt') else None
                        previous=dict(watch.counts);torch.cuda.synchronize();step_start=time.monotonic()
                        pred=h.step(image)
                        if arm!='N': progress['online']+=1
                        torch.cuda.synchronize();step_time=time.monotonic()-step_start
                        if not torch.isfinite(pred).all(): raise ValueError('nonfinite final prediction')
                        watch.verify()
                        # First semantic target-label access follows final fixed prediction.
                        metrics=evaluate_after_step(pred,read_mask(row['mask_path'],task,row['image_size']),task)
                        counters=sorted({m.sample_num for m in h.model.modules() if isinstance(m,h.adabn)}) if p0 is not None else []
                        delta={k:v-previous[k] for k,v in watch.counts.items()}
                        wanted=dict(visit=i+1,sample_id=row['sample_id'],group_id=row['group_id'],segment='clean')
                        record=dict(wanted,domain=row['domain'],task=task,arm=arm,metrics=metrics,
                            prompt_state_delta_norm=float((h.prompt.data_prompt-p0).norm()) if p0 is not None else 0.,
                            prompt_initialization_delta_norm=float((watch.init-p0).norm()) if p0 is not None else 0.,
                            optimizer_update_norm=watch.update_norm,optimizer_steps_this_visit=delta['online_adam'],
                            native_counts=counters,adam_step=int(h.optimizer.state[h.prompt.data_prompt]['step']) if p0 is not None else 0,
                            memory_size=h.memory_bank.get_size() if p0 is not None else 0,host_loss=watch.loss,
                            proxy_loss=float(h.last_proxy_loss.region.detach()) if getattr(h,'last_proxy_loss',None) is not None else None,
                            source_versions_unchanged=True,optimizer_state_finite=True,pipeline_elapsed_seconds=time.monotonic()-start,
                            host_step_elapsed_seconds=step_time,peak_allocated_bytes=torch.cuda.max_memory_allocated(),counts=delta)
                        if p0 is not None and any(not torch.isfinite(v).all() for v in h.optimizer.state[h.prompt.data_prompt].values() if isinstance(v,torch.Tensor)): raise ValueError('Adam state nonfinite')
                        validate_arm([record],[wanted],task,'B' if arm in ('R','D','O') else arm)
                        if delta['memory_pushes']!=int(arm!='N') or delta['backward_calls']!=int(arm!='N'): raise ValueError('lifecycle observation mismatch')
                        append(f,record);progress['records']+=1
                        del pred,image,p0
                source_unchanged(h,state)
                private_json(out/f'{stage}_{task}_{arm}.completion.json',dict(status='COMPLETE',records=len(selected),source_unchanged=True,counts=watch.counts))
                watch.release();del h,watch;gc.collect()
                print(json.dumps(dict(stage=stage,task=task,arm=arm,records=len(selected))),flush=True)
        del state,real,proxies;gc.collect()


def run(receipt,out,registration):
    sm=json.loads((out/'smoke.completion.json').read_text());budget=Budget(out,sm['gpu_seconds'])
    progress=dict(stage='initialization',online=0,outer=0,inner=0,records=0)
    private_json(out/'run.start.json',dict(commit=receipt['commit'],started_utc=time.time()))
    try:
        train_all(out,registration,budget,progress);evaluate_all(out,registration,budget,progress)
        q=sum(len(r['target']) for r in registration['tasks'].values())
        if (progress['online'],progress['outer'],progress['inner'],progress['records'])!=(64+208+4*q,2400,1200,260+5*q): raise ValueError('run budget/coverage mismatch')
        private_json(out/'run.completion.json',dict(status='M1_RUN_COMPLETE',progress=progress,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as error:
        private_json(out/'run.failure.json',dict(status='BUDGET_EXHAUSTED' if isinstance(error,BudgetExhausted) else 'M1_PARTIAL',progress=progress,exception=type(error).__name__,reason=str(error),gpu_seconds=budget.seconds(),exit_code=1))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args(argv)
    os.umask(0o077);receipt=json.loads(a.receipt.read_text())
    out,registration=validate(receipt,a.stage)
    if a.stage=='recompute':
        from .m1_analysis import recompute
        recompute(out,registration,receipt)
        if torch.cuda.is_initialized(): raise ValueError('CPU recompute initialized CUDA')
    else:
        private_json(out/(a.stage+'.environment.json'),environment())
        (smoke if a.stage=='smoke' else run)(receipt,out,registration)
    return 0


if __name__=='__main__': raise SystemExit(main())
