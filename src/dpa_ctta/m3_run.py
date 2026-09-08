"""Finite M3 conditioned-proxy experiment; fixed M2 data and objectives."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

from .m1_run import Budget, BudgetExhausted, Observed, make_host as old_host, tensor_save, new_log
from .m3_registration import load_registered
from .m2_registration import inspect_history
from .m3_conditioning import make_host,ConditionedEpisode
from .offline.adaptation_dd import restore_history
from .m2_run import deterministic_smoke_pair
from .offline.adaptation_dd import cpu_tree
from .host_diagnostic import close, snapshot, rng, restore
from .host_diagnostic_run import private_json, append
from .source_io import source_proxy, read_pixels, read_mask
from .source_pilot import seed_all, transform_pixels, evaluate_after_step, validate_arm
from .source_pilot_release import digest, environment, source_unchanged, CONFIG_SHA
from .integrations.ctta_suite import checkout_root, REFERENCE_COMMIT
from .proxy_loss import FixedProxy, ProxyProvenance

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/m3_conditioned_proxy_v1.json'
NEW_ARMS=('R3','D3','O3','O2T')
TRAIN_ARMS=('D3','O3')


def base_method(arm):
    if arm not in TRAIN_ARMS:raise ValueError('only D3/O3 train')
    return arm[0]


def validate_new_arm(records,expected,task,arm):
    if arm not in NEW_ARMS:raise ValueError('M3 new scoring arms only')
    validate_arm(records,expected,task,'B')
    for row in records:
        if row['arm']!=arm or row['task']!=task:raise ValueError('M3 record task/arm mismatch')
        if row['counts']['online_adam']!=1 or row['counts']['memory_pushes']!=1 or row['counts']['condition_transforms']!=1:raise ValueError('M3 step/push mismatch')
        json.dumps(row,allow_nan=False)
    return True


def validate(receipt,stage):
    if receipt['authorization']!='direct_user_M3_CONDITIONED_PROXY':raise ValueError('M3 authorization mismatch')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if commit!=receipt['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('unclean/mismatched execution commit')
    if digest(CONFIG)!=receipt['config_sha256'] or digest(ROOT/'configs/source_pilot_v0.json')!=CONFIG_SHA:raise ValueError('science config drift')
    cfg=json.loads(CONFIG.read_text())
    for path,sha in cfg['inherited_file_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('M1 scientific source changed: '+path)
    checkout_root()
    if receipt['reference_commit']!=REFERENCE_COMMIT:raise ValueError('reference mismatch')
    out=Path(receipt['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_uid!=os.getuid() or out.stat().st_mode&0o077:raise ValueError('private output location/owner/mode')
    if digest(out/'registration.json')!=receipt['registration_sha256']:raise ValueError('overlay drift')
    overlay,registration=load_registered(out)
    if stage!='recompute':
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        devices={int(s.split(',')[0]):s.split(',')[1].strip() for s in lines.splitlines()}
        if receipt['physical_gpu'] not in range(4,8) or devices.get(receipt['physical_gpu'])!=receipt['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=receipt['gpu_uuid']:raise ValueError('GPU scope/UUID mismatch')
    if stage in ('run','recompute'):
        smoke=json.loads((out/'smoke.completion.json').read_text())
        if digest(out/'smoke.completion.json')!=receipt['smoke_sha256']:raise ValueError('smoke digest mismatch')
        if smoke['status']!='M3_SMOKE_PASS' or smoke['updates']!=dict(online=16,outer=4,inner=2):raise ValueError('smoke coverage mismatch')
        if any(smoke[k]!=receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')):raise ValueError('smoke binding mismatch')
    return out,overlay,registration


def sealed_o2(overlay,task,real):
    artifact=torch.load(overlay['o2_artifacts'][task]['path'],map_location='cpu',weights_only=True)
    if artifact['episode']!=600:raise ValueError('sealed O2 step mismatch')
    return FixedProxy(artifact['pixels'],real.mask,real.signed_distance,ProxyProvenance.SOURCE)


def smoke(receipt,out,overlay,registration):
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels,proxy as fixture
    budget=Budget(out);updates=dict(online=0,outer=0,inner=0);evidence={};task=None;current_counts={}
    def step(h,x):
        nonlocal current_counts
        watch=Observed(h);current_counts=watch.counts
        pred=h.step(x);updates['online']+=1
        if not torch.isfinite(pred).all():raise ValueError('nonfinite smoke prediction')
        watch.verify();source_unchanged(h,state)
        if watch.counts['online_adam']!=1 or watch.counts['memory_pushes']!=1:raise ValueError('smoke step/push mismatch')
        counts=dict(watch.counts,condition_transforms=getattr(h,'condition_calls',0));watch.release()
        return pred,counts
    try:
        for task,reg in registration['tasks'].items():
            budget.check();seed_all(20260907)
            state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task)
            history=inspect_history(overlay['tasks'][task]['history']['path'],task)[16]
            x=pixels(task);y=fixture(task).mask;task_ev=evidence[task]={};torch.cuda.reset_peak_memory_stats()
            with deterministic_smoke_pair():
                a=old_host(task,'R',state,real);b=make_host(task,'R3',state,real,strength=0)
                initial=rng();pa,ca=step(a,x);after=rng();restore(initial);pb,cb=step(b,x)
                close(pa,pb);close(snapshot(a),snapshot(b));close(after,rng(),exact=True)
                for ga,gb in zip(a.model.parameters(),b.model.parameters()):close(ga.grad,gb.grad)
                if b.condition_calls:raise ValueError('rho0 constructed T')
                task_ev['rho0']=dict(max_logit_difference=float((pa-pb).abs().max()),counts=[ca,cb])
                del a,b,pa,pb;gc.collect()
            proxies={'R3':real,'O2T':sealed_o2(overlay,task,real)};captured=None
            for arm in TRAIN_ARMS:
                seed_all(20260907);ep=ConditionedEpisode(task,state,'cuda:0');current_counts=ep.counts
                S=torch.nn.Parameter(real.pixel_rgb.to('cuda:0').clone());opt=torch.optim.Adam([S],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
                # O3's one functional result is shared with the paired actual-Adam check below.
                # This pair is evaluated under the same strict backend; no second inner call.
                from contextlib import nullcontext
                with deterministic_smoke_pair() if arm=='O3' else nullcontext():
                    loss=ep.objective(base_method(arm),S,real.mask,x,y,history,capture_inner=arm=='O3')
                    if arm=='O3':updates['inner']+=1;captured=ep.inner_result
                    g,=torch.autograd.grad(loss,S)
                if not torch.isfinite(g).all() or not g.norm()>0:raise ValueError('nonfinite/broken fixture image meta-gradient')
                before=S.detach().clone();S.grad=g;opt.step();updates['outer']+=1
                with torch.no_grad():S.clamp_(0,1)
                if not torch.isfinite(S).all() or torch.equal(S,before):raise ValueError('invalid fixture proxy update')
                task_ev[arm+'_outer']=dict(loss=float(loss.detach()),gradient_norm=float(g.norm()),image_delta_norm=float((S-before).norm()),condition_stats=ep.condition_stats,counts=dict(ep.counts))
                proxies[arm]=FixedProxy(S.detach().cpu(),real.mask,real.signed_distance,ProxyProvenance.SOURCE)
                source_unchanged(ep.host,state)
                if any(not torch.equal(v.cpu(),state[n]) for n,v in ep.clone.state_dict().items()):raise ValueError('offline clone changed')
                ep.clear_graphs();del ep,S,opt,loss,g,before;gc.collect()
            for arm in NEW_ARMS:
                h=make_host(task,arm,state,proxies[arm]);original=h.proxy.pixel_rgb.detach().clone();mask=h.proxy.mask.detach().clone()
                pred,counts=step(h,x)
                if h.condition_calls!=1 or not torch.equal(h.proxy.pixel_rgb,original) or not torch.equal(h.proxy.mask,mask):raise ValueError('condition lifecycle changed proxy body/mask')
                task_ev[arm+'_online']=dict(counts=counts,condition_stats=h.condition_stats)
                del h,pred,original,mask;gc.collect()
            with deterministic_smoke_pair():
                # Both hosts use original Real S: captured inner precedes O3 outer update.
                a=make_host(task,'O3',state,real);b=make_host(task,'O3',state,real)
                for h in (a,b):
                    h._initial_hook.remove();h._started=True;restore_history(h,history)
                initial=rng();pa,ca=step(a,x);after=rng();restore(initial);pb,cb=step(b,x)
                # Different evaluator labels must leave predictions and state untouched.
                snap_a=snapshot(a);snap_b=snapshot(b)
                evaluate_after_step(pa,y,task);evaluate_after_step(pb,1-y,task)
                close(snap_a,snapshot(a),exact=True);close(snap_b,snapshot(b),exact=True)
                close(pa,pb);close(snapshot(a),snapshot(b));close(after,rng(),exact=True)
                for h in (a,b):
                    close(captured['prompt'],h.prompt.data_prompt.detach().cpu())
                    adam=h.optimizer.state[h.prompt.data_prompt]
                    for k in ('exp_avg','exp_avg_sq'):close(captured['adam'][k],adam[k].detach().cpu())
                    if int(adam['step'])!=captured['adam']['step'] or {m.sample_num for m in h.model.modules() if isinstance(m,h.adabn)}!={17}:raise ValueError('inner Adam/counter mismatch')
                task_ev['functional_actual']=dict(max_prompt_difference=float((captured['prompt']-a.prompt.data_prompt.detach().cpu()).abs().max()),adam_step=captured['adam']['step'],counts=[ca,cb],functional_results_reused=1,labels_isolated=True)
                del a,b,pa,pb,snap_a,snap_b;gc.collect()
            task_ev['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            del state,real,history,proxies,captured,x,y;gc.collect()
            print(json.dumps(dict(stage='smoke',task=task,status='PASS',updates=updates)),flush=True)
        if updates!=dict(online=16,outer=4,inner=2):raise ValueError('M3 smoke budget mismatch')
        private_json(out/'smoke.completion.json',dict(status='M3_SMOKE_PASS',updates=updates,evidence=evidence,
            **{k:receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')},gpu_seconds=budget.seconds(),exit_code=0,
            paired_comparison_backend=dict(deterministic_algorithms=True,scope='rho0 pair and shared O3 functional/actual pair only',restored_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))))
    except Exception as error:
        private_json(out/'smoke.failure.json',dict(status='M3_PARTIAL',task=task,updates=updates,current_counts=current_counts,evidence=evidence,reason=str(error),exception=type(error).__name__,gpu_seconds=budget.seconds(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),exit_code=1))
        raise


def train_all(out,registration,budget,progress,overlay):
    from .offline.adaptation_dd import cpu_tree
    for task,reg in registration['tasks'].items():
        progress.update(stage='training',task=task,method=None,episode=0,current_counts=None)
        state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True)
        real=source_proxy(reg['proxy'],task)
        library=inspect_history(overlay['tasks'][task]['history']['path'],task)
        for method in TRAIN_ARMS:
            progress.update(stage='training',method=method,episode=0)
            seed_all(20260907)
            ep=ConditionedEpisode(task,state,'cuda:0');S=torch.nn.Parameter(real.pixel_rgb.to('cuda:0').clone())
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
                    loss=ep.objective(base_method(method),S,real.mask,x,y,library[episode['state_index']])
                    g,=torch.autograd.grad(loss,S)
                    if not torch.isfinite(g).all(): raise ValueError('nonfinite image meta-gradient')
                    S.grad=g;optimizer.step();progress['outer']+=1
                    if method=='O3': progress['inner']+=1
                    with torch.no_grad(): S.clamp_(0,1)
                    if not torch.isfinite(S).all(): raise ValueError('nonfinite trained proxy')
                    progress['episode']=episode['episode']
                    append(f,dict(**episode,loss=float(loss.detach()),gradient_norm=float(g.norm()),image_delta_norm=float((S-before).norm()),
                        image_adam_step=int(optimizer.state[S]['step']),condition_stats=dict(ep.condition_stats),counts=dict(ep.counts),elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    ep.clear_graphs();del loss,g,before,x,y
                    if episode['episode'] in (100,300,600): save(episode['episode'])
                    if episode['episode']%25==0: print(json.dumps(dict(stage='training',task=task,method=method,episode=episode['episode'],gpu_seconds=budget.seconds())),flush=True)
            source_unchanged(ep.host,state)
            for name,v in ep.clone.state_dict().items():
                if not torch.equal(v.cpu(),state[name]): raise ValueError('offline clone state changed')
            artifact=out/f'{task}_{method}_600.pt'
            private_json(out/f'train_{task}_{method}.completion.json',dict(status='COMPLETE',episodes=600,episodes_sha256=overlay['tasks'][task]['episodes_sha256'],final_sha256=digest(artifact),source_unchanged=True,
                elapsed_seconds=time.monotonic()-started,counts=ep.counts,peak_allocated_bytes=torch.cuda.max_memory_allocated(),artifact_bytes=artifact.stat().st_size))
            del ep,S,optimizer;gc.collect()
        del library,state,real;gc.collect()
    # Freeze both tasks and both methods before any source/target scoring.
    finals={t:{m:json.loads((out/f'train_{t}_{m}.completion.json').read_text())['final_sha256'] for m in TRAIN_ARMS} for t in registration['tasks']}
    private_json(out/'final_artifacts.frozen.json',finals)

def evaluate_all(out,registration,budget,progress,overlay):
    for task,reg in registration['tasks'].items():
        state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task)
        proxies={'R3':real,'O2T':sealed_o2(overlay,task,real)}
        for method in TRAIN_ARMS:
            path=out/f'{task}_{method}_600.pt'
            if digest(path)!=json.loads((out/'final_artifacts.frozen.json').read_text())[task][method]: raise ValueError('final proxy drift')
            artifact=torch.load(path,map_location='cpu',weights_only=True)
            if artifact['episode']!=600: raise ValueError('only step600 can be evaluated')
            proxies[method]=FixedProxy(artifact['pixels'],real.mask,real.signed_distance,ProxyProvenance.SOURCE)
        for stage in ('source','target'):
            selected=reg['source_query'] if stage=='source' else reg['target']
            if not selected: continue
            for arm in NEW_ARMS:
                progress.update(stage=stage,task=task,method=arm,visit=0)
                h=make_host(task,arm,state,proxies[arm]);watch=Observed(h);torch.cuda.reset_peak_memory_stats()
                progress['current_counts']=watch.counts
                watch.counts['condition_transforms']=0
                with new_log(out/f'{stage}_{task}_{arm}.jsonl') as f:
                    for i,row in enumerate(selected):
                        budget.check();progress['visit']=i+1;torch.cuda.synchronize();start=time.monotonic()
                        image=read_pixels(row['image_path'],task,row['image_size'])
                        p0=h.prompt.data_prompt.detach().clone() if hasattr(h,'prompt') else None
                        previous=dict(watch.counts);torch.cuda.synchronize();step_start=time.monotonic()
                        pred=h.step(image)
                        watch.counts['condition_transforms']=h.condition_calls
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
                            memory_size=h.memory_bank.get_size() if p0 is not None else 0,host_loss=watch.loss,condition_stats=dict(h.condition_stats),
                            proxy_loss=float(h.last_proxy_loss.region.detach()) if getattr(h,'last_proxy_loss',None) is not None else None,
                            source_versions_unchanged=True,optimizer_state_finite=True,pipeline_elapsed_seconds=time.monotonic()-start,
                            host_step_elapsed_seconds=step_time,peak_allocated_bytes=torch.cuda.max_memory_allocated(),counts=delta)
                        if p0 is not None and any(not torch.isfinite(v).all() for v in h.optimizer.state[h.prompt.data_prompt].values() if isinstance(v,torch.Tensor)): raise ValueError('Adam state nonfinite')
                        validate_new_arm([record],[wanted],task,arm)
                        if delta['memory_pushes']!=int(arm!='N') or delta['backward_calls']!=int(arm!='N'): raise ValueError('lifecycle observation mismatch')
                        append(f,record);progress['records']+=1
                        del pred,image,p0
                source_unchanged(h,state)
                private_json(out/f'{stage}_{task}_{arm}.completion.json',dict(status='COMPLETE',records=len(selected),source_unchanged=True,counts=watch.counts))
                watch.release();del h,watch;gc.collect()
                print(json.dumps(dict(stage=stage,task=task,arm=arm,records=len(selected))),flush=True)
        del state,real,proxies;gc.collect()


def run(receipt,out,overlay,registration):
    smoke=json.loads((out/'smoke.completion.json').read_text());budget=Budget(out,smoke['gpu_seconds'])
    progress=dict(stage='initialization',online=0,outer=0,inner=0,records=0)
    private_json(out/'run.start.json',dict(commit=receipt['commit'],started_utc=time.time()))
    try:
        train_all(out,registration,budget,progress,overlay);evaluate_all(out,registration,budget,progress,overlay)
        expected=sum(4*(len(r['source_query'])+len(r['target'])) for r in registration['tasks'].values())
        if (progress['online'],progress['outer'],progress['inner'],progress['records'])!=(expected,2400,1200,expected):raise ValueError('M3 actual budget/coverage mismatch')
        private_json(out/'run.completion.json',dict(status='M3_RUN_COMPLETE',progress=progress,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as error:
        private_json(out/'run.failure.json',dict(status='BUDGET_EXHAUSTED' if isinstance(error,BudgetExhausted) else 'M3_PARTIAL',progress=progress,exception=type(error).__name__,reason=str(error),gpu_seconds=budget.seconds(),exit_code=1))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args(argv)
    os.umask(0o077)
    if a.stage=='smoke':
        if torch.cuda.is_initialized():raise ValueError('smoke workspace must be configured before CUDA initialization')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    receipt=json.loads(a.receipt.read_text());out,overlay,registration=validate(receipt,a.stage)
    if a.stage=='recompute':
        from .m3_analysis import recompute
        recompute(out,overlay,registration,receipt)
        if torch.cuda.is_initialized():raise ValueError('CPU recompute initialized CUDA')
    else:
        env=environment();private_json(out/(a.stage+'.environment.json'),env)
        prior=json.loads((Path(overlay['old_directory'])/'run.environment.json').read_text())
        differences={k:{'M1':prior.get(k),'M3':v} for k,v in env.items() if prior.get(k)!=v}
        private_json(out/(a.stage+'.environment_comparison.json'),dict(differences=differences,compatible=not differences))
        (smoke if a.stage=='smoke' else run)(receipt,out,overlay,registration)
    return 0


if __name__=='__main__':raise SystemExit(main())
