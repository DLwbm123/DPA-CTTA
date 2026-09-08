"""Finite M2 orchestration. M1 scientific primitives and online hosts are unchanged."""
import argparse
from contextlib import contextmanager
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch

from .m1_run import Budget, BudgetExhausted, Observed, make_host, tensor_save, new_log
from .m2_registration import load_registered, inspect_history
from .offline.adaptation_dd import OfflineEpisode, cpu_tree
from .host_diagnostic import close, snapshot, rng, restore
from .host_diagnostic_run import private_json, append
from .source_io import source_proxy, read_pixels, read_mask
from .source_pilot import seed_all, transform_pixels, evaluate_after_step, validate_arm
from .source_pilot_release import digest, environment, source_unchanged, CONFIG_SHA
from .integrations.ctta_suite import checkout_root, REFERENCE_COMMIT
from .proxy_loss import FixedProxy, ProxyProvenance

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/m2_episode_coverage_v1.json'
NEW_ARMS=('D2','O2')


def base_method(arm):
    if arm not in NEW_ARMS:raise ValueError('M2 runs only D2/O2')
    return arm[0]


def m2_host(task,arm,state,proxy,device='cuda:0'):
    return make_host(task,base_method(arm),state,proxy,device)


@contextmanager
def deterministic_smoke_pair():
    """Control CUDA reduction order only for the two-host equivalence check.

    cuDNN deterministic alone does not cover native CUDA interpolation backward.
    Restore both flags even on failure; formal M1-compatible execution is unchanged.
    """
    enabled=torch.are_deterministic_algorithms_enabled()
    warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True,warn_only=False)
        yield
    finally:
        torch.use_deterministic_algorithms(enabled,warn_only=warn_only)


def validate_new_arm(records,expected,task,arm):
    base_method(arm)
    validate_arm(records,expected,task,'B')
    for row in records:
        if row['arm']!=arm or row['task']!=task:raise ValueError('M2 record task/arm mismatch')
        if row['counts']['online_adam']!=1 or row['counts']['memory_pushes']!=1:raise ValueError('M2 step/push mismatch')
        json.dumps(row,allow_nan=False)
    return True


def validate(receipt,stage):
    if receipt['authorization']!='direct_user_M2_EPISODE_COVERAGE_COMPARISON':raise ValueError('M2 authorization mismatch')
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
        if smoke['status']!='M2_SMOKE_PASS' or smoke['updates']!=dict(online=8,outer=4,inner=2):raise ValueError('smoke coverage mismatch')
        if any(smoke[k]!=receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')):raise ValueError('smoke binding mismatch')
    return out,overlay,registration


def smoke(receipt,out,overlay,registration):
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels,proxy as fixture
    budget=Budget(out);updates=dict(online=0,outer=0,inner=0);evidence={};task=None;arm=None;current_counts={}
    try:
        for task,reg in registration['tasks'].items():
            state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task)
            library=inspect_history(overlay['tasks'][task]['history']['path'],task)
            evidence[task]={};torch.cuda.reset_peak_memory_stats()
            for arm in NEW_ARMS:
                budget.check();seed_all(20260907)
                ep=OfflineEpisode(task,state,'cuda:0');current_counts=ep.counts
                S=torch.nn.Parameter(real.pixel_rgb.to('cuda:0').clone())
                opt=torch.optim.Adam([S],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
                loss=ep.objective(base_method(arm),S,real.mask,pixels(task),fixture(task).mask,library[16])
                g,=torch.autograd.grad(loss,S)
                if not torch.isfinite(g).all() or not g.norm()>0:raise ValueError('nonfinite or broken smoke image gradient')
                before=S.detach().clone();S.grad=g;opt.step();updates['outer']+=1
                if arm=='O2':updates['inner']+=1
                with torch.no_grad():S.clamp_(0,1)
                if not torch.isfinite(S).all() or torch.equal(S,before):raise ValueError('smoke pixels unchanged/nonfinite')
                item=dict(loss=float(loss.detach()),image_gradient_norm=float(g.norm()),pixel_update_norm=float((S-before).norm()),offline_counts=dict(ep.counts),status='OUTER_PASS_ONLINE_PENDING')
                evidence[task][arm]=item
                payload=FixedProxy(S.detach().cpu(),real.mask,real.signed_distance,ProxyProvenance.SOURCE)
                source_unchanged(ep.host,state)
                for n,v in ep.clone.state_dict().items():
                    if not torch.equal(v.cpu(),state[n]):raise ValueError('offline clone changed')
                ep.clear_graphs();del ep,S,opt,loss,g,before;gc.collect()
                with deterministic_smoke_pair():
                    direct=make_host(task,base_method(arm),state,payload);wrapped=m2_host(task,arm,state,payload)
                    wa,wb=Observed(direct),Observed(wrapped);current_counts={'reference':wa.counts,'new':wb.counts}
                    initial=rng();a=direct.step(pixels(task));updates['online']+=1;after=rng()
                    restore(initial);b=wrapped.step(pixels(task));updates['online']+=1
                    close(a,b);close(after,rng(),exact=True);close(snapshot(direct),snapshot(wrapped))
                    # Counters and memory keys/values are discrete identity/state observations.
                    close(history_state_discrete(direct),history_state_discrete(wrapped),exact=True)
                    for pa,pb in zip(direct.model.parameters(),wrapped.model.parameters()):close(pa.grad,pb.grad)
                    item.update(status='PASS',max_logit_difference=float((a-b).abs().max()),online_counts=[dict(wa.counts),dict(wb.counts)])
                    for h,w in ((direct,wa),(wrapped,wb)):
                        w.verify();source_unchanged(h,state)
                        if w.counts['online_adam']!=1 or w.counts['memory_pushes']!=1:raise ValueError('smoke lifecycle')
                        w.release()
                    evidence[task][arm]=item;del direct,wrapped,wa,wb,a,b,payload;gc.collect()
            evidence[task]['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            del state,real,library;gc.collect()
        if updates!=dict(online=8,outer=4,inner=2):raise ValueError('smoke budget mismatch')
        private_json(out/'smoke.completion.json',dict(status='M2_SMOKE_PASS',updates=updates,evidence=evidence,
            **{k:receipt[k] for k in ('commit','config_sha256','registration_sha256','gpu_uuid')},
            paired_comparison_backend=dict(deterministic_algorithms=True,cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),scope='paired online smoke only',restored_deterministic_algorithms=torch.are_deterministic_algorithms_enabled()),gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as error:
        private_json(out/'smoke.failure.json',dict(status='M2_PARTIAL',task=task,arm=arm,updates=updates,current_counts=current_counts,evidence=evidence,reason=str(error),exception=type(error).__name__,gpu_seconds=budget.seconds(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),exit_code=1))
        raise


def history_state_discrete(host):
    return dict(counters=[(n,m.sample_num,m.new_sample) for n,m in host.model.named_modules() if isinstance(m,host.adabn)],
        adam_step=int(host.optimizer.state[host.prompt.data_prompt]['step']),memory=host.memory_bank.memory)


def train_all(out,registration,budget,progress,overlay):
    from .offline.adaptation_dd import cpu_tree
    for task,reg in registration['tasks'].items():
        progress.update(stage='training',task=task,method=None,episode=0,current_counts=None)
        state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True)
        real=source_proxy(reg['proxy'],task)
        library=inspect_history(overlay['tasks'][task]['history']['path'],task)
        for method in NEW_ARMS:
            progress.update(stage='training',method=method,episode=0)
            seed_all(20260907)
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
                    loss=ep.objective(base_method(method),S,real.mask,x,y,library[episode['state_index']])
                    g,=torch.autograd.grad(loss,S)
                    if not torch.isfinite(g).all(): raise ValueError('nonfinite image meta-gradient')
                    S.grad=g;optimizer.step();progress['outer']+=1
                    if method=='O2': progress['inner']+=1
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
            private_json(out/f'train_{task}_{method}.completion.json',dict(status='COMPLETE',episodes=600,episodes_sha256=overlay['tasks'][task]['episodes_sha256'],final_sha256=digest(artifact),source_unchanged=True,
                elapsed_seconds=time.monotonic()-started,counts=ep.counts,peak_allocated_bytes=torch.cuda.max_memory_allocated(),artifact_bytes=artifact.stat().st_size))
            del ep,S,optimizer;gc.collect()
        del library,state,real;gc.collect()
    # Freeze both tasks and both methods before any source/target scoring.
    finals={t:{m:json.loads((out/f'train_{t}_{m}.completion.json').read_text())['final_sha256'] for m in NEW_ARMS} for t in registration['tasks']}
    private_json(out/'final_artifacts.frozen.json',finals)

def evaluate_all(out,registration,budget,progress):
    for task,reg in registration['tasks'].items():
        state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task)
        proxies={}
        for method in NEW_ARMS:
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
                h=m2_host(task,arm,state,proxies.get(arm));watch=Observed(h);torch.cuda.reset_peak_memory_stats()
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
        train_all(out,registration,budget,progress,overlay);evaluate_all(out,registration,budget,progress)
        expected=sum(2*(len(r['source_query'])+len(r['target'])) for r in registration['tasks'].values())
        if (progress['online'],progress['outer'],progress['inner'],progress['records'])!=(expected,2400,1200,expected):raise ValueError('M2 actual budget/coverage mismatch')
        private_json(out/'run.completion.json',dict(status='M2_RUN_COMPLETE',progress=progress,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as error:
        private_json(out/'run.failure.json',dict(status='BUDGET_EXHAUSTED' if isinstance(error,BudgetExhausted) else 'M2_PARTIAL',progress=progress,exception=type(error).__name__,reason=str(error),gpu_seconds=budget.seconds(),exit_code=1))
        raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args(argv)
    os.umask(0o077)
    if a.stage=='smoke':
        if torch.cuda.is_initialized():raise ValueError('smoke workspace must be configured before CUDA initialization')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    receipt=json.loads(a.receipt.read_text());out,overlay,registration=validate(receipt,a.stage)
    if a.stage=='recompute':
        from .m2_analysis import recompute
        recompute(out,overlay,registration,receipt)
        if torch.cuda.is_initialized():raise ValueError('CPU recompute initialized CUDA')
    else:
        env=environment();private_json(out/(a.stage+'.environment.json'),env)
        prior=json.loads((Path(overlay['old_directory'])/'run.environment.json').read_text())
        differences={k:{'M1':prior.get(k),'M2':v} for k,v in env.items() if prior.get(k)!=v}
        private_json(out/(a.stage+'.environment_comparison.json'),dict(differences=differences,compatible=not differences))
        (smoke if a.stage=='smoke' else run)(receipt,out,overlay,registration)
    return 0


if __name__=='__main__':raise SystemExit(main())
