"""M4 finite six-fit / two-order comparison; old native online step is unchanged."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from .m1_run import Budget,BudgetExhausted,Observed,make_host,tensor_save,new_log
from .m2_run import deterministic_smoke_pair
from .m2_registration import inspect_history
from .m4_registration import load_registered
from .m4_sequences import target_order
from .offline.trajectory_dd import TrajectoryEpisode,detach_state
from .offline.adaptation_dd import cpu_tree,restore_history
from .host_diagnostic import close
from .host_diagnostic_run import private_json,append
from .source_io import source_proxy,read_pixels,read_mask
from .source_pilot import seed_all,transform_pixels,evaluate_after_step,validate_arm
from .source_pilot_release import digest,environment,source_unchanged,CONFIG_SHA
from .integrations.ctta_suite import checkout_root,REFERENCE_COMMIT
from .proxy_loss import FixedProxy,ProxyProvenance

ROOT=Path(__file__).resolve().parents[2]
CONFIG=ROOT/'configs/m4_trajectory_distillation_v1.json'
TRAIN_ARMS=('D4','L4','T4')


def validate_new_arm(records,expected,task,arm):
    if arm not in (*TRAIN_ARMS,'A','D2','O2'):raise ValueError('M4 scoring arm')
    validate_arm(records,expected,task,'A' if arm=='A' else 'B')
    for r in records:
        if r['arm']!=arm or r['task']!=task or r['counts']['online_adam']!=1 or r['counts']['memory_pushes']!=1:raise ValueError('M4 scoring lifecycle')
        json.dumps(r,allow_nan=False)
    return True


def validate(receipt,stage):
    if receipt['authorization']!='direct_user_M4_TRAJECTORY_DISTILLATION':raise ValueError('M4 authorization mismatch')
    commit=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
    if commit!=receipt['commit'] or subprocess.check_output(['git','-C',str(ROOT),'status','--porcelain'],text=True).strip():raise ValueError('execution commit dirty/mismatch')
    if digest(CONFIG)!=receipt['config_sha256'] or digest(ROOT/'configs/source_pilot_v0.json')!=CONFIG_SHA:raise ValueError('science config drift')
    for path,sha in json.loads(CONFIG.read_text())['inherited_file_sha256'].items():
        if digest(ROOT/path)!=sha:raise ValueError('old scientific file changed: '+path)
    checkout_root()
    if receipt['reference_commit']!=REFERENCE_COMMIT:raise ValueError('reference changed')
    out=Path(receipt['output_directory']).resolve()
    if out.is_relative_to(ROOT) or out.stat().st_uid!=os.getuid() or out.stat().st_mode&0o077:raise ValueError('private directory')
    if digest(out/'registration.json')!=receipt['registration_sha256']:raise ValueError('registration changed')
    overlay,reg=load_registered(out)
    if stage!='recompute':
        lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True)
        devices={int(s.split(',')[0]):s.split(',')[1].strip() for s in lines.splitlines()}
        if receipt['physical_gpu'] not in range(4,8) or devices.get(receipt['physical_gpu'])!=receipt['gpu_uuid'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=receipt['gpu_uuid']:raise ValueError('GPU binding')
    if stage in ['run','recompute']:
        smoke=json.loads((out/'smoke.completion.json').read_text())
        if digest(out/'smoke.completion.json')!=receipt['smoke_sha256'] or smoke['status']!='M4_SMOKE_PASS' or smoke['updates']!=dict(online=8,outer=6,inner=24):raise ValueError('smoke not bound/passed')
        if any(smoke[k]!=receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']):raise ValueError('smoke binding')
    return out,overlay,reg


def check_source(ep,state):
    source_unchanged(ep.host,state)
    if any(not torch.equal(v.cpu(),state[n]) for n,v in ep.clone.state_dict().items()):raise ValueError('offline source clone mutation')


def smoke(receipt,out,overlay,registration):
    sys.path.insert(0,str(ROOT/'tests'))
    from test_vptta_host import pixels,proxy as fixture
    budget=Budget(out);updates=dict(online=0,outer=0,inner=0);evidence={};current_counts={};task=None;arm=None
    try:
        for task,reg in registration['tasks'].items():
            seed_all(20260907);budget.check();torch.cuda.reset_peak_memory_stats()
            source=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task)
            history=inspect_history(overlay['tasks'][task]['history']['path'],task)[16]
            queries=[(pixels(task,i),fixture(task).mask) for i in range(4)];traces={};ev=evidence[task]={}
            with deterministic_smoke_pair():
                for arm in TRAIN_ARMS:
                    ep=TrajectoryEpisode(task,source,'cuda:0');current_counts=ep.counts
                    S=torch.nn.Parameter(real.pixel_rgb.to('cuda:0').clone());opt=torch.optim.Adam([S],lr=.01,betas=(.9,.999),eps=1e-8)
                    loss,end,trace=ep.window(arm,S,real.mask,queries,ep.initial(history),True);updates['inner']+=4
                    g,=torch.autograd.grad(loss,S)
                    if not torch.isfinite(g).all() or not g.norm()>0:raise ValueError('fixture meta-gradient nonfinite/zero')
                    before=S.detach().clone();S.grad=g;opt.step();updates['outer']+=1
                    with torch.no_grad():S.clamp_(0,1)
                    if not torch.isfinite(S).all() or torch.equal(S,before):raise ValueError('fixture outer update')
                    ev[arm]=dict(loss=float(loss.detach()),gradient_norm=float(g.norm()),image_delta_norm=float((S-before).norm()),counts=dict(ep.counts))
                    traces[arm]=trace;check_source(ep,source);ep.clear_graphs()
                    del ep,S,opt,loss,end,trace,g,before;gc.collect()
                close(traces['L4'],traces['T4'])
                h=make_host(task,'R',source,real);h._initial_hook.remove();h._started=True;restore_history(h,history)
                watch=Observed(h);current_counts=watch.counts;max_errors=[]
                for i,(x,y) in enumerate(queries):
                    pred=h.step(x);updates['online']+=1;watch.verify()
                    for arm in TRAIN_ARMS:
                        got=traces[arm][i];close(got['prediction'],pred.cpu());s=got['state']
                        close(s['prompt'],h.prompt.data_prompt.detach().cpu())
                        opt=h.optimizer.state[h.prompt.data_prompt]
                        for k in ['exp_avg','exp_avg_sq']:close(s['adam'][k],opt[k].cpu())
                        if s['adam']['step']!=int(opt['step']) or s['count']!=17+i or {m.sample_num for m in h.model.modules() if isinstance(m,h.adabn)}!={17+i}:raise ValueError('smoke counter')
                        if list(s['memory'])!=list(h.memory_bank.memory):raise ValueError('native memory selection/eviction')
                        for k,v in s['memory'].items():close(v,torch.from_numpy(h.memory_bank.memory[k]))
                        max_errors.append(float((got['prediction']-pred.cpu()).abs().max()))
                    del pred
                source_unchanged(h,source);watch.release()
                ev['native_reference']=dict(counts=watch.counts,max_prediction_difference=max(max_errors),positions=[17,18,19,20])
                del h,watch,traces;gc.collect()
            ev['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            print(json.dumps(dict(stage='smoke',task=task,status='PASS',updates=updates)),flush=True)
            del source,real,history,queries;gc.collect()
        if updates!=dict(online=8,outer=6,inner=24):raise ValueError('M4 smoke coverage')
        private_json(out/'smoke.completion.json',dict(status='M4_SMOKE_PASS',updates=updates,evidence=evidence,
            **{k:receipt[k] for k in ['commit','config_sha256','registration_sha256','gpu_uuid']},gpu_seconds=budget.seconds(),exit_code=0,
            paired_comparison_backend=dict(deterministic_algorithms=True,restored_deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))))
    except Exception as error:
        private_json(out/'smoke.failure.json',dict(status='M4_PARTIAL',task=task,arm=arm,updates=updates,current_counts=current_counts,evidence=evidence,reason=str(error),exception=type(error).__name__,gpu_seconds=budget.seconds(),exit_code=1))
        raise


def train_all(out,reg,budget,progress,overlay):
    for task,r in reg['tasks'].items():
        source=torch.load(r['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(r['proxy'],task)
        for arm in TRAIN_ARMS:
            seed_all(20260907);ep=TrajectoryEpisode(task,source,'cuda:0');S=torch.nn.Parameter(real.pixel_rgb.to('cuda:0').clone())
            opt=torch.optim.Adam([S],lr=.01,betas=(.9,.999),eps=1e-8,weight_decay=0)
            progress.update(stage='training',task=task,arm=arm,window=0,current_counts=ep.counts)
            def save(index):tensor_save(out/f'{task}_{arm}_{index}.pt',dict(pixels=S.detach().cpu(),optimizer=cpu_tree(opt.state_dict()),outer_step=progress['window'],seed=20260907))
            save(0);started=time.monotonic();torch.cuda.reset_peak_memory_stats();state=None
            with new_log(out/f'train_{task}_{arm}.jsonl') as f:
                for w in range(150):
                    budget.check();rows=r['sequence'][w*4:(w+1)*4]
                    if w%30==0:state=ep.initial()
                    queries=[]
                    for row in rows:
                        q=r['train_query'][row['query_index']]
                        queries.append((transform_pixels(read_pixels(q['image_path'],task,q['image_size']),row['transform']),read_mask(q['mask_path'],task,q['image_size'])))
                    before=S.detach().clone();opt.zero_grad(set_to_none=True)
                    loss,state,_=ep.window(arm,S,real.mask,queries,state);progress['inner']+=4;progress['source_visits']+=4
                    g,=torch.autograd.grad(loss,S)
                    if not torch.isfinite(g).all():raise ValueError('nonfinite image meta-gradient')
                    S.grad=g;opt.step();progress['outer']+=1
                    with torch.no_grad():S.clamp_(0,1)
                    if not torch.isfinite(S).all():raise ValueError('nonfinite image')
                    state=detach_state(state);progress['window']=w+1
                    if state['count']!=(w%30+1)*4 or state['adam']['step']!=state['count']:raise ValueError('trajectory was reset/truncated incorrectly')
                    append(f,dict(window=w+1,sequence=rows,stream=rows[0]['stream'],count=state['count'],memory_size=len(state['memory']),loss=float(loss.detach()),gradient_norm=float(g.norm()),image_delta_norm=float((S-before).norm()),image_adam_step=int(opt.state[S]['step']),counts=dict(ep.counts),elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
                    ep.clear_graphs();ep.memory.memory=dict(state['memory']);del loss,g,before,queries
                    if w+1 in [50,100,150]:save(w+1)
                    if (w+1)%10==0:print(json.dumps(dict(stage='training',task=task,arm=arm,window=w+1,source_visits=(w+1)*4,gpu_seconds=budget.seconds())),flush=True)
            check_source(ep,source);artifact=out/f'{task}_{arm}_150.pt'
            private_json(out/f'train_{task}_{arm}.completion.json',dict(status='COMPLETE',source_visits=600,outer_steps=150,sequence_sha256=overlay['tasks'][task]['sequence']['sha256'],counts=ep.counts,source_unchanged=True,final_sha256=digest(artifact),artifact_bytes=artifact.stat().st_size,elapsed_seconds=time.monotonic()-started,peak_allocated_bytes=torch.cuda.max_memory_allocated()))
            del ep,S,opt,state;gc.collect()
        del source,real;gc.collect()
    private_json(out/'final_artifacts.frozen.json',{t:{a:json.loads((out/f'train_{t}_{a}.completion.json').read_text())['final_sha256'] for a in TRAIN_ARMS} for t in reg['tasks']})


def evaluate_all(out,registration,budget,progress,overlay):
    frozen=json.loads((out/'final_artifacts.frozen.json').read_text())
    for task,reg in registration['tasks'].items():
        source=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);real=source_proxy(reg['proxy'],task);proxies={}
        for arm in (*TRAIN_ARMS,'D2','O2'):
            path=out/f'{task}_{arm}_150.pt' if arm in TRAIN_ARMS else Path(overlay['artifacts'][task][arm]['path'])
            if arm in TRAIN_ARMS and digest(path)!=frozen[task][arm]:raise ValueError('final proxy changed')
            artifact=torch.load(path,map_location='cpu',weights_only=True)
            if artifact.get('outer_step',artifact.get('episode'))!=(150 if arm in TRAIN_ARMS else 600):raise ValueError('evaluation checkpoint mismatch')
            proxies[arm]=FixedProxy(artifact['pixels'],real.mask,real.signed_distance,ProxyProvenance.SOURCE)
        for stage,order,arms in [('source',0,TRAIN_ARMS),('target',0,TRAIN_ARMS),('target',1,(*TRAIN_ARMS,'A','D2','O2'))]:
            selected=reg['source_query'] if stage=='source' else target_order(reg['target'],order)
            for arm in arms:
                name=f'{stage}{order}_{task}_{arm}';progress.update(stage=stage,order=order,task=task,arm=arm,visit=0)
                h=make_host(task,'A' if arm=='A' else 'R',source,proxies.get(arm));watch=Observed(h);progress['current_counts']=watch.counts
                torch.cuda.reset_peak_memory_stats()
                with new_log(out/(name+'.jsonl')) as f:
                    for i,row in enumerate(selected):
                        budget.check();torch.cuda.synchronize();start=time.monotonic();image=read_pixels(row['image_path'],task,row['image_size'])
                        p0=h.prompt.data_prompt.detach().clone();previous=dict(watch.counts);torch.cuda.synchronize();step_start=time.monotonic()
                        pred=h.step(image);progress['online']+=1;torch.cuda.synchronize();step_time=time.monotonic()-step_start;watch.verify()
                        if not torch.isfinite(pred).all():raise ValueError('nonfinite prediction')
                        # Labels first read after this visit's final prediction; never passed to host.
                        metrics=evaluate_after_step(pred,read_mask(row['mask_path'],task,row['image_size']),task)
                        delta={k:v-previous[k] for k,v in watch.counts.items()}
                        if any(not torch.isfinite(v).all() for v in h.optimizer.state[h.prompt.data_prompt].values() if isinstance(v,torch.Tensor)):raise ValueError('nonfinite native Adam')
                        record=dict(visit=i+1,sample_id=row['sample_id'],group_id=row['group_id'],segment='clean',domain=row['domain'],task=task,arm=arm,order=order,metrics=metrics,
                            prompt_state_delta_norm=float((h.prompt.data_prompt-p0).norm()),prompt_initialization_delta_norm=float((watch.init-p0).norm()),optimizer_update_norm=watch.update_norm,optimizer_steps_this_visit=delta['online_adam'],
                            native_counts=sorted({m.sample_num for m in h.model.modules() if isinstance(m,h.adabn)}),adam_step=int(h.optimizer.state[h.prompt.data_prompt]['step']),memory_size=h.memory_bank.get_size(),host_loss=watch.loss,
                            proxy_loss=float(h.last_proxy_loss.region.detach()) if h.last_proxy_loss is not None else None,source_versions_unchanged=True,optimizer_state_finite=True,pipeline_elapsed_seconds=time.monotonic()-start,host_step_elapsed_seconds=step_time,peak_allocated_bytes=torch.cuda.max_memory_allocated(),counts=delta)
                        validate_new_arm([record],[dict(visit=i+1,sample_id=row['sample_id'],group_id=row['group_id'],segment='clean')],task,arm)
                        if delta['backward_calls']!=1:raise ValueError('online backward count')
                        append(f,record);progress['records']+=1;progress['visit']=i+1
                        del pred,p0,image
                source_unchanged(h,source);watch.release()
                private_json(out/(name+'.completion.json'),dict(status='COMPLETE',records=len(selected),counts=watch.counts,source_unchanged=True))
                print(json.dumps(dict(stage=stage,order=order,task=task,arm=arm,records=len(selected),gpu_seconds=budget.seconds())),flush=True)
                del h,watch;gc.collect()
        del proxies,source,real;gc.collect()


def run(receipt,out,overlay,reg):
    smoke=json.loads((out/'smoke.completion.json').read_text());budget=Budget(out,smoke['gpu_seconds'])
    progress=dict(stage='start',online=0,outer=0,inner=0,source_visits=0,records=0)
    private_json(out/'run.start.json',dict(commit=receipt['commit'],status='M4_RUNNING'))
    try:
        train_all(out,reg,budget,progress,overlay);evaluate_all(out,reg,budget,progress,overlay)
        if {k:progress[k] for k in ['online','outer','inner','source_visits','records']}!=dict(online=2172,outer=900,inner=3600,source_visits=3600,records=2172):raise ValueError('M4 budget mismatch')
        private_json(out/'run.completion.json',dict(status='M4_RUN_COMPLETE',progress=progress,gpu_seconds=budget.seconds(),exit_code=0))
    except Exception as error:
        private_json(out/'run.failure.json',dict(status='M4_PARTIAL',progress=progress,exception=type(error).__name__,reason=str(error),gpu_seconds=budget.seconds(),exit_code=1));raise


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['smoke','run','recompute']);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args(argv);os.umask(0o077)
    if a.stage=='smoke':
        if torch.cuda.is_initialized():raise ValueError('CUDA initialized before deterministic scope setup')
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    receipt=json.loads(a.receipt.read_text());out,overlay,reg=validate(receipt,a.stage)
    if a.stage=='recompute':
        from .m4_analysis import recompute
        recompute(out,overlay,reg,receipt)
        if torch.cuda.is_initialized():raise ValueError('CPU recompute initialized CUDA')
    else:
        env=environment();private_json(out/(a.stage+'.environment.json'),env)
        prior=json.loads((Path(overlay['m2_directory'])/'run.environment.json').read_text())
        differences={k:{'M2':prior.get(k),'M4':v} for k,v in env.items() if prior.get(k)!=v}
        private_json(out/(a.stage+'.environment_comparison.json'),dict(differences=differences,compatible=not differences))
        if differences:raise ValueError('formal environment changed from M2')
        (smoke if a.stage=='smoke' else run)(receipt,out,overlay,reg)
    return 0
