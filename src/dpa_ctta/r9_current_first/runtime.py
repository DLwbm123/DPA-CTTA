"""Native R9 worker kernel. Every private/GPU entry requires new bound authorization."""
import errno
import json
import os
import time
from pathlib import Path
import torch
from .protocol import SPEC,SPEC_SHA,CAPS,digest,parse_recipe
from .admission import require_authorized
from .identity import verify_runtime
from .storage import write_json,lease,PhaseJournal
from .physical import Meter
from .ledger import Ledger


def read(path):return json.loads(Path(path).read_text())
def source_receipts(root):return {j['id']:read(root/'source'/j['id']/'complete.json') for j in SPEC['source_tasks']}
def disk(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()) if Path(root).exists() else 0


def failure_class(exc):
    if isinstance(exc,FloatingPointError) or any(t in str(exc).lower() for t in ('non-finite','nonfinite','nan','infinite')):return 'NUMERICAL'
    if isinstance(exc,OSError) and exc.errno in (errno.EIO,errno.ETIMEDOUT,errno.ESTALE,errno.ECONNRESET):return 'INFRASTRUCTURE'
    if isinstance(exc,OSError) and exc.errno==errno.ENOSPC:return 'RESOURCE'
    if 'aggregate resource cap' in str(exc):return 'RESOURCE'
    if isinstance(exc,PermissionError):return 'ISOLATION'
    return 'IDENTITY_OR_IMPLEMENTATION'


class Budget:
    def __init__(self,ledger,attempt,token,root,initial,gpu):
        self.ledger,self.attempt,self.token,self.root=ledger,attempt,token,root
        self.initial=initial.copy();self.gpu=gpu;self.last=0;self.meter=None
        self.cost=dict.fromkeys(CAPS,0);self.baseline=disk(root)
        self.run_baseline=disk(ledger.root.parent)
    def observe(self,value=None,extra=0,force=False):
        if value is not None:self.cost.update({k:v for k,v in value.items() if k!='disk_bytes'})
        now=time.monotonic()
        if extra or force or now-self.last>=2:
            growth=max(0,disk(self.root)+extra-self.baseline)
            if self.run_baseline+growth+8*1024**2>CAPS['disk_bytes']:
                self.ledger.stop('aggregate resource cap: occupied output bytes');raise RuntimeError('aggregate resource cap: occupied output bytes')
            self.cost['disk_bytes']=max(self.cost['disk_bytes'],growth)
        if not self.gpu:self.cost['gpu_seconds']=0
        if force or now-self.last>=2 or any(self.cost[k]>self.initial[k] for k in CAPS):
            self.ledger.observe(self.attempt,self.token,self.cost);self.last=now
            self.initial={k:max(self.initial[k],self.cost[k]) for k in CAPS}
    def check_write(self,path,size):
        if Path(path).resolve().is_relative_to(self.root.resolve()):self.observe(extra=size)
    def __call__(self):
        if self.meter:self.meter.check()
        else:self.observe()


def source_job(config,root,node,assignment,budget,failure):
    from .assets import open_source,make_method,legacy_snapshot,ContinuationUnavailable
    from .training import SourceTrainer
    from .source_run import run
    job=node['job'];selected=read(root/'recipe_selection.json')['recipes'] if 'SELECTED' in job['recipe'] else None
    b=config['bindings'];jobroot=root/'source'/job['id'];jobroot.mkdir(parents=True,exist_ok=True)
    with open_source(b,assignment,budget) as (data,segmenter,oracles,bases,scaler):
        budget.meter.attach(segmenter.model)
        method,kind=make_method(job['recipe'],job['seed'],b,bases,scaler,selected)
        binding=dict(code_sha=config['code_sha'],spec_sha256=SPEC_SHA,job=job,recipe=kind,assets_sha256=digest(b))
        trainer=SourceTrainer(segmenter,method,data,oracles['fit'],job['seed'],binding,recipe=kind)
        if kind=='LEGACY' and failure is None:
            try:
                snapshot,ref=legacy_snapshot(b,job);trainer.fork_legacy(snapshot,ref['binding'],ref['sha256'])
                start=dict(mode='FULL_SNAPSHOT_CONTINUATION',step=4000,parent_sha256=ref['sha256'])
            except ContinuationUnavailable as exc:start=dict(mode='PREREGISTERED_FRESH_16000',step=0,reason=str(exc))
            write_json(jobroot/'start.json',start)
        return run(trainer,jobroot,oracles['cal'],oracles['val'],budget,failure)


def lr_job(config,root,node,assignment,budget,failure):
    from .assets import open_source
    from .gradient import STEPS,BN_LR,LATENT_LR
    from .gradient_calibration import POLICIES,episode,select
    from .resolution import resolve
    from .factory import construct
    from .source_run import drive
    policy=config['lr_source_policy']
    if policy not in POLICIES:raise ValueError('LR source policy unresolved')
    with open_source(config['bindings'],assignment,budget) as (data,_,oracles,__,___):pass
    recipes=read(root/'recipe_selection.json');receipts=source_receipts(root)
    binding=dict(code_sha=config['code_sha'],spec_sha256=SPEC_SHA,sources=digest(receipts),recipes=digest(recipes),policy=policy)
    allrows=[];jobroot=root/'gradient_selection'
    class Episodes:
        def __init__(self,host,seed):self.host,self.seed=host,seed;self.initial=host.snapshot();self.steps=0;self.rows=[]
        def step(self):
            from ..r7_shared.numerics import COUNTS
            counts=COUNTS.copy();self.host.restore(self.initial);COUNTS.clear();COUNTS.update(counts)
            row=episode(self.host,data,oracles['cal'],self.steps,self.seed,budget)
            self.rows.append(row);self.steps+=1;return row
        def snapshot(self):return dict(steps=self.steps,rows=self.rows)
        def restore(self,state):
            if state['steps']!=len(state['rows']) or not 0<=state['steps']<=64:raise ValueError('LR episode snapshot')
            self.steps=state['steps'];self.rows=state['rows']
    for arm in STEPS:
        for lr in (BN_LR if arm=='BN_RESET_G1' else LATENT_LR):
            for seed in POLICIES[policy]:
                phase=f'{arm}.{lr}.{seed}';journal=PhaseJournal(jobroot,phase,binding);done=journal.completed()
                if done is None:
                    slot=dict(id=phase,arm=arm,seed=seed,order=0,phase='GRADIENT',checkpoint='selected')
                    resolved=resolve(slot,recipes,receipts)
                    host,close=construct(resolved,config,root/'source',{'sha256':digest(binding)},dict(selected_lr={arm:{'global':lr}}))
                    handle=budget.meter.attach(host.segmenter.model)
                    try:
                        worker=Episodes(host,seed)
                        drive(worker,journal,64,budget,failure if journal.root.exists() else None)
                        done=journal.complete(worker.rows)
                    finally:handle.remove();close()
                allrows.extend(done['result'])
    result=select(allrows,policy);write_json(root/'gradient_selection.json',result);return result


def dispatch(config,root,node,assignment,budget,failure):
    kind=node['kind']
    if failure and kind in ('source','select_lr'):
        old=root/'source'/node['job']['id'] if kind=='source' else root/'gradient_selection'
        if not list(old.glob('*/latest.json')):raise ValueError('no equivalent source snapshot for infrastructure recovery')
    if kind=='source':return source_job(config,root,node,assignment,budget,failure)
    if kind=='select_lr':return lr_job(config,root,node,assignment,budget,failure)
    if kind=='bind_assets':
        from .assets import validate_metadata
        from ..r8_ba.inputs import bind_metadata
        validate_metadata(config['bindings']);meta=bind_metadata(config['bindings']['refs'])
        result=dict(schema='R9_ASSET_BINDING_V1',bindings_sha256=digest(config['bindings']),audit=meta['audit'])
        write_json(root/'assets.json',result);return result
    if kind=='select_recipes':
        from .selection import recipes
        rows={}
        for j in SPEC['source_tasks']:
            if j['seed'] not in (20260924,20260925) or j['recipe'].startswith('MLP'):continue
            route,recipe,mode=parse_recipe(j['recipe']);rows[route,recipe,mode,j['seed']]=read(root/'source'/j['id']/'complete.json')['selection']
        result=recipes(rows);write_json(root/'recipe_selection.json',result);return result
    if kind=='lock':
        from .target import lock_sources
        result=lock_sources(source_receipts(root),read(root/'gradient_selection.json'),read(root/'recipe_selection.json'))
        write_json(root/'source_lock.json',result);return result
    if kind in ('online','score'):
        from .streams import sequence
        from ..r7_source_prep.registry import verified
        from .target import online,score,retire_probabilities
        from .resolution import resolve
        from .factory import construct
        lock=read(root/'source_lock.json')
        from .target import lock_sources
        if lock!=lock_sources(source_receipts(root),read(root/'gradient_selection.json'),read(root/'recipe_selection.json')):raise ValueError('frozen source choices changed')
        ref=config['bindings']['refs']['target']
        rows=sequence(json.loads(verified(ref['path'],ref['sha256'],32*1024**2)),node['job']['order'])
        resolved=resolve(node['job'],read(root/'recipe_selection.json'),source_receipts(root))
        jobroot=root/'target'/node['job']['id'];jobid=node['job']['id']
        if kind=='online':
            host,close=construct(resolved,config,root/'source',lock,read(root/'gradient_selection.json'))
            model=host.native.model if hasattr(host,'native') else host.segmenter.model
            handle=budget.meter.attach(model)
            try:
                result=online(host,rows,config['bindings']['target_root'],jobroot,jobid,lock,budget,failure)
                write_json(jobroot/'resolved.json',resolved);write_json(jobroot/'online_attempt.json',{'name':budget.attempt});return result
            finally:handle.remove();close()
        seal=read(jobroot/'online_complete.json')
        if (jobroot/'score_complete.json').exists():result=read(jobroot/'score_complete.json')
        else:result=score(rows,config['bindings']['target_root'],jobroot,jobid,seal['identity']['context_sha256'],lock,budget,failure)
        retire_probabilities(jobroot);retired=read(jobroot/'probabilities_retired.json')
        budget.ledger.release_disk(read(jobroot/'online_attempt.json')['name'],retired['bytes'],digest(retired))
        return result
    raise ValueError('unregistered R9 node')


def worker(config,node,assignment,attempt,token,failure=None):
    root=require_authorized(config);identity=verify_runtime(config)
    if config['lr_source_policy'] is None:raise ValueError('LR policy unresolved')
    ledger=Ledger(root/'ledger',identity,[g['physical_id'] for g in config['gpu_assignments']])
    gpu=node['resource']=='gpu';jobroot=root/('source' if node['kind']=='source' else 'target')/node.get('job',{}).get('id',node['id'])
    if node['kind']=='select_lr':jobroot=root/'gradient_selection'
    budget=Budget(ledger,attempt,token,jobroot,config['profile']['node_budgets'][node['id']],gpu)
    start=time.monotonic();result=None;failure_record=None
    from ..r8_ba import worker_budget as oldbudget
    original=oldbudget.check_write;oldbudget.check_write=budget.check_write
    try:
        with lease(root/'leases'/node['id'],dict(identity,node=node['id'])):
            if gpu:
                from .assets import gpu_policy
                gpu_policy(assignment)
                meter=Meter(None,budget.observe);meter.start_override=start
                torch.cuda.reset_peak_memory_stats()
                with meter:
                    budget.meter=meter;result=dispatch(config,root,node,assignment,budget,failure)
                budget.observe(meter.cost,force=True)
            else:result=dispatch(config,root,node,None,budget,failure)
    except BaseException as exc:
        failure_record=dict(node=node['id'],attempt=attempt,**{'class':failure_class(exc)},reason=str(exc),error_type=type(exc).__name__,evidence=f'attempts/{attempt}.json')
        if failure_record['class'] in ('RESOURCE','ISOLATION','IDENTITY_OR_IMPLEMENTATION'):ledger.stop(failure_record['class']+': '+str(exc))
    finally:
        oldbudget.check_write=original
        budget.cost['disk_bytes']=max(budget.cost['disk_bytes'],max(0,disk(jobroot)-budget.baseline))
        if budget.meter:budget.cost.update({k:v for k,v in budget.meter.cost.items() if k!='disk_bytes'})
        receipt=dict(schema='R9_ATTEMPT_V1',identity=identity,node=node['id'],attempt=attempt,physical_gpu=assignment if gpu else None,actual=budget.cost,wall_seconds=time.monotonic()-start,status='FAILED' if failure_record else 'COMPLETE',failure=failure_record,result=result,
                     peak_allocated_bytes=torch.cuda.max_memory_allocated() if gpu and torch.cuda.is_initialized() else 0,
                     peak_reserved_bytes=torch.cuda.max_memory_reserved() if gpu and torch.cuda.is_initialized() else 0)
        (root/'attempts').mkdir(exist_ok=True);write_json(root/'attempts'/f'{attempt}.json',receipt)
        # A global stop intentionally prevents further ledger mutations; the complete failed reservation stays charged.
        try:ledger.observe(attempt,token,budget.cost,settle=True,failed=failure_record is not None)
        except RuntimeError:
            if failure_record is None:raise
    return receipt
