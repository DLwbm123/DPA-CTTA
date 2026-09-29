"""Independent R10 dispatch; source-only training and image/CPU-score separation."""
import json,errno,time,os
from pathlib import Path
from .protocol import SPEC,SPEC_SHA,CAPS,digest
from .identity import require
from .ledger import Ledger
from ..r9_current_first.storage import write_json,lease,load_torch
from ..r9_current_first.physical import Meter


def read(p):return json.loads(Path(p).read_text())
def disk(root):return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file()) if Path(root).exists() else 0
def receipts(root):return {j['id']:read(root/'source'/j['id']/'complete.json') for j in SPEC['source_jobs'] if (root/'source'/j['id']/'complete.json').exists()}

def classification(exc):
    if isinstance(exc,FloatingPointError) or any(x in str(exc).lower() for x in ('nonfinite','non-finite','nan','infinite')):return 'NUMERICAL'
    if isinstance(exc,OSError) and exc.errno in (errno.EIO,errno.ETIMEDOUT,errno.ESTALE,errno.ECONNRESET):return 'INFRASTRUCTURE'
    if isinstance(exc,PermissionError):return 'ISOLATION'
    if 'aggregate resource cap' in str(exc) or isinstance(exc,OSError) and exc.errno==errno.ENOSPC:return 'RESOURCE'
    return 'IDENTITY_OR_IMPLEMENTATION'


class Budget:
    def __init__(self,ledger,attempt,token,root,gpu):
        self.ledger,self.attempt,self.token,self.root,self.gpu=ledger,attempt,token,root,gpu
        self.cost=dict.fromkeys(CAPS,0);self.meter=None;self.last=0;self.baseline=disk(root)
    def observe(self,cost=None,extra=0,force=False):
        if cost:self.cost.update(cost)
        if not self.gpu:self.cost['gpu_seconds']=0
        now=time.monotonic()
        if force or extra or now-self.last>1:
            occupied=disk(self.ledger.root.parent)
            if occupied+extra+8*1024**2>CAPS['disk_bytes']:raise RuntimeError('aggregate resource cap: disk')
            self.cost['disk_bytes']=max(self.cost['disk_bytes'],disk(self.root)-self.baseline+extra)
            self.ledger.observe(self.attempt,self.token,self.cost);self.last=now
    def __call__(self):
        if self.meter:self.meter.check()
        else:self.observe()
    def check_write(self,path,size):
        if Path(path).resolve().is_relative_to(self.root.resolve()):self.observe(extra=size)


def dispatch(config,root,node,gpu,budget,failure):
    kind=node['kind'];b=config['bindings']
    if kind in ('bind','freeze'):
        if kind=='bind':
            from .assets import validate_metadata
            validate_metadata(b)
        result=dict(schema='R10_'+kind.upper()+'_V1',spec_sha256=SPEC_SHA,assets_sha256=digest(b),code_sha=config['code_sha']);write_json(root/(kind+'.json'),result);return result
    if kind=='select':
        from .evaluation import choose_families
        result=choose_families(receipts(root));write_json(root/'selection.json',result);return result
    if kind=='lock':
        from .target import lock_sources
        result=lock_sources(receipts(root),read(root/'selection.json'));write_json(root/'source_lock.json',result);return result
    if kind in ('train','d0'):
        from .assets import open_source
        from .source import Source
        from .controller import Actor
        from .learning import Trainer
        from .source_run import run
        from .evaluation import d0
        j=node.get('job');seed=j['seed'] if j else node['seed'];a=Actor(seed);reference=None
        method='WARM' if j and j['kind']=='warmup' else j['method'] if j else 'D0'
        if method.startswith('SELECTED_'):method=read(root/'selection.json')['selected'][method.split('_')[1]]
        if method is None:raise ValueError('no valid selected family')
        if method!='WARM':
            ref=receipts(root)[f'WARM_{seed}']['artifact'];saved=load_torch(root/'source'/f'WARM_{seed}'/ref['file'],ref['sha256']);a.load_state_dict(saved['actor']);reference=a
        binding=dict(code_sha=config['code_sha'],spec_sha256=SPEC_SHA,assets=digest(b),node=node,method=method)
        with open_source(b,gpu,budget) as (data,seg,oracles,controller):
            handle=budget.meter.attach(seg.model)
            try:
                source=Source(data,oracles,seg,controller,digest(b))
                if kind=='d0':
                    result=d0(source,a,seed,budget);write_json(root/f'D0_{seed}.json',result);return result
                t=Trainer(a,controller,source,seed,method,binding,reference,budget)
                return run(t,root/'source'/j['id'],budget,failure)
            finally:handle.remove()
    if kind in ('online','score'):
        from .streams import sequence
        from .factory import resolve,construct
        from .target import online,score,retire_probabilities
        from ..r7_source_prep.registry import verified
        slot=node['job'];lock=read(root/'source_lock.json');sources=receipts(root);selection=read(root/'selection.json')
        if lock['payload']['sources']!={k:digest(v) for k,v in sources.items()} or lock['payload']['selection']!=digest(selection):raise ValueError('locked source receipts changed')
        resolved=resolve(slot,selection,sources)
        reg=b['refs']['target'];rows=sequence(json.loads(verified(reg['path'],reg['sha256'],16*1024**2)),slot['order']);jobroot=root/'target'/slot['id']
        if kind=='online':
            host,close=construct(resolved,config,root,lock)
            model=host.native.model if hasattr(host,'native') else host.segmenter.model;handle=budget.meter.attach(model)
            try:return online(host,rows,b['target_root'],jobroot,slot['id'],lock,budget,failure)
            finally:handle.remove();close()
        seal=read(jobroot/'online_complete.json');result=score(rows,b['target_root'],jobroot,slot['id'],seal['identity']['context_sha256'],lock,budget,failure)
        retire_probabilities(jobroot);budget.ledger.sync_disk(digest(result));return result
    raise ValueError('R10 node kind')


def worker(config,node,assignment,attempt,token,failure=None):
    identity=require(config);root=Path(config['output_root']);root.mkdir(parents=True,exist_ok=True)
    jobroot=root/'source'/node['job']['id'] if node['kind']=='train' else root/'target'/node['job']['id'] if node['kind'] in ('online','score') else root/'control'/node['id']
    jobroot.parent.mkdir(parents=True,exist_ok=True)
    ledger=Ledger(root/'ledger',identity,[g['physical_id'] for g in config['gpu_assignments']]);budget=Budget(ledger,attempt,token,jobroot,assignment is not None)
    result=None;fail=None;start=time.monotonic()
    import torch
    from ..r8_ba import worker_budget
    original=worker_budget.check_write;worker_budget.check_write=budget.check_write
    try:
        if failure:
            from .recovery import verify_failure
            verify_failure(root,jobroot,node['id'],identity,failure)
        if assignment:
            from .assets import gpu_policy
            gpu_policy(assignment);meter=Meter(None,budget.observe)
            with meter:
                budget.meter=meter;result=dispatch(config,root,node,assignment,budget,failure)
        else:result=dispatch(config,root,node,None,budget,failure)
    except BaseException as exc:
        from .recovery import evidence
        fail=dict(node=node['id'],attempt=attempt,**{'class':classification(exc)},reason=str(exc),error_type=type(exc).__name__,evidence=evidence(identity,node['id'],attempt,jobroot))
    finally:
        worker_budget.check_write=original
        if budget.meter:budget.cost.update(budget.meter.cost)
        budget.cost['disk_bytes']=max(0,disk(jobroot)-budget.baseline)
        record=dict(schema='R10_ATTEMPT_V1',identity=identity,node=node['id'],attempt=attempt,status='FAILED' if fail else 'COMPLETE',failure=fail,actual=budget.cost,result=result,wall_seconds=time.monotonic()-start,physical_gpu=assignment)
        (root/'attempts').mkdir(exist_ok=True);write_json(root/'attempts'/f'{attempt}.json',record)
        ledger.observe(attempt,token,budget.cost,settle=True,failed=fail is not None);ledger.sync_disk()
        if fail and fail['class'] not in ('INFRASTRUCTURE','NUMERICAL'):ledger.stop(fail['class']+': '+fail['reason'])
    return record
