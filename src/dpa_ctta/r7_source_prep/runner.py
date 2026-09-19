"""Finite, separately authorized SOURCE_PREP only. No target dispatch or GPU route."""
import copy,io,json,os,signal,stat,subprocess,sys,time
from collections import Counter
from pathlib import Path
import torch
from .registry import verified,ordinary,audit,Reader,digest
from ..r7_shared.context import SCIENCE,provenance,HASH_COST
from ..r7_shared.numerics import COUNTS
from ..r7_shared.network import Segmenter
from ..r7_shared.preparation import prepare_tensors,inference_from_tensors
from ..r7_shared.io import Output,checked_path
from ..r1.supervise import stop_owned

ROOT=Path(__file__).resolve().parents[3]
PHYSICAL=('backbone_forwards','source_backward_calls','source_Adam','source_AdamW','calibration_backward_calls','calibration_Adam','source_VJP')
DEFAULTS=ROOT/'configs/r7_source_prep.defaults.json'

def code_identity():
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=normal'],cwd=ROOT,text=True).strip():raise ValueError('clean final code required')
    return sha

def device_policy(config):
    if config.get('device')=='cpu':
        if config.get('physical_GPU_ids') is not None or config.get('dtype_policy')!='R7_CPU_FP32_MODEL_FP64_LATENT_V1':raise ValueError('CPU dtype/device route')
    elif config.get('device')=='cuda:0':
        ids=config.get('physical_GPU_ids')
        if not isinstance(ids,list) or len(ids)!=1 or type(ids[0]) is not int or ids[0] not in (5,6,7):raise ValueError('one authorized physical GPU required')
        if config.get('dtype_policy')!='R7_CUDA_FP32_BACKBONE_CPU_FP64_LATENT_V1':raise ValueError('GPU dtype route')
        if os.environ.get('CUDA_VISIBLE_DEVICES')!=str(ids[0]):raise ValueError('physical GPU visibility binding')
    else:raise ValueError('unsupported source device')
    for key in ('wall_seconds','output_bytes','max_asset_bytes','max_decoded_bytes'):
        if type(config.get(key)) is not int or config[key]<=0:raise ValueError('finite positive resource cap required')
    if config.get('workers')!=1 or config.get('threads')!=2 or config.get('retry')!=False:raise ValueError('one worker/two CPU threads/no retry required')

def preflight(receipt):
    # Authority/scope are checked before config/registries, raw data or devices.
    if receipt.get('schema')!='R7_SOURCE_PREP_AUTH_V1' or receipt.get('scope')!='SOURCE_PREP' or receipt.get('enabled') is not True:raise PermissionError('SOURCE_PREP explicitly disabled')
    auth=receipt.get('user_authorization',{})
    if auth.get('granted') is not True or auth.get('scope')!='SOURCE_PREP' or not auth.get('receipt_id'):raise PermissionError('new scope-specific user receipt required')
    review=receipt.get('execution_layer_review',{})
    actual=code_identity()
    if review.get('status') not in ('PASS','USER_AUTHORIZED_GPU_QUALIFIED') or review.get('scope')!='SOURCE_PREP' or review.get('code_sha')!=actual or receipt.get('code_sha')!=actual:raise PermissionError('final execution-layer review/code mismatch')
    if receipt.get('science_sha256')!=SCIENCE:raise ValueError('science binding')
    for name,expected in SCIENCE.items():verified(ROOT/'docs/review/r7/input/specs'/name,expected,1024**2)
    for key in ('manifest','split','target','config'):
        if review.get(key+'_sha256')!=receipt[key]['sha256']:raise ValueError('review metadata/resource binding')
    config=json.loads(verified(receipt['config']['path'],receipt['config']['sha256'],1024**2))
    if config.get('enabled') is not True or config.get('scope')!='SOURCE_PREP':raise PermissionError('source resource configuration disabled')
    device_policy(config)
    if config['device']!='cpu':
        if review['status']!='USER_AUTHORIZED_GPU_QUALIFIED':raise PermissionError('new GPU transition authority required; old CPU PASS is insufficient')
        if auth.get('gpu_transition_authorized') is not True or auth.get('base_cpu_code_sha')!='f719c703087b38c07bdfbe7ce9dcfa62d88a12d9':raise PermissionError('explicit GPU transition grant required')
        binding=receipt['gpu_qualification'];q=json.loads(verified(binding['path'],binding['sha256'],1024**2))
        if q.get('status')!='PASSED' or q.get('code_sha')!=actual or q.get('physical_GPU_ids')!=config['physical_GPU_ids'] or q.get('failures')!=0 or not q.get('checks'):raise ValueError('GPU qualification binding')
    elif review['status']!='PASS':raise PermissionError('CPU route requires original review')
    if config.get('resource_authorization')!={'scope':'SOURCE_PREP','receipt_id':auth['receipt_id']}:raise PermissionError('resource authorization is not this user receipt')
    out=checked_path(receipt['output_dir']);root=checked_path(receipt['source_root']);storage=checked_path(config['storage_root'])
    if not root.is_dir() or not storage.is_dir():raise ValueError('existing source/storage directories required')
    if out.exists() or not out.is_relative_to(storage):raise ValueError('fresh approved storage required')
    protected=[root,checked_path(receipt['checkpoint_path']),checked_path(ROOT),*[checked_path(receipt[k]['path']) for k in ('manifest','split','target','config')]]
    for path in protected:
        if out==path or out in path.parents or path in out.parents:raise ValueError('source/code/metadata output overlap')
    documents={k:json.loads(verified(receipt[k]['path'],receipt[k]['sha256'],32*1024**2)) for k in ('manifest','split','target')}
    manifest,frozen,target=(documents[k] for k in ('manifest','split','target'));info=audit(manifest,frozen,target)
    if receipt.get('source_binding_status')!='BOUND' or review.get('checkpoint_sha256')!=manifest['checkpoint']['sha256']:raise ValueError('source file bindings pending')
    if info['groups']*5*512*512*4>config['max_decoded_bytes']:raise ValueError('decoded source memory cap; no subsampling')
    return dict(receipt=copy.deepcopy(receipt),config=config,manifest=manifest,split=frozen,target=target,audit=info)

def expected_counts(folds):
    f,c,v=(len(folds[k]) for k in ('fit','cal','val'));result={}
    for fold,n in [('fit',128),('cal',32),('val',32)]:result['oracle_'+fold]=dict(backbone_forwards=n*32,source_backward_calls=n*16,source_Adam=n*16)
    result.update(shared_basis={},A_basis=dict(backbone_forwards=32,source_VJP=1024),scaler=dict(backbone_forwards=128*f))
    for group in 'ABC':
        for mode in ('FULL','STATIC'):
            name=group+'_'+mode;result[name+'/fit']=dict(backbone_forwards=12000,source_backward_calls=1000,source_AdamW=1000);result[name+'/cal']=dict(backbone_forwards=1024,calibration_backward_calls=256,calibration_Adam=256);result[name+'/val']=dict(backbone_forwards=768)
            if group=='C':result[name+'/constant_variance']=dict(backbone_forwards=32*c)
    for fold,n,groups in [('fit',128,f),('cal',32,c),('val',32,v)]:result['oracle_query_'+fold]=dict(backbone_forwards=2*n*(groups-2))
    return result

class Meter:
    def __init__(self,expected,wall_seconds,sink,counter=None):
        self.counter=COUNTS if counter is None else counter
        self.expected=expected;self.wall_seconds=wall_seconds;self.sink=sink;self.started=time.monotonic();self.phase=None;self.records=[];self.baseline=self.counter.copy();self.hash_before=HASH_COST.copy();self.time=self.started;self.cpu=time.process_time()
    def check(self):
        if time.monotonic()-self.started>=self.wall_seconds:raise TimeoutError('finite source wall cap')
        if self.phase in self.expected:
            for k,v in self.expected[self.phase].items():
                if self.counter[k]-self.baseline[k]>v:raise ValueError('source phase budget exceeded: '+self.phase+'/'+k)
    def mark(self,name,validate=True):
        if self.phase is not None:
            previous=self.phase;delta=self.counter-self.baseline;expected=self.expected.get(previous,{})
            mismatch=[k for k in PHYSICAL if delta[k]!=expected.get(k,0)] if validate else []
            row=dict(phase=previous,counts=dict(delta),hash_cost=dict(HASH_COST-self.hash_before),wall_seconds=time.monotonic()-self.time,cpu_seconds=time.process_time()-self.cpu,completed=validate and not mismatch)
            self.phase=None;self.records.append(row);self.sink(row)
            if mismatch:raise ValueError('source budget mismatch: '+previous+'/'+mismatch[0])
        self.phase=name;self.baseline=self.counter.copy();self.hash_before=HASH_COST.copy();self.time=time.monotonic();self.cpu=time.process_time()
        if validate:self.check()
    def before_forward(self,*args):
        self.check()
        if self.counter["backbone_forwards"]-self.baseline["backbone_forwards"]>=self.expected.get(self.phase,{}).get("backbone_forwards",0):raise ValueError("forward phase budget exhausted")
    def complete(self):
        done={r['phase'] for r in self.records if r['completed']}
        if not set(self.expected)<=done:raise ValueError('missing mandatory source phases')

# Parent first/cleanup/resource/evidence errors, supervisor and completion: six
# bounded 16 KiB records reserved globally after the child has been reaped.
TERMINAL_RESERVE=6*16384

def tree_bytes(path):
    total=0
    for p in Path(path).rglob('*'):
        info=p.lstat()
        if stat.S_ISDIR(info.st_mode):continue
        if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1:raise ValueError('nonordinary output tree entry')
        total+=info.st_size
    return total

def resource_check(out,caps,started,reserve=TERMINAL_RESERVE):
    if time.monotonic()-started>=caps['wall_seconds']:raise TimeoutError('source worker wall cap')
    if tree_bytes(out.path)+reserve>caps['output_bytes']:raise ValueError('source worker output cap; terminal reserve required')

class BudgetOutput(Output):
    def __init__(self,path,binding,limit):self.limit=limit;self.used=0;super().__init__(path,binding)
    def bytes(self,name,data):
        if Path(name).name!=name or name in ('.','..'):raise ValueError('output path')
        if tree_bytes(self.path)+len(data)>self.limit-65536:raise ValueError('output cap; evidence reserve retained')
        if name!='owner.json' and json.loads((self.path/'owner.json').read_text())!=dict(owner=self.owner,binding=self.binding):raise ValueError('output ownership')
        with (self.path/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        self.used+=len(data)
    def write(self,name,value):self.bytes(name,(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())
    def evidence(self,name,value):
        if Path(name).name!=name or name in ('.','..'):raise ValueError('evidence output path')
        if json.loads((self.path/'owner.json').read_text())!=dict(owner=self.owner,binding=self.binding):raise ValueError('evidence output ownership')
        # Small reserved terminal evidence; never overwrite a first failure.
        raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
        if len(raw)>16384 or tree_bytes(self.path)+len(raw)>self.limit:raise ValueError('failure evidence cap')
        with (self.path/name).open('xb') as f:f.write(raw)
        self.used+=len(raw)

def load_model(raw,device='cpu'):
    from ..integrations.ctta_suite import build_reference_model
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    model,_=build_reference_model('fundus');model.load_state_dict(state,strict=True)
    if any(t.device.type!='cpu' or t.dtype!=torch.float32 for t in model.parameters()):raise ValueError('source model CPU float32 required')
    return Segmenter(model,device=device)

def configure_backend(config):
    if config['device']=='cpu':return
    # Explicit FP32, deterministic kernels. CPU RNG and small FP64 algebra stay
    # unchanged; no mixed precision or TF32 numerical-policy substitution.
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('deterministic cuBLAS configuration required')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise ValueError('exactly one visible CUDA device required')

def release(result,segmenter,out):
    required={g+'_'+mode for g in 'ABC' for mode in ('FULL','STATIC')}
    if set(result['models'])!=required:raise ValueError('six independent artifacts required')
    inventory={}
    for name,artifact in result['models'].items():
        if artifact['fit_steps']!=1000 or artifact['cal_steps']!=256:raise ValueError('full source training budget required')
        # Exercise the actual loader using independently retained expected metadata.
        expected=copy.deepcopy(artifact['binding']);weights=artifact['weights']
        inference_from_tensors(segmenter,name[0],name.endswith('STATIC'),weights['basis'],weights,expected)
        payload=dict(schema=artifact['schema'],weights=weights,method_digest=artifact['method_digest'])
        buffer=io.BytesIO();torch.save(payload,buffer);raw=buffer.getvalue();file=name+'.pt'
        out.bytes(file,raw);out.write(name+'.context.json',expected)
        inventory[name]=dict(file=file,training_asset_file_sha256=digest(raw),bytes=len(raw),context_sha256=expected['sha256'],context_file_sha256=digest((out.path/(name+'.context.json')).read_bytes()))
    # No source image/mask/oracle support tensors are deployed; diagnostic arrays
    # remain in a separate private CPU scalar/tensor file, never public by default.
    diagnostics={k:v for k,v in result.items() if k!='models'}
    diagnostics['validation']={k:{x:v[x] for x in ('validation','projection_audit','fit_steps','cal_steps')} for k,v in result['models'].items()}
    buffer=io.BytesIO();torch.save(diagnostics,buffer);out.bytes('diagnostics.private.pt',buffer.getvalue())
    out.write('artifacts.json',inventory);return inventory

def load_artifact(segmenter,name,root,trusted_inventory,max_bytes=256*1024**2):
    if name not in {g+'_'+m for g in 'ABC' for m in ('FULL','STATIC')}:raise ValueError('deployment artifact name')
    row=trusted_inventory[name];root=Path(root)
    if row['file']!=name+'.pt':raise ValueError('deployment asset path')
    # Context and file digests come from a separately trusted release inventory.
    context=json.loads(verified(root/(name+'.context.json'),row['context_file_sha256'],1024**2))
    if context['sha256']!=row['context_sha256']:raise ValueError('release context inventory')
    raw=verified(root/row['file'],row['training_asset_file_sha256'],max_bytes)
    if len(raw)!=row['bytes']:raise ValueError('release asset length')
    package=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    if package.get('schema')!='R7_PREPARED_TENSORS_V2':raise ValueError('release artifact schema')
    weights=package['weights']
    return inference_from_tensors(segmenter,name[0],name.endswith('STATIC'),weights['basis'],weights,context)

def execute(approved,out):
    r=approved['receipt'];cfg=approved['config'];counts=Counter();reader=Reader(approved['manifest'],approved['split'],approved['target'],r['source_root'],counts,cfg['max_asset_bytes'])
    meter=Meter(expected_counts(approved['split']['folds']),cfg['wall_seconds'],lambda row:out.write('phase_'+str(len(meter.records))+'.json',row));segmenter=None;hook=None;first=None;after=None;checkpoint_read=False;started=time.monotonic();before=COUNTS.copy();hash_before=HASH_COST.copy()
    try:
        meter.mark('decode');data=reader.data();meter.check()
        entry=approved['manifest']['checkpoint'];raw=verified(r['checkpoint_path'],entry['sha256'],cfg['max_asset_bytes'],counts)
        if len(raw)!=entry['bytes']:raise ValueError('checkpoint size')
        checkpoint_read=True;meter.mark('model_load');counts.update(checkpoint_deserialization_attempts=1);segmenter=load_model(raw,device=cfg['device']);counts.update(checkpoint_deserializations=1);del raw
        hook=segmenter.register_forward_pre_hook(meter.before_forward)
        source=provenance();source.update(source_binding_status='BOUND',checkpoint_file_sha256=entry['sha256'],source_manifest_sha256=r['manifest']['sha256'],source_split_sha256=r['split']['sha256'])
        result=prepare_tensors(segmenter,data,source_provenance=source,progress=meter.mark);meter.complete();meter.mark('release');release(result,segmenter,out);meter.mark(None)
    except BaseException as exc:
        first=exc
        try:meter.mark(None,validate=False)
        except BaseException as err:after=err
    finally:
        if hook is not None:hook.remove()
        if segmenter is not None:segmenter.close()
        try:
            reader.after_check()
            if checkpoint_read:verified(r['checkpoint_path'],approved['manifest']['checkpoint']['sha256'],cfg['max_asset_bytes'],counts)
            for key in ('manifest','split','target','config'):verified(r[key]['path'],r[key]['sha256'],32*1024**2,counts)
        except BaseException as exc:
            if after is None:after=exc
        summary=dict(status='FAILED' if first or after else 'SOURCE_PREP_COMPLETE_PENDING_REVIEW',counts=dict(COUNTS-before),hash_cost=dict(HASH_COST-hash_before),IO=dict(counts),wall_seconds=time.monotonic()-started,source_after_check='FAILED' if after else 'UNCHANGED',next_scope_authorized=False)
        def error(e):return dict(type=type(e).__name__,message=str(e)[:3000])
        for name,value in ([('first_error.json',error(first))] if first else [])+([('after_check_error.json',error(after))] if after else [])+[('execution.json',summary)]:
            try:out.evidence(name,value)
            except BaseException as exc:
                print('SOURCE_PREP_EVIDENCE_WRITE_ERROR '+json.dumps(dict(file=name,error=error(exc),original_first_error=None if first is None else error(first))),file=sys.stderr,flush=True)
                if first is None:first=exc

    if first:raise first
    if after:raise after
    return summary

def cleanup_owned(record):
    try:stop_owned(record)
    except PermissionError as first:
        # macOS may return EPERM for signal-0 after a group exited. Accept only
        # a reaped child AND an independent process-table proof of absent PGID.
        p=record['process']
        if sys.platform!='darwin':raise
        # SIGTERM may have succeeded just before signal-0 raised EPERM. Wait
        # once for this owned child, without another signal or a process retry.
        try:p.wait(timeout=.5)
        except subprocess.TimeoutExpired:raise first
        rows=subprocess.check_output(['ps','-A','-o','pgid='],text=True)
        if p.pid in {int(x.strip()) for x in rows.splitlines() if x.strip()}:raise
        record.update(cleaned=True,cleanup_probe='reaped_and_ps_group_absent_after_EPERM')

def supervise_one(start,caps,out):
    """One owned CPU child; preserve execution, cleanup and persistence failures."""
    record=None;started=time.monotonic();handlers={};first=None;cleanup_error=None;resource_error=None;evidence_errors=[];attempted=set()
    def error(exc):return dict(type=type(exc).__name__,message=str(exc)[:3000])
    def persist(name,value):
        nonlocal first
        attempted.add(name)
        try:out.evidence(name,value)
        except BaseException as exc:
            evidence_errors.append(dict(file=name,error=error(exc)))
            if first is None:first=exc
            # No disk retry; even a broken stderr cannot replace the first error.
            try:print('SOURCE_PREP_EVIDENCE_WRITE_ERROR '+json.dumps(dict(file=name,error=error(exc),original_first_error=error(first))),file=sys.stderr,flush=True)
            except BaseException:pass
    def interrupted(signum,frame):raise InterruptedError('source supervisor signal '+str(signum))
    try:
        for sig in (signal.SIGTERM,signal.SIGINT):handlers[sig]=signal.signal(sig,interrupted)
        process=start();record=dict(process=process);out.write('process.json',dict(pid=process.pid,pgid=process.pid))
        while process.poll() is None:
            resource_check(out,caps,started)
            time.sleep(.1)
        if process.returncode:raise RuntimeError('source worker nonzero exit '+str(process.returncode))
    except BaseException as exc:first=exc
    finally:
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            if record:cleanup_owned(record)
        except BaseException as exc:
            cleanup_error=exc
            if first is None:first=exc
        finally:
            for sig,old in handlers.items():signal.signal(sig,old)
        # Always audit after exit/cleanup, including nonzero and already-exited
        # children. A resource error never erases an earlier execution failure.
        try:resource_check(out,caps,started)
        except BaseException as exc:
            resource_error=exc
            if first is None:first=exc
        if first:persist('supervisor.first_error.json',error(first))
        if cleanup_error:persist('supervisor.cleanup_error.json',error(cleanup_error))
        if resource_error:persist('supervisor.resource_error.json',error(resource_error))
        if evidence_errors:persist('supervisor.evidence_errors.json',dict(errors=list(evidence_errors)))
        persist('supervisor.json',dict(cleanup_probe=None if record is None else record.get('cleanup_probe','legacy_stop_owned'),exit_code=None if record is None else record['process'].returncode,wall_seconds=time.monotonic()-started,complete=first is None,cleaned=record is not None and record.get('cleaned',False),retry=False))
        # A failure of the last summary is itself a first failure. Attempt the
        # still-unwritten error records once; never retry or overwrite a file.
        if first and 'supervisor.first_error.json' not in attempted:persist('supervisor.first_error.json',error(first))
        if evidence_errors and 'supervisor.evidence_errors.json' not in attempted:persist('supervisor.evidence_errors.json',dict(errors=list(evidence_errors)))
    if first:raise first
    return started

def publish_completion(out,caps,started):
    # No success publication until parent/child/log bytes and terminal writes
    # have been counted. Failed pending evidence remains, never relabeled PASS.
    try:
        resource_check(out,caps,started,reserve=16384)
        completion=json.loads((out.path/'worker/execution.json').read_text())
        if completion['status']!='SOURCE_PREP_COMPLETE_PENDING_REVIEW' or completion['source_after_check']!='UNCHANGED':raise ValueError('worker execution completion absent or invalid')
        out.evidence('completion.pending.json',dict(status='SOURCE_PREP_COMPLETE_PENDING_REVIEW',execution=completion,other_scopes_authorized=False))
        resource_check(out,caps,started,reserve=0)
        if (out.path/'completion.json').exists():raise FileExistsError('completion already published')
        (out.path/'completion.pending.json').rename(out.path/'completion.json')
    except BaseException as first:
        failure=dict(type=type(first).__name__,message=str(first)[:3000],complete=False,retry=False)
        try:out.evidence('completion.error.json',failure)
        except BaseException as exc:
            try:print('SOURCE_PREP_COMPLETION_WRITE_ERROR '+json.dumps(dict(first=failure,evidence_error=str(exc)[:3000])),file=sys.stderr,flush=True)
            except BaseException:pass
        raise

def main():
    # No receipt means disabled before any real source metadata/asset or device.
    path=os.environ.get('SOURCE_PREP_RECEIPT')
    if not path:raise PermissionError('SOURCE_PREP receipt absent; disabled')
    raw=Path(path).read_bytes();receipt=json.loads(raw);approved=preflight(receipt)
    torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    cfg=approved['config'];out=BudgetOutput(receipt['output_dir'],dict(scope='SOURCE_PREP',code_sha=receipt['code_sha'],receipt_sha256=digest(raw)),cfg['output_bytes'])
    out.write('source_split.frozen.json',approved['split']);out.write('preflight.json',approved['audit'])
    from ..b3_runtime import ENTRY
    env=dict(os.environ,RUN_FILE=str(ROOT/'scripts/r7/source_prep.py'),SOURCE_PREP_CHILD='1',SOURCE_PREP_PARENT_OWNER=out.owner,CUDA_VISIBLE_DEVICES='' if cfg['device']=='cpu' else str(cfg['physical_GPU_ids'][0]),PYTHONDONTWRITEBYTECODE='1')
    # Child rechecks all bindings with its own fresh output directory.
    child_receipt=copy.deepcopy(receipt);child_receipt['output_dir']=str(out.path/'worker')
    out.write('child.receipt.json',child_receipt);env['SOURCE_PREP_RECEIPT']=str(out.path/'child.receipt.json')
    def start():
        with (out.path/'worker.log').open('xb') as log:return subprocess.Popen([sys.executable,'-c',ENTRY],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    started=supervise_one(start,cfg,out)
    publish_completion(out,cfg,started)

def child_entry():
    r=json.loads(Path(os.environ['SOURCE_PREP_RECEIPT']).read_bytes());approved=preflight(r);torch.set_num_threads(2);torch.set_default_dtype(torch.float32)
    configure_backend(approved['config'])
    out=BudgetOutput(r['output_dir'],dict(scope='SOURCE_PREP',code_sha=r['code_sha']),approved['config']['output_bytes']-65536)
    return execute(approved,out)
