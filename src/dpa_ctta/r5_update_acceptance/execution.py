"""Future finite A or B_NEW run; never transitions automatically from A to B."""
import gc,json,os,subprocess,sys,time,uuid
from pathlib import Path
from ..r1.plan import binding,bound,REF,GRATA
from ..r1.assets import checkpoint
from ..r1.evidence import write,output_bytes
from ..r1.run import claim,failure_scope
from ..r3.execution import worker_environment,check_worker_environment,backend_policy
from .plan import ROOT,SCIENCE_SHA,authorize,science,stream,stream_summary,registration_digest,fingerprint,matrix,allocation
from .rule import PHYSICAL
from .evaluation import current


def smoke(state,device,out,identity,recipe,p_accept=None):
    import torch
    from ..r1.host import Host as Old
    from ..source_pilot import seed_all
    from ..b4_run import capture
    from ..host_diagnostic import close
    from ..m2_run import deterministic_smoke_pair
    from ..b3_runtime import process_audit
    from .host import Host
    sys.path.insert(0,str(ROOT/'tests'));from test_vptta_host import pixels
    saved=[];physical={k:0 for k in PHYSICAL};evidence={}
    arms=('OLD','C') if recipe['recipe']=='PROCEDURAL_OLD_C4_DIAG_C4_V1' else ('OLD','C','C_HALF','C_RANDOM','C_VERIFY')
    try:
        with deterministic_smoke_pair():
            for arm in arms:
                h=core=z=value=None
                try:
                    seed_all(20260907);h=Old('C',state,device) if arm=='OLD' else Host(arm,state,device,p_accept=p_accept if arm=='C_RANDOM' else None)
                    core=h if arm=='OLD' else h.core
                    if arm=='OLD' and torch.device(device).type=='cuda':process_audit(out,'smoke')
                    for i in recipe['pixel_indices']:
                        z,t=h.step(pixels('fundus',i))
                        value=dict(logits=z.detach().cpu().clone(),state=capture(core))
                        if arm=='OLD':saved.append(value)
                        elif arm!='C_HALF':close(value,saved[i],exact=torch.device(device).type=='cpu');close(core.rng,saved[i]['state']['rng'],exact=True)
                        if arm!='OLD':h.take_evaluation().clear()
                    evidence[arm]=dict(visits=4,counts=core.counts.copy(),parity=arm not in ('OLD','C_HALF'))
                    h.finish(state)
                finally:
                    # Each host's live cumulative counters are charged once, including a failed step.
                    if core is not None:
                        for public,internal in (('network_forwards','forwards'),('loss_backward_calls','backwards'),('adam_calls','base_adam')):physical[public]+=core.counts[internal]
                        for owner in (core,h) if h is not core else (core,):
                            for handle in owner.handles:handle.remove()
                    h=core=z=value=None;gc.collect()
        if physical!={k:recipe[k] for k in PHYSICAL}:raise ValueError('smoke physical budget')
        write(out/'smoke.completion.json',dict(binding=identity,status='MECHANICAL_SMOKE_COMPLETE',C_parity_valid=True,recipe=recipe,physical=physical,evidence=evidence,backend=backend_policy()))
    except BaseException as exc:
        try:
            write(out/'smoke.failure.json',dict(binding=identity,status='INCOMPLETE',physical=physical,
                physical_accounting='OBSERVED_HOOK_COUNTS_LOWER_BOUND_ON_FAILURE',
                uncertainty='Live forward/gradient/Adam-post hooks; work interrupted before its hook or during construction is not observable. No VJP exists in this smoke.',
                reason=str(exc),scope=failure_scope(exc)))
        except FileExistsError:pass  # Preserve the first failure; never replace it or retry compute.
        except OSError as logging_error:print('smoke failure evidence write failed: '+str(logging_error),file=sys.stderr)
        raise


def trajectory(job,state,reg,out,identity,caps,p_accept=None):
    import torch
    from ..source_pilot import seed_all
    from ..b3_runtime import process_audit
    from .host import Host
    p=claim(out,job['job_id']);h=None;written=0;start=time.monotonic()
    try:
        seed_all(20260907);h=Host(job['arm'],state,'cuda:0',p_accept=p_accept if job['arm']=='C_RANDOM' else None);process_audit(p,'run')
        with (p/'unlabeled.jsonl').open('x') as unlabel,(p/'evaluation.jsonl').open('x') as labels:
            for t,entry in enumerate(stream(reg,job['order']),1):
                if time.monotonic()-start>caps['trajectory_seconds']:raise RuntimeError('trajectory cap')
                trace,evaluation=current(h,entry)
                ident=dict(binding=identity,visit=t,**{k:entry[k] for k in ('sample_id','group_id','domain','subset')})
                unlabel.write(json.dumps(dict(**ident,trace=trace),allow_nan=False)+'\n');unlabel.flush()
                labels.write(json.dumps(dict(**ident,evaluation=evaluation),allow_nan=False)+'\n');labels.flush();written+=1
                if written%64==0 and output_bytes(out)>caps['bytes']:raise RuntimeError('output cap')
        h.finish(state)
        if written!=job['records'] or time.monotonic()-start>caps['trajectory_seconds']:raise ValueError('trajectory coverage/time')
        write(p/'completion.json',dict(binding=identity,status='TRAJECTORY_COMPLETE',records=written,physical=h.physical,totals=h.totals,seconds=time.monotonic()-start))
    except BaseException as exc:
        write(p/'failure.json',dict(binding=identity,status='INCOMPLETE',records=written,physical=None if h is None else h.core.counts,reason=str(exc),scope=failure_scope(exc)));raise
    finally:del h;gc.collect()


def context():
    packet=json.loads(Path(os.environ['RUN_PACKET']).read_text());reg=packet['assets']['registration'];auth=packet['authorization']
    ids=authorize(auth,reg);i=int(os.environ['RUN_WORKER']);scope=auth['scope']
    want=dict(code_sha=auth['approved_code_sha'],science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'],c_fingerprint=fingerprint(),scope=scope)
    if any(packet['binding'].get(k)!=v for k,v in want.items()) or [d['index'] for d in packet['devices']]!=ids or not 0<=i<len(ids):raise PermissionError('worker binding')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=packet['devices'][i]['uuid'] or packet['jobs']!=matrix(scope) or packet['schedule']!=allocation(packet['jobs'],len(ids)):raise PermissionError('worker device/schedule')
    if packet['caps']!=auth['caps'] or packet['smoke_recipe']!=auth['smoke_recipe'] or packet['calibration']!=auth.get('calibration') or packet['reuse_A']!=auth.get('reuse_A'):raise PermissionError('worker recipe')
    return packet,reg,i,Path(packet['out'])


def worker():
    packet,reg,i,out=context();check_worker_environment(os.environ['RUN_MODE'],os.environ)
    import torch
    torch.set_num_threads(2);ident=binding(packet,i)
    p_accept=None if packet['calibration'] is None else packet['calibration']['p_accept']
    if os.environ['RUN_MODE']=='smoke':
        p=claim(out,'device'+str(i))
        try:state,io=checkpoint(reg);smoke(state,'cuda:0',p,ident,packet['smoke_recipe'],p_accept)
        except BaseException as exc:
            if not (p/'smoke.failure.json').exists():write(p/'smoke.failure.json',dict(binding=ident,status='INCOMPLETE',reason=str(exc),scope=failure_scope(exc)))
            raise
    elif os.environ['RUN_MODE']=='formal':
        job=next(j for j in packet['jobs'] if j['job_id']==os.environ['RUN_JOB'])
        if next(a['worker'] for a in packet['schedule']['assignments'] if a['job_id']==job['job_id'])!=i:raise PermissionError('job assignment')
        proof=json.loads((out/f'device{i}/smoke.completion.json').read_text());bound(proof,ident)
        if proof['status']!='MECHANICAL_SMOKE_COMPLETE' or proof['C_parity_valid'] is not True or proof['recipe']!=packet['smoke_recipe'] or proof['physical']!={k:packet['smoke_recipe'][k] for k in PHYSICAL}:raise PermissionError('smoke gate')
        state,io=checkpoint(reg);trajectory(job,state,reg,out,binding(packet,i,job),packet['caps'],p_accept)
    else:raise PermissionError('worker phase')


def launch(auth,assets,output):
    reg=assets['registration'];ids=authorize(auth,reg)
    for order in range(5):stream(reg,order)
    if auth['scope']=='B_NEW':
        from .analyze import reuse_A
        reuse_A(auth['reuse_A'],reg,auth['calibration'])
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('clean exact execution checkout')
    for key,sha in (('ctta_dependency_root',REF),('grata_root',GRATA)):
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=assets[key],text=True).strip()!=sha:raise ValueError('fixed dependency')
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader'],text=True)
    mapping={int(r[0]):dict(uuid=r[1].strip(),model=r[2].strip(),free_MiB=int(r[3].split()[0])) for r in [s.split(',') for s in inventory.splitlines()]}
    devices=[dict(index=i,**mapping[i]) for i in ids]
    if any(d['free_MiB']<4096 for d in devices):raise RuntimeError('insufficient headroom; no wait')
    out=Path(output).resolve();root=Path(assets['private_storage_root']).resolve();caps=auth['caps']
    if out.is_relative_to(ROOT) or not out.is_relative_to(root):raise ValueError('new private output path')
    fs=os.statvfs(out.parent)
    if fs.f_bavail*fs.f_frsize<caps['bytes']+1024**3:raise RuntimeError('storage cap')
    import tempfile
    with tempfile.TemporaryFile(dir=out.parent) as f:
        f.write(b'probe');f.flush();os.fsync(f.fileno());f.seek(0)
        if f.read()!=b'probe':raise RuntimeError('storage probe')
    out.mkdir(mode=0o700);jobs=matrix(auth['scope'])
    ident=dict(run_id=uuid.uuid4().hex,code_sha=auth['approved_code_sha'],science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'],c_fingerprint=fingerprint(),scope=auth['scope'])
    packet=dict(binding=ident,authorization=auth,assets=assets,out=str(out),devices=devices,jobs=jobs,schedule=allocation(jobs,len(ids)),caps=caps,smoke_recipe=auth['smoke_recipe'],calibration=auth.get('calibration'),reuse_A=auth.get('reuse_A'))
    write(out/'R5_SCOPE.json',dict(schema='R5_UPDATE_ACCEPTANCE_V1',scope=auth['scope'],run_id=ident['run_id']))
    write(out/'packet.private.json',packet);write(out/'receipt.json',{k:v for k,v in packet.items() if k not in ('assets','authorization','out')})
    from ..b3_runtime import ENTRY
    from ..r1.supervise import supervise
    def start(spec,log):
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(ROOT/'src'),assets.get('dependency_pythonpath','')]),DPA_CTTA_BASE_ROOT=assets['ctta_dependency_root'],DPA_GRATA_ROOT=assets['grata_root'],CUDA_VISIBLE_DEVICES=devices[spec['worker']]['uuid'],RUN_FILE=str(ROOT/'scripts/run_r5.py'),RUN_MODE=spec['phase'],RUN_WORKER=str(spec['worker']),RUN_JOB=spec['key'],RUN_PACKET=str(out/'packet.private.json'))
        return subprocess.Popen([sys.executable,'-c',ENTRY],cwd=ROOT,env=worker_environment(env,spec['phase']),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    supervise(out,packet,start,caps=caps)
    from .analyze import recompute
    return recompute(out,reg)
