"""Foreground finite matrix runner. No daemon, polling for approval, or retry."""
import argparse,gc,json,os,subprocess,sys,time,uuid
from pathlib import Path
from .plan import ROOT,DEFAULTS,SCIENCE,REF,GRATA,science,matrix,authorize,registration_digest,digest,stream,allocation,binding,bound
from .assets import checkpoint,target,AssetMismatch
from .evidence import write


def claim(out,job_id):
    p=Path(out)/job_id;p.mkdir(mode=0o700);return p


def charge(out,worker,started,trajectory=None):
    now=time.monotonic()
    v=json.loads((Path(out)/'usage.json').read_text())
    if v['seconds']>=86400 or v['wall_seconds']>=86400:raise TimeoutError('matrix active/wall cap')
    if trajectory is not None and now-trajectory>7200:raise RuntimeError('trajectory cap')


def smoke(state,device,out,identity,backend,checkpoint_io):
    import torch
    from ..b1_host import Host as OldC
    from ..b4_run import capture,Calls
    from ..source_pilot import seed_all
    from ..host_diagnostic import close
    from ..m2_run import deterministic_smoke_pair
    from ..b3_runtime import process_audit
    from .host import Host,ARMS
    sys.path.insert(0,str(ROOT/'tests'));from test_vptta_host import pixels
    calls=Calls();saved=[];ev={};start=time.monotonic()
    try:
        with deterministic_smoke_pair():
            for arm in ('OLD',*ARMS):
                seed_all(20260907);h=OldC('C',state,device) if arm=='OLD' else Host(arm,state,device);calls.attach(h,True)
                if arm=='OLD':process_audit(out,'smoke')
                if arm=='C_PER256':h.visit=256;h.counts['base_adam']=256
                if arm=='C_SENS':
                    h.controller.age=49;h.controller.ema=1e-6;h.controller.best=1e-6
                    observed=h.controller.observe;h.controller.observe=lambda s:observed(1.)
                if arm.startswith('C_PCA_'):
                    # Fixed programmatic statistics, discarded with the test host.
                    g=torch.Generator().manual_seed(42)
                    for bank in h.memory.banks:
                        for v in range(1,17):
                            x=torch.randn(8,32,generator=g);bank.merge(x/x.norm(dim=1,keepdim=True),v)
                    h.visit=16;h.counts['base_adam']=16
                for i in range(2):
                    z,a=h.step(pixels('fundus',i));current=dict(z=z.cpu(),state=capture(h))
                    if arm=='OLD':saved.append(current)
                    elif arm=='C':close(current,saved[i]);close(current['state']['rng'],saved[i]['state']['rng'],exact=True)
                    if arm=='C_SENS' and i==0:
                        if not a['reset_before_current']:raise ValueError('sensitivity fixture recovery')
                        h.controller.observe=observed
                    if arm=='C_PER256' and i==0 and not a['reset_before_current']:raise ValueError('period smoke fixture')
                ev[arm]=dict(parity=arm=='C',resets=getattr(h,'reset_count',0),PCA=None if not getattr(h,'memory',None) else h.memory.audit())
                calls.release();h.finish(state);del h,z,current;gc.collect()
        if calls.counts!=dict(forwards=118,backwards=14,base_adam=14,perturb=0,restore=0):raise ValueError('smoke budget')
        write(out/'smoke.completion.json',dict(binding=identity,status='MECHANICAL_SMOKE_COMPLETE',physical=calls.counts,evidence=ev,backend=backend,checkpoint_io=checkpoint_io,seconds=time.monotonic()-start))
    except Exception as e:write(out/'smoke.failure.json',dict(binding=identity,status='INCOMPLETE',reason=str(e),physical=calls.counts,scope=failure_scope(e)));raise


def failure_scope(error):
    if isinstance(error,AssetMismatch):return 'shared_assets'
    if isinstance(error,(ValueError,AssertionError,PermissionError)):return 'shared_code'
    return 'trajectory'


def current(host,row):
    """The sole current-image path: no mask bytes until host prediction returns."""
    from ..p1_analysis import evaluate
    from ..p1_run import sync
    begin=time.monotonic();pixels,rgb_io=target(row,'image')
    sync();tic=time.monotonic();z,a=host.step(pixels);sync();host_seconds=time.monotonic()-tic
    if z.requires_grad:raise ValueError('prediction must be fixed before label access')
    mask,mask_io=target(row,'mask');metrics=evaluate(z.sigmoid(),mask,'fundus')
    return dict(**a,metrics=metrics,prediction_fixed_before_label=True,host_seconds=host_seconds,pipeline_seconds=time.monotonic()-begin,asset_io=dict(image=rgb_io,mask=mask_io))


def trajectory(job,state,reg,out,worker,start,identity,backend,checkpoint_io):
    import torch
    from .host import Host
    from ..source_pilot import seed_all
    from ..b3_runtime import process_audit
    p=claim(out,job['job_id']);h=None;written=0;t0=time.monotonic()
    try:
        seed_all(20260907);h=Host(job['arm'],state,'cuda:0');torch.cuda.reset_peak_memory_stats();process_audit(p,'run')
        with (p/'records.jsonl').open('x') as log:
            for row in stream(reg,job['order']):
                charge(out,worker,start,t0);a=current(h,row)
                r=dict(binding=identity,arm=job['arm'],order=job['order'],**{k:row[k] for k in ('group_id','sample_id','domain','subset')},**a,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();written+=1
                if written%64==0:
                    if sum(f.stat().st_size for f in Path(out).rglob('*') if f.is_file())>2*1024**3:raise RuntimeError('private output cap')
        h.finish(state);charge(out,worker,start,t0);write(p/'completion.json',dict(binding=identity,status='TRAJECTORY_COMPLETE',records=written,physical=h.counts,backend=backend,checkpoint_io=checkpoint_io,seconds=time.monotonic()-t0))
    except Exception as e:
        write(p/'failure.json',dict(binding=identity,status='INCOMPLETE',reason=str(e),records=written,physical=None if h is None else h.counts,scope=failure_scope(e)));raise
    finally:del h;gc.collect()


def context():
    packet=json.loads(Path(os.environ['RUN_PACKET']).read_text());reg=packet['assets']['registration']
    ids=authorize(packet['authorization'],reg);index=int(os.environ['RUN_WORKER'])
    if [d['index'] for d in packet['devices']]!=ids:raise PermissionError('authorized device list binding')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=packet['devices'][index]['uuid']:raise PermissionError('device UUID binding')
    if packet['binding']['code_sha']!=packet['authorization']['approved_code_sha'] or packet['binding']['science_sha256']!=digest(SCIENCE) or packet['binding']['registration_digest']!=registration_digest(reg):raise PermissionError('packet binding')
    if packet['jobs']!=matrix()['jobs'] or packet['schedule']!=allocation(packet['jobs'],len(ids)):raise PermissionError('frozen schedule binding')
    return packet,reg,index,Path(packet['out'])


def worker():
    import torch
    from ..source_pilot_release import environment
    packet,reg,index,out=context();p=claim(out,'device'+str(index));identity=binding(packet,index)
    try:
        # Wrong content is rejected before environment() queries or initializes CUDA.
        state,io=checkpoint(reg);torch.set_num_threads(2)
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';backend=environment()
        smoke(state,'cuda:0',p,identity,backend,io)
    except Exception as e:
        if not (p/'smoke.failure.json').exists():write(p/'smoke.failure.json',dict(binding=identity,status='INCOMPLETE',reason=str(e),scope=failure_scope(e)))
        raise


def formal_worker():
    import torch
    from ..source_pilot_release import environment
    packet,reg,index,out=context();job=next(j for j in packet['jobs'] if j['job_id']==os.environ['RUN_JOB'])
    if next(a['worker'] for a in packet['schedule']['assignments'] if a['job_id']==job['job_id'])!=index:raise PermissionError('job device rotation binding')
    identity=binding(packet,index,job)
    try:
        p=out/('device'+str(index));proof=json.loads((p/'smoke.completion.json').read_text());bound(proof,binding(packet,index))
        if proof['status']!='MECHANICAL_SMOKE_COMPLETE' or list(p.glob('*.failure.json')) or proof['physical']!=dict(forwards=118,backwards=14,base_adam=14,perturb=0,restore=0):raise PermissionError('smoke evidence gate')
        state,io=checkpoint(reg);torch.set_num_threads(2);os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None);backend=environment()
        trajectory(job,state,reg,out,index,time.monotonic(),identity,backend,io)
    except Exception as e:
        p=out/job['job_id'];p.mkdir(mode=0o700,exist_ok=True)
        if not (p/'failure.json').exists():write(p/'failure.json',dict(binding=identity,status='INCOMPLETE',reason=str(e),scope=failure_scope(e)))
        raise


def launch(auth,assets,out):
    reg=assets['registration'];ids=authorize(auth,reg)
    for order in range(4):stream(reg,order)
    if subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip():raise PermissionError('clean reviewed checkout required')
    for key,sha in (('ctta_dependency_root',REF),('grata_root',GRATA)):
        if subprocess.check_output(['git','rev-parse','HEAD'],cwd=assets[key],text=True).strip()!=sha:raise ValueError('fixed dependency')
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.free','--format=csv,noheader'],text=True)
    mapping={int(row[0]):dict(uuid=row[1].strip(),model=row[2].strip(),free_MiB=int(row[3].split()[0])) for row in [line.split(',') for line in rows.splitlines()]}
    devices=[dict(index=i,**mapping[i]) for i in ids]
    if any(d['free_MiB']<4096 for d in devices):raise RuntimeError('need 4 GiB peak headroom per worker; no waiting/retry')
    out=Path(out).resolve()
    if out.is_relative_to(ROOT):raise ValueError('private output outside code')
    if not out.is_relative_to(Path(assets['private_storage_root']).resolve()):raise ValueError('registered private storage root')
    import tempfile
    fs=os.statvfs(out.parent)
    if fs.f_bavail*fs.f_frsize<3*1024**3:raise RuntimeError('storage capacity')
    with tempfile.TemporaryFile(dir=out.parent) as probe:
        probe.write(b'probe');probe.flush();os.fsync(probe.fileno());probe.seek(0)
        if probe.read()!=b'probe':raise RuntimeError('storage write/read')
    out.mkdir(mode=0o700);write(out/'usage.json',dict(seconds=0.,wall_seconds=0.,active_workers=0))
    jobs=matrix()['jobs'];identity=dict(run_id=uuid.uuid4().hex,code_sha=auth['approved_code_sha'],science_sha256=digest(SCIENCE),registration_digest=registration_digest(reg))
    packet=dict(binding=identity,authorization=auth,assets=assets,out=str(out),devices=devices,jobs=jobs,schedule=allocation(jobs,len(devices)))
    write(out/'packet.private.json',packet)
    write(out/'receipt.json',dict(binding=identity,devices=devices,jobs=jobs,schedule=packet['schedule'],formal_budget=science()['formal_budget'],smoke_budget=science()['per_gpu_smoke_budget'],mixed_device_models=len({d['model'] for d in devices})>1))
    from ..b3_runtime import ENTRY
    from .supervise import supervise
    def start_process(spec,log):
        i=spec['worker'];env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(ROOT/'src'),assets.get('dependency_pythonpath','')]),DPA_CTTA_BASE_ROOT=assets['ctta_dependency_root'],DPA_GRATA_ROOT=assets['grata_root'],CUDA_VISIBLE_DEVICES=devices[i]['uuid'],RUN_FILE=str(ROOT/'scripts/run_r1_matrix.py'),RUN_MODE='worker' if spec['phase']=='smoke' else 'formal',RUN_WORKER=str(i),RUN_JOB=spec['key'],RUN_PACKET=str(out/'packet.private.json'))
        return subprocess.Popen([sys.executable,'-c',ENTRY],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    supervise(out,packet,start_process)
    from .analyze import recompute
    recompute(out,reg)


def main():
    os.umask(0o077)
    if os.environ.get('RUN_MODE')=='worker':return worker()
    if os.environ.get('RUN_MODE')=='formal':return formal_worker()
    parser=argparse.ArgumentParser();parser.add_argument('--run',action='store_true');parser.add_argument('--authorization',type=Path,default=DEFAULTS);parser.add_argument('--assets',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if not args.run:print(json.dumps(matrix(),indent=2));return
    auth=json.loads(args.authorization.read_text())
    if auth.get('enabled') is not True:raise PermissionError('execution disabled')
    if args.assets is None or args.output is None:raise ValueError('explicit assets/output required')
    launch(auth,json.loads(args.assets.read_text()),args.output)
