"""Foreground finite matrix runner. No daemon, polling for approval, or retry."""
import argparse,gc,json,os,subprocess,sys,time
from pathlib import Path
from .plan import ROOT,DEFAULTS,SCIENCE,REF,GRATA,science,matrix,authorize,registration_digest,digest,stream


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def claim(out,job_id):
    p=Path(out)/job_id;p.mkdir(mode=0o700);return p


def charge(out,worker,started,trajectory=None):
    import fcntl
    now=time.monotonic()
    with (Path(out)/'usage.json').open('r+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);v=json.load(f);old=v['last'].get(str(worker),started);v['seconds']+=now-old;v['last'][str(worker)]=now
        f.seek(0);json.dump(v,f);f.truncate()
    if v['seconds']>86400 or time.time()-v['wall_start']>86400:raise RuntimeError('matrix active/wall cap')
    if trajectory is not None and now-trajectory>7200:raise RuntimeError('trajectory cap')


def smoke(state,device,out):
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
        write(out/'smoke.completion.json',dict(status='MECHANICAL_SMOKE_COMPLETE',physical=calls.counts,evidence=ev,seconds=time.monotonic()-start))
    except Exception as e:write(out/'smoke.failure.json',dict(status='INCOMPLETE',reason=str(e),physical=calls.counts));raise


def trajectory(job,state,reg,out,worker,start):
    import torch
    from .host import Host
    from ..source_pilot import seed_all
    from ..source_io import read_pixels,read_mask
    from ..p1_analysis import evaluate
    from ..p1_run import sync
    from ..b3_runtime import process_audit
    p=claim(out,job['job_id']);h=None;written=0;t0=time.monotonic()
    try:
        seed_all(20260907);h=Host(job['arm'],state,'cuda:0');torch.cuda.reset_peak_memory_stats();process_audit(p,'run')
        with (p/'records.jsonl').open('x') as log:
            for row in stream(reg,job['order']):
                charge(out,worker,start,t0);begin=time.monotonic();pixels=read_pixels(row['image_path'],'fundus',row['image_size']);sync();tic=time.monotonic();z,a=h.step(pixels);sync();host_seconds=time.monotonic()-tic
                mask=read_mask(row['mask_path'],'fundus',row['image_size']);metrics=evaluate(z.sigmoid(),mask,'fundus')
                r=dict(arm=job['arm'],order=job['order'],**{k:row[k] for k in ('group_id','sample_id','domain','subset')},**a,metrics=metrics,prediction_fixed_before_label=True,host_seconds=host_seconds,pipeline_seconds=time.monotonic()-begin,peak_allocated_bytes=torch.cuda.max_memory_allocated())
                log.write(json.dumps(r,allow_nan=False)+'\n');log.flush();written+=1;del z,mask,pixels
                if written%64==0:
                    if sum(f.stat().st_size for f in Path(out).rglob('*') if f.is_file())>2*1024**3:raise RuntimeError('private output cap')
        h.finish(state);charge(out,worker,start,t0);write(p/'completion.json',dict(status='TRAJECTORY_COMPLETE',records=written,physical=h.counts,seconds=time.monotonic()-t0))
    except Exception as e:
        write(p/'failure.json',dict(status='INCOMPLETE',reason=str(e),records=written,physical=None if h is None else h.counts));raise
    finally:del h;gc.collect()


def worker():
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    from ..source_pilot_release import environment
    packet=json.loads(Path(os.environ['RUN_PACKET']).read_text());assets=packet['assets'];reg=assets['registration'];authorize(packet['authorization'],reg)
    torch.set_num_threads(2);environment();out=Path(packet['out']);index=int(os.environ['RUN_WORKER']);entry=packet['devices'][index]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=entry['uuid']:raise PermissionError('device UUID binding')
    p=claim(out,'device'+str(index));start=time.monotonic()
    # The only weight asset: no source data/proxy/history traversal.
    cp=Path(reg['checkpoint']['path']);st=cp.stat()
    if st.st_size!=reg['checkpoint']['bytes'] or st.st_mtime_ns!=reg['checkpoint']['mtime_ns']:raise ValueError('checkpoint metadata changed')
    state=torch.load(cp,map_location='cpu',weights_only=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';smoke(state,'cuda:0',p);os.environ.pop('CUBLAS_WORKSPACE_CONFIG',None);charge(out,index,start);torch.cuda.empty_cache()
    # Formal backend lives in a fresh process, never inherits smoke kernel scope.
    env=os.environ.copy();env['RUN_MODE']='formal';run_child(env)


def run_child(env):
    from ..b3_runtime import ENTRY
    rc=subprocess.call([sys.executable,'-c',ENTRY],cwd=ROOT,env=env,stdin=subprocess.DEVNULL)
    if rc:raise RuntimeError('child failed: '+str(rc))


def formal_worker():
    import torch
    from ..source_pilot_release import environment
    packet=json.loads(Path(os.environ['RUN_PACKET']).read_text());reg=packet['assets']['registration'];authorize(packet['authorization'],reg);torch.set_num_threads(2);environment()
    index=int(os.environ['RUN_WORKER']);entry=packet['devices'][index]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=entry['uuid']:raise PermissionError('device binding')
    out=Path(packet['out']);proof=json.loads((out/('device'+str(index))/'smoke.completion.json').read_text())
    if proof['status']!='MECHANICAL_SMOKE_COMPLETE':raise PermissionError('smoke gate')
    state=torch.load(reg['checkpoint']['path'],map_location='cpu',weights_only=True);start=time.monotonic()
    for job in matrix()['jobs'][index::len(packet['devices'])]:trajectory(job,state,reg,out,index,start)


def launch(auth,assets,out):
    reg=assets['registration'];ids=authorize(auth,reg)
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
    out.mkdir(mode=0o700);write(out/'usage.json',dict(seconds=0.,last={},wall_start=time.time()))
    packet=dict(authorization=auth,assets=assets,out=str(out),devices=devices);write(out/'packet.private.json',packet)
    processes=[]
    from ..b3_runtime import ENTRY
    for i,d in enumerate(devices):
        env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join([str(ROOT/'src'),assets.get('dependency_pythonpath','')]),DPA_CTTA_BASE_ROOT=assets['ctta_dependency_root'],DPA_GRATA_ROOT=assets['grata_root'],CUDA_VISIBLE_DEVICES=d['uuid'],RUN_FILE=str(ROOT/'scripts/run_r1_matrix.py'),RUN_MODE='worker',RUN_WORKER=str(i),RUN_PACKET=str(out/'packet.private.json'))
        with (out/f'worker{i}.log').open('x') as f:processes.append(subprocess.Popen([sys.executable,'-c',ENTRY],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT))
    codes=[p.wait() for p in processes];write(out/'matrix.processes.json',dict(exit_codes=codes,status='INCOMPLETE' if any(codes) else 'COMPUTE_COMPLETE'))
    if any(codes):raise RuntimeError('failed matrix; no retry or automatic repair')
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
