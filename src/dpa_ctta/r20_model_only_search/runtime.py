"""Finite, detached workers. Scheduler and CPU scorers cannot feed online state."""
import concurrent.futures,hashlib,json,os,signal,sys,threading,time,traceback,subprocess,shutil
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from ..r10_12h_core.run import read,save,sha
from ..r16_evidence_correction.run import process_identity
from ..r9_current_first.storage import lease
from ..r9_current_first.physical import Meter
from ..r9_current_first.assets import gpu_policy,available_memory
from ..r8_ba.journal import TargetJournal,_digest
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader,image_records
from ..r19_model_only.runtime import weights,image
from ..r19_model_only.method import equal,snapshot
from .method import Host

ID='R20_MODEL_ONLY_BUDGETED_SEARCH';WIDTH=65536;SOFT_WIDTH=2097152

def config():
    c=read(os.environ['RUN_CONFIG'])
    if c['experiment_id']!=ID or sha(c)!=os.environ['RUN_CONFIG_SHA']:raise ValueError('configuration identity')
    return c

def amounts(c):
    root=Path(c['output_root']);done=set();total=0
    for p in (root/'attempts').glob('*.json'):
        r=read(p);done.add((r['phase'],r['attempt']));total+=r['cost'].get('gpu_seconds',0)
    for p in (root/'processes').glob('*.json'):
        r=read(p)
        if r.get('active') and not r['phase'].startswith('score_') and (r['phase'],r['attempt']) not in done:total+=time.time()-r['started']
    return total,{}

def disk_bytes(root):
    total=0
    for directory,_,files in os.walk(root):
        for f in files:
            try:total+=(Path(directory)/f).stat().st_size
            except FileNotFoundError:pass
    return total

def filesystem_guard(c):
    root=Path(c['output_root']).resolve();nas=root.parent;checkpoint=Path(c['checkpoint_path']).resolve();code=[Path(os.environ[k]).resolve() for k in ('DPA_CTTA_BASE_ROOT','DPA_GRATA_ROOT','PYTHONPATH')];runtime=Path(sys.prefix).resolve();current={'image':None};opened=set();rejections=[]
    def audit(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if not p.is_relative_to(nas):
            if p.suffix.lower() in ('.pt','.pth','.safetensors','.npz','.npy','.png','.jpg','.jpeg') and not (p.is_relative_to(runtime) or any(p.is_relative_to(a) for a in code)):
                rejections.append(str(p));raise PermissionError('unregistered external model/data asset')
            return
        allow=p==checkpoint or p==current['image'] or p.is_relative_to(root) or p.is_relative_to(runtime) or any(p.is_relative_to(a) for a in code)
        if any(p.is_relative_to(root/d) for d in ('scorer','scores','stages','reports')):allow=False
        if not allow:rejections.append(str(p));raise PermissionError('model-only current-image access boundary')
        opened.add(str(p))
    sys.addaudithook(audit)
    # Exercise source/cache, target-label and outer-score rejection without opening assets.
    for p in [nas/'data/Fundus/source-denial-probe',root/'scorer/FULL_o0.json',root/'scores/forbidden.json']:
        try:p.read_bytes()
        except PermissionError:continue
        raise ValueError('asset boundary failed closed')
    return current,opened,rejections

class Guard:
    def __init__(self,c,phase):
        self.c=c;self.root=Path(c['output_root']);self.phase=phase;self.deadline=float(os.environ['RUN_DEADLINE']);self.last=0;self.disk_last=0;self.meter=None;self.extra=Counter();self.size=None
    def observe(self,cost=None):
        if time.time()>self.deadline-10:raise TimeoutError('registered task deadline')
        if time.monotonic()-self.last<10:return
        cap = self.c['origin']['gpu_worker_cap_seconds']
        if cap is not None and not self.phase.startswith('score_') and amounts(self.c)[0]>=cap-30:raise TimeoutError('cumulative GPU-worker cap')
        if time.monotonic()-self.disk_last>300:
            self.size=disk_bytes(self.root);self.disk_last=time.monotonic()
            if self.size>self.c['origin']['disk_cap_bytes']:raise OSError('campaign disk cap')
        save(self.root/'live'/(self.phase+'.json'),dict(at=time.time(),pid=os.getpid(),cost=cost or {},extra=dict(self.extra),disk_bytes=self.size))
        self.last=time.monotonic()
    def __call__(self):self.observe(self.meter.cost if self.meter else {})

def attach(meter,h):
    for m in (h.native.model,h.teacher,h.f0):
        if m is not None:meter.attach(m)

def profile(c,guard,permit,job):
    root=Path(c['output_root']);rows=read(root/'private/SCREEN_o0.json')[:12];reader=TargetReader(c['target_root'],256*1024**2,'image');start=time.time();h=Host(weights(c),job['candidate'],c['native_seed'],sha(job));attach(guard.meter,h);init=time.time()-start;times=[]
    reference=None;checks=[]
    if job['candidate']['id']=='G':
        import random
        from ..b1_host import Host as Native
        random.seed(c['native_seed']);np.random.seed(c['native_seed']);torch.manual_seed(c['native_seed']);reference=Native('G',state=weights(c),device='cuda:0');guard.meter.attach(reference.model)
    torch.cuda.reset_peak_memory_stats()
    for i,row in enumerate(rows):
        guard();x=image(reader,row,permit,guard)
        if reference is not None:rz,_=reference.step(x)
        start=time.perf_counter();z,_=h.step(x);torch.cuda.synchronize();times.append(time.perf_counter()-start)
        if reference is not None:
            a,b=h.snapshot(),snapshot(reference);ok=torch.equal(z,rz.cpu()) and all(equal(a[k],b[k]) for k in ('parameters','gradients','adam','grata','native_rng'));checks.append(ok)
            if not ok:raise ValueError('real native G wrapper mismatch')
    h.check_frozen(True);peak=torch.cuda.max_memory_reserved();h.close()
    if reference is not None:
        for handle in reference.handles:handle.remove()
    result=dict(status='PASS',candidate=job['candidate']['id'],family=job['candidate']['family'],warmup_arrivals=4,timed_arrivals=8,seconds=times[4:],initialization_seconds=init,peak_reserved_bytes=peak,native_parity=checks,source_reads=0,label_reads=0)
    save(root/'profiles'/(job['candidate']['id']+'.json'),result);return result

def online(c,guard,permit,job):
    root=Path(c['output_root']);kind='SCREEN' if job['stream']=='screen' else 'ONLINE';rows=read(root/'private'/f'{kind}_o{job["order"]}.json')
    if rows_sha(rows)!=c['manifests'][kind][job['order']]['sha256']:raise ValueError('online manifest identity')
    h=Host(weights(c),job['candidate'],job['seed'],sha(dict(config=sha(c),job={k:v for k,v in job.items() if k!='recovery'})));attach(guard.meter,h)
    dest=root/'target'/job['id'];journal=TargetJournal(dest,h,job['id'],rows_sha(rows),WIDTH);reader=TargetReader(c['target_root'],256*1024**2,'image')
    if job.get('recovery'):journal.recover_once(job['recovery'])
    else:journal.create()
    soft=None
    if job.get('soft'):
        soft=(dest/'probabilities.f32').open('r+b' if job.get('recovery') else 'xb');soft.truncate(h.visits*SOFT_WIDTH);soft.seek(h.visits*SOFT_WIDTH)
    try:
        for row in rows[h.visits:]:
            guard();x=image(reader,row,permit,guard);start=time.perf_counter()
            try:z,t=h.step(x)
            except BaseException as e:journal.record_failed_call(h.visits+1,guard.meter.cost.copy(),e);raise
            p=z.float().sigmoid().numpy()[0]
            if soft:soft.write(p.astype('<f4').tobytes());soft.flush()
            t['seconds']=time.perf_counter()-start;t['counts']=guard.meter.cost.copy();journal.append(np.packbits(p>=.5).tobytes(),t)
        if soft:soft.flush();os.fsync(soft.fileno())
        journal.checkpoint();result=journal.complete(len(rows))
        if soft:save(dest/'soft_complete.json',dict(bytes=(dest/'probabilities.f32').stat().st_size,dtype='float32 sigmoid',visits=len(rows)))
        return result
    finally:
        if soft:soft.close()
        h.close()

def worker_peak(c, phase, job):
    if not c.get('use_profiled_worker_memory') or phase.startswith('profile_'):
        return int(job.get('peak_bytes', 5*1024**3))
    candidate = job['candidate']['id']
    measured = read(Path(c['output_root'])/'profiles'/(candidate+'.json'))
    peak = measured.get('peak_reserved_bytes')
    if (measured.get('status') != 'PASS' or measured.get('candidate') != candidate
            or type(peak) is not int or peak <= 0):
        raise ValueError('valid same-round profile peak required for worker admission')
    return peak


def worker():
    c=config();root=Path(c['output_root']);phase=os.environ['RUN_PHASE'];start=float(os.environ['RUN_STARTED']);guard=Guard(c,phase);cpu=phase.startswith('score_');meter=None;failure=None;result=None;opened=set();rejections=[]
    torch.set_num_threads(2);torch.set_num_interop_threads(2)
    timer=threading.Timer(max(.1,guard.deadline-time.time()),lambda:os.killpg(os.getpgrp(),signal.SIGTERM));timer.daemon=True;timer.start()
    try:
        job=json.loads(os.environ['RUN_JOB'])
        if cpu:
            if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('scorer must be CPU-only')
            from .score import score_job
            result=score_job(c,guard,job)
        else:
            assignment=json.loads(os.environ['RUN_ASSIGNMENT']);gpu_policy(assignment);available_memory(assignment,worker_peak(c,phase,job))
            if subprocess.check_output(['findmnt','-n','-o','FSTYPE','-T',str(root)],text=True).strip() not in ('nfs','nfs4'):raise OSError('required NAS mount absent')
            disk=shutil.disk_usage(root)
            if disk.free<c['origin']['disk_cap_bytes'] or disk.free/disk.total<.2:raise OSError('NAS free reserve')
            permit,opened,rejections=filesystem_guard(c)
            with lease(root/'leases'/f'gpu{assignment["physical_id"]}',dict(experiment_id=ID,GPU=assignment)),Meter(None,guard.observe) as meter:
                guard.meter=meter;result=profile(c,guard,permit,job) if phase.startswith('profile_') else online(c,guard,permit,job)
    except BaseException as e:failure=dict(reason=str(e),error_type=type(e).__name__,errno=getattr(e,'errno',None));traceback.print_exc()
    finally:timer.cancel()
    cost=meter.cost.copy() if meter else {};cost.update(guard.extra);cost['gpu_seconds']=0 if cpu else time.time()-start;cost['cpu_seconds']=time.time()-start if cpu else 0
    status='COMPLETE' if failure is None else ('FAILED_NUMERICAL' if failure['error_type'] in ('ValueError','FloatingPointError') else 'INCOMPLETE')
    row=dict(status=status,phase=phase,attempt=int(os.environ.get('RUN_ATTEMPT',0)),started=start,ended=time.time(),wall_seconds=time.time()-start,cost=cost,failure=failure,result=result,code_sha=c['code_sha'],config_sha256=sha(c))
    save(root/'attempts'/f'{phase}.{row["attempt"]}.json',row)
    if not cpu:save(root/'private'/f'{phase}-access-audit.json',dict(allowed_opens=sorted(opened),deliberate_rejections=rejections,source_reads=0,target_label_reads=0))
    return row

def run_task(c,phase,assignment,seconds,job,attempt=0):
    root=Path(c['output_root']);start=time.time();path=root/'attempts'/f'{phase}.{attempt}.json';cpu=phase.startswith('score_')
    if path.exists():raise ValueError('task already attempted')
    deadline=min(start+seconds,c['origin']['absolute_deadline_epoch'] if cpu else c['origin']['online_deadline_epoch'])
    env=dict(os.environ,RUN_MODE='worker',RUN_PHASE=phase,RUN_CONFIG_SHA=sha(c),RUN_STARTED=str(start),RUN_DEADLINE=str(deadline),RUN_ASSIGNMENT=json.dumps(assignment),RUN_ATTEMPT=str(attempt),RUN_JOB=json.dumps(job),CUDA_VISIBLE_DEVICES='' if cpu else str(assignment['physical_id']))
    with (root/'logs'/f'{phase}.{attempt}.log').open('ab') as f:
        p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=env,start_new_session=True,stdout=f,stderr=subprocess.STDOUT);ident=process_identity(p);ident.update(phase=phase,assignment=assignment,started=start,deadline=deadline,attempt=attempt);save(root/'processes'/f'{phase}.json',ident)
        try:p.wait(timeout=max(1,deadline-time.time()+5))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
        ident.update(active=False,ended=time.time(),exit_code=p.returncode);save(root/'processes'/f'{phase}.json',ident)
    if path.exists():return read(path)
    row=dict(status='INCOMPLETE',phase=phase,attempt=attempt,started=start,ended=time.time(),wall_seconds=time.time()-start,cost=dict(gpu_seconds=0 if cpu else time.time()-start,cpu_seconds=time.time()-start if cpu else 0),failure=dict(reason='exit without receipt',error_type='MissingReceipt',exit_code=p.returncode),code_sha=c['code_sha']);save(path,row);return row

def ledger(c,state):
    root=Path(c['output_root']);rs=[read(p) for p in sorted((root/'attempts').glob('*.json'))];x=dict(experiment_id=ID,status=state['status'],T0=c['origin']['T0'],wall_seconds=time.time()-c['origin']['T0_epoch'],gpu_worker_seconds=sum(r['cost'].get('gpu_seconds',0) for r in rs),cpu_worker_seconds=sum(r['cost'].get('cpu_seconds',0) for r in rs),gpu_cap_seconds=c['origin']['gpu_worker_cap_seconds'],disk_bytes=disk_bytes(root),attempts=rs);save(root/'RESOURCE_LEDGER.json',x);return x

def watch():
    c=config();root=Path(c['output_root']);p=subprocess.Popen([sys.executable,os.environ['RUN_ENTRY']],env=dict(os.environ,RUN_MODE='supervise'),start_new_session=True)
    save(root/'watchdog.json',dict(pid=os.getpid(),supervisor_pid=p.pid,absolute_deadline=c['origin']['absolute_deadline_epoch']))
    try:
        try:p.wait(timeout=max(1,c['origin']['absolute_deadline_epoch']-time.time()))
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM)
    finally:
        for f in (root/'processes').glob('*.json'):
            a=read(f);stat=Path(f'/proc/{a["pid"]}/stat')
            if a['active'] and stat.exists() and stat.read_text().split()[21]==a['start_ticks']:
                try:os.killpg(a['pgid'],signal.SIGTERM)
                except ProcessLookupError:pass

def main():
    mode=os.environ['RUN_MODE']
    if mode=='worker':r=worker();sys.exit(0 if r['status']=='COMPLETE' else 1)
    elif mode=='supervise':
        from .schedule import supervise
        supervise()
    elif mode=='watch':watch()
    else:raise ValueError('unknown execution mode')
