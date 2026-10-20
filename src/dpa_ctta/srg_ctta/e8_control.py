"""Independent elapsed-time and allocated-storage guard; owns only registered identities."""
import os,time,stat,signal,subprocess
from pathlib import Path
from .signal_common import read,save,identity

def used(root):
    total=0
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in files:
            try:s=os.lstat(Path(directory)/name)
            except FileNotFoundError:continue
            if stat.S_ISREG(s.st_mode):total+=s.st_size
    return total

def stop_registered(root):
    import ctypes
    libc=ctypes.CDLL(None,use_errno=True)
    handles=[]
    for p in (root/'processes').glob('*.json'):
        r=read(p)
        if r.get('active') and identity(r['pid'])==r['start_ticks']:
            fd=libc.syscall(434,r['pid'],0)
            if fd<0:continue
            if identity(r['pid'])==r['start_ticks']:handles.append(fd)
            else:os.close(fd)
    for sig in (signal.SIGTERM,signal.SIGKILL):
        for fd in handles:libc.syscall(424,fd,int(sig),0,0)
        if sig==signal.SIGTERM:time.sleep(5)
    for fd in handles:os.close(fd)

def guard(c):
    root=Path(c['output_root']);save(root/'GUARD_IDENTITY.private.json',dict(pid=os.getpid(),start_ticks=identity(os.getpid())))
    while not (root/'FINAL.json').exists():
        origin=read(root/'EXPERIMENT_ORIGIN.private.json')
        elapsed=time.clock_gettime(time.CLOCK_BOOTTIME)-origin['boottime']
        size=used(root)
        if elapsed>=c['save_stop_after_seconds'] or size>=c['output_cap_bytes']:
            save(root/'HARD_STOP.json',dict(elapsed_seconds=elapsed,bytes=size));stop_registered(root);return
        time.sleep(2)

def controller(c):
    root=Path(c['output_root']);stages=[]
    modes=('rebuild','worker','score') if c.get('migration_reconstruction') else ('check','worker','score')
    for mode in modes:
        if mode in ('rebuild','worker'):
            q=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
            selected=[x for x in q.splitlines() if x.split(',')[0].strip()==str(c['gpu']['index'])]
            assert len(selected)==1 and c['gpu']['uuid'] in selected[0]
            if int(selected[0].split(',')[-1])<8192:
                save(root/'WAITING_GPU.json',dict(status='WAITING_GPU',index=c['gpu']['index']))
                return
            save(root/'GPU_ADMISSION.private.json',dict(status='PASS',line=selected[0]))
        start=time.time();env=dict(os.environ,DIAG_MODE=mode,CUDA_VISIBLE_DEVICES=str(c['gpu']['index']) if mode=='worker' else '',OPENBLAS_NUM_THREADS='2')
        with (root/f'logs/{mode}.log').open('x') as log:
            p=subprocess.Popen([c.get('runtime_python','/tmp/q133r/env/bin/python'),c.get('runtime_entry','/tmp/e379.py')],env=env,stdout=log,stderr=subprocess.STDOUT)
            rec=dict(pid=p.pid,start_ticks=identity(p.pid),active=True,exit_code=None)
            save(root/f'processes/{mode}.json',rec)
            args=subprocess.check_output(['ps','-p',str(p.pid),'-o','args='],text=True).strip();assert args==' '.join([c.get('runtime_python','/tmp/q133r/env/bin/python'),c.get('runtime_entry','/tmp/e379.py')])
            rc=p.wait()
        rec.update(active=False,exit_code=rc,process_seconds=time.time()-start);save(root/f'processes/{mode}.json',rec);stages.append(dict(stage=mode,exit_code=rc,seconds=rec['process_seconds']))
        save(root/'STAGE_PROGRESS.private.json',dict(stages=stages))
        if mode=='check':save(root/'public/CPU_CHECK.json',dict(status='PASS' if rc==0 else 'FAIL',**stages[-1],synthetic_only=True))
        if rc:break
    save(root/'public/RESOURCE_LEDGER.json',dict(stages=stages,all_workers_retired=True,source_sha=c['source_sha'],no_retry=True,elapsed_seconds=time.clock_gettime(time.CLOCK_BOOTTIME)-read(root/'EXPERIMENT_ORIGIN.private.json')['boottime']))
    save(root/'FINAL.json',dict(status='COMPLETE' if len(stages)==3 and not rc else 'FAILED',exit_code=rc))
