"""Finite GPU/CPU stages without time/GPU quotas; enforce registered storage envelope."""
import os,sys,time,subprocess,resource
from pathlib import Path
from .run import read,save

from .storage_recovery import bytes_used

def controller(c):
    root=Path(c['output_root']);window=Path(c['storage_window_root']);reg=read(window/'STORAGE_LEDGER.json')
    assert reg['hardBytes']==20000000000 and any(v['root']==str(root) for v in reg['registered_remote_roots'])
    with (root/'START_ONCE.json').open('x') as f:f.write('{"cpu_only":false,"retry":false}')
    records=[]
    for mode in ('worker','score'):
        env=dict(os.environ,DIAG_MODE=mode,CUDA_VISIBLE_DEVICES=str(c['gpu']['index']) if mode=='worker' else '');start=time.time()
        def limit_file():resource.setrlimit(resource.RLIMIT_FSIZE,(c['output_cap_bytes'],c['output_cap_bytes']))
        with (root/'logs'/f'{mode}.log').open('x') as log:
            p=subprocess.Popen([sys.executable,os.environ['DIAG_ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=limit_file)
            process=dict(pid=p.pid,start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().split()[21]),active=True,mode=mode)
            save(root/'processes'/f'{mode}.json',process);space_stopped=False
            while p.poll() is None:
                time.sleep(2)
                if bytes_used(root)>c['output_cap_bytes']:
                    p.terminate();space_stopped=True;break
            if space_stopped:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait()
            code=p.wait()
        used=time.time()-start;process.update(active=False,exit_code=code,ended=time.time());save(root/'processes'/f'{mode}.json',process)
        rec=dict(mode=mode,status='COMPLETE' if code==0 else 'FAILED',exit_code=code,cpu_worker_seconds=used if mode=='score' else 0.,gpu_worker_seconds=used if mode=='worker' else 0.,storage_stop=space_stopped);records.append(rec);save(root/'attempts'/f'{mode}.json',rec)
        if code:break
    retired=True
    for v in reg['registered_remote_roots']:
        if v['root']==str(root):v.update(all_files_bytes=bytes_used(root),active=False,reserved_bytes=0)
    reg['actual_registered_bytes']=sum(v.get('all_files_bytes',0) for v in reg['registered_remote_roots']);assert reg['actual_registered_bytes']+reg['ancillary_reserve_bytes']<reg['workingBytes'];save(window/'STORAGE_LEDGER.json',reg)
    save(root/'RESOURCE_LEDGER.json',dict(attempts=records,all_workers_retired=retired,no_retry=True,gpu_worker_seconds=sum(v['gpu_worker_seconds'] for v in records),cpu_worker_seconds=sum(v['cpu_worker_seconds'] for v in records),time_gpu_quotas_removed_by_user=True,storage_cap_bytes=c['output_cap_bytes']))
    return 0 if len(records)==2 and all(v['exit_code']==0 for v in records) else 1
