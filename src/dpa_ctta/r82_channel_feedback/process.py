"""Neutral detached processes; record all costs, no user GPU/time cap."""
import os,subprocess,sys,time
from pathlib import Path
from ..r51_same_state.run import save


def child(c,mode,stage,slot,cap):
    root=Path(c['output_root']);start=time.time();name=f'{stage}_{mode}{slot}'
    if mode=='worker':
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        inv={int(i):(u.strip(),int(m)) for i,u,m in (x.split(',') for x in raw.splitlines())};g=c['gpus'][slot]
        if inv[g['physical_id']][0]!=g['uuid'] or inv[g['physical_id']][1]<10240:raise ValueError('GPU memory/UUID admission')
    env=dict(os.environ,MODE=mode,STAGE=stage,SLOT=str(slot),CHILD_DEADLINE='inf',CUDA_VISIBLE_DEVICES=str(c['gpus'][slot]['physical_id']) if mode=='worker' else '')
    log=(root/'logs'/f'{name}.log').open('x')
    p=subprocess.Popen([sys.executable,os.environ['ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r=dict(pid=p.pid,start_ticks=Path(f'/proc/{p.pid}/stat').read_text().split()[21],active=True,started=start,mode=mode,stage=stage,slot=slot)
    save(root/'processes'/f'{name}.json',r);return dict(process=p,record=r,name=name,log=log)


def finish_child(c,item):
    p=item['process']
    if p.poll() is None:return None
    root=Path(c['output_root']);r=item['record'];elapsed=time.time()-r['started'];item['log'].close()
    r.update(active=False,ended=time.time(),exit_code=p.returncode);save(root/'processes'/f'{item["name"]}.json',r)
    a=dict(name=item['name'],status='COMPLETE' if p.returncode==0 else 'FAILED',exit_code=p.returncode,seconds=elapsed,
           gpu_process_seconds=elapsed if r['mode']=='worker' else 0,cpu_process_seconds=elapsed if r['mode']!='worker' else 0)
    save(root/'attempts'/f'{item["name"]}.json',a);return a
