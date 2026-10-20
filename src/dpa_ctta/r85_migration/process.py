"""Detached owned process groups plus an independent absolute-deadline watchdog."""
import os,signal,subprocess,sys,time
from pathlib import Path
from ..r51_same_state.run import save,read


def identity(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except FileNotFoundError:return None


def alive(r):return identity(r['pid'])==r['start_ticks']


def terminate(r):
    if alive(r):os.killpg(r['pid'],signal.SIGTERM)


def child(c,mode,stage,slot,cap=None):
    root=Path(c['output_root']);name=f'{stage}_{mode}{slot}';start=time.time()
    if start>c['deadline']-180:raise TimeoutError('absolute admission deadline')
    if mode=='worker':
        raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
        inv={int(i):(u.strip(),int(m)) for i,u,m in (x.split(',') for x in raw.splitlines())};g=c['gpus'][slot]
        if inv[g['physical_id']] [0]!=g['uuid'] or inv[g['physical_id']][1]<10240:raise RuntimeError('GPU UUID/free-memory admission')
    env=dict(os.environ,MODE=mode,STAGE=stage,SLOT=str(slot),CHILD_DEADLINE=str(c['deadline']-120),CUDA_VISIBLE_DEVICES=str(c['gpus'][slot]['physical_id']) if mode=='worker' else '')
    log=(root/'logs'/f'{name}.log').open('x')
    p=subprocess.Popen([sys.executable,os.environ['ENTRY']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    r=dict(pid=p.pid,start_ticks=identity(p.pid),active=True,started=start,mode=mode,stage=stage,slot=slot)
    save(root/'processes'/f'{name}.json',r);return dict(process=p,record=r,name=name,log=log)


def finish_child(c,item):
    p=item['process'];r=item['record']
    if time.time()>c['deadline']-120 and p.poll() is None:terminate(r)
    if p.poll() is None:return None
    item['log'].close();elapsed=time.time()-r['started'];r.update(active=False,ended=time.time(),exit_code=p.returncode)
    root=Path(c['output_root']);save(root/'processes'/f'{item["name"]}.json',r)
    a=dict(name=item['name'],status='COMPLETE' if p.returncode==0 else 'FAILED',exit_code=p.returncode,seconds=elapsed,
        gpu_process_seconds=elapsed if r['mode']=='worker' else 0,cpu_process_seconds=elapsed if r['mode']!='worker' else 0)
    save(root/'attempts'/f'{item["name"]}.json',a);return a


def watchdog(c):
    """No dependency on controller or SSH; kill only identity-verified owned groups."""
    root=Path(c['output_root']);save(root/'WATCHDOG.json',dict(pid=os.getpid(),start_ticks=identity(os.getpid()),deadline=c['deadline']))
    while time.time()<c['deadline']-90:
        if (root/'CONTROLLER_EXIT.json').exists() and all(not alive(read(p)) for p in (root/'processes').glob('*.json')):
            save(root/'WATCHDOG_EXIT.json',dict(reason='all_owned_children_retired',ended=time.time()));return 0
        time.sleep(5)
    records=[read(p) for p in (root/'processes').glob('*.json')]
    if (root/'LAUNCH.json').exists():records.append(read(root/'LAUNCH.json'))
    for r in records:
        try:terminate(r)
        except ProcessLookupError:pass
    time.sleep(5)
    for r in records:
        if alive(r):
            try:os.killpg(r['pid'],signal.SIGKILL)
            except ProcessLookupError:pass
    save(root/'HARD_TIMEOUT.json',dict(status='NOT_EVALUABLE',ended=time.time(),deadline=c['deadline'],only_registered_identities=True));return 0
