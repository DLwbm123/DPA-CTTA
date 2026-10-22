"""Detached supervisor: shared absolute deadline and strict output budget."""
import os,json,time,signal,subprocess,datetime
from pathlib import Path
ROOT=Path(os.environ['WORK_ROOT']);cfg=json.loads((ROOT/'config.private.json').read_text())
def write(p,x):
 p=ROOT/p;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def size():return sum(p.stat().st_blocks*512 for p in ROOT.rglob('*') if p.is_file())
old=json.loads((ROOT/'RUNTIME_RECEIPT.json').read_text()) if (ROOT/'RUNTIME_RECEIPT.json').exists() else None
start=old['origin_unix'] if old else time.time();deadline=min(start+cfg['hard_seconds'],datetime.datetime.fromisoformat(cfg['window_end_utc']).timestamp())
write('RUNTIME_RECEIPT.json',{'origin_unix':start,'deadline_unix':deadline,'pid':os.getpid(),'baseline_bytes':size(),'hard_seconds':cfg['hard_seconds'],'window_limit':30000000000,'run_limit':cfg['run_storage_limit_bytes']})
with (ROOT/'train.log').open('a') as out:
 p=subprocess.Popen([cfg['python'],cfg['neutral_train']],env=os.environ.copy(),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 write('PROCESS.json',{'worker_pid':p.pid,'controller_pid':os.getpid(),'worker_argv':[cfg['python'],cfg['neutral_train']]})
 reason=None
 while p.poll() is None:
  if time.time()>=deadline-90:reason='DEADLINE'
  if size()>=cfg['run_storage_limit_bytes']-50000000:reason='STORAGE'
  if reason:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(10)
 write('EXIT.json',{'code':p.returncode,'reason':reason,'seconds':time.time()-start,'allocated_bytes':size(),'ended_unix':time.time()})
