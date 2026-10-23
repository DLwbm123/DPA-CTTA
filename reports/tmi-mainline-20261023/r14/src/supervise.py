"""Single r14 supervisor; retain original clock, stop before strict limits."""
import os,json,time,signal,subprocess
from pathlib import Path
R=Path(os.environ['WORK_ROOT']);c=json.loads((R/'config.private.json').read_text())
def write(n,x):
 p=R/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def size():return sum(p.stat().st_blocks*512 for p in R.rglob('*') if p.is_file())
f=R/'RUNTIME_RECEIPT.json';existing=json.loads(f.read_text()) if f.exists() else None
origin=existing['origin_unix'] if existing else time.time();deadline=min(origin+10800,c['batch_deadline_unix'])
write('RUNTIME_RECEIPT.json',{'origin_unix':origin,'deadline_unix':deadline,'hard_seconds':10800,'batch_origin_unix':c['batch_origin_unix'],'batch_deadline_unix':c['batch_deadline_unix'],'baseline_bytes':size(),'expected_peak_bytes':100000000,'soft_stop_bytes':3000000000,'hard_storage_bytes':4000000000,'save_margin_seconds':300,'r15_authorized':False})
if time.time()>=deadline-300:raise RuntimeError('Original budget expired')
if (R/'TRAINING_FINISHED.json').exists() or (R/'PROCESS.json').exists():raise RuntimeError('Refuse duplicate or completed launch')
with (R/'train.log').open('a') as out:
 p=subprocess.Popen([c['python'],c['neutral_train']],env=os.environ.copy(),stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
 write('PROCESS.json',{'worker_pid':p.pid,'controller_pid':os.getpid(),'worker_argv':[c['python'],c['neutral_train']]});reason=None
 while p.poll() is None:
  if time.time()>=deadline-300:reason='TIME_STOP'
  if size()>=3000000000:reason='STORAGE_SOFT_STOP'
  if reason:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=60)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   break
  time.sleep(5)
 write('EXIT.json',{'code':p.returncode,'reason':reason,'seconds':time.time()-origin,'allocated_bytes':size(),'ended_unix':time.time()})
