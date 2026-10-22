"""Own process group, absolute deadline and bounded-output guard independent of SSH."""
import json,os,subprocess,time,signal,sys
from pathlib import Path
r=Path('/tmp/n107').resolve();c=json.loads((r/'private/config.json').read_text());c.update(json.loads((r/'recovery/AMENDMENT.json').read_text())['config_overrides']);mode=os.environ['MODE']
sys.path.insert(0,str(r/'recovery'));from codec import inspect,emergency
def ticks(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def usage():return inspect(r)['usage']
start=time.time();limit=min(3600 if mode in ('score','qualify','post_audit') else 72000,int(c['deadline']-start-60));assert limit>10
if mode in ('qualify','formal'):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 row=[s.split(',') for s in raw.splitlines() if int(s.split(',')[0])==0][0]
 assert row[1].strip()==c['gpus'][0]['uuid'] and int(row[2])>=10240,'GPU UUID/free-memory'
log=(r/'private'/f'recovery_{mode}.log').open('xb')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
p=subprocess.Popen(['timeout','-k','5',str(limit),'/tmp/q85m/bin/python','/tmp/e107r.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=p.pid,start_ticks=ticks(p.pid),mode=mode,started=start,timeout_seconds=limit)
(r/'private'/f'recovery_{mode}_timeout.json').write_text(json.dumps(rec))
reason=None
while p.poll() is None:
 if time.time()>c['deadline']-20 or usage()>=c['remote_normal_cap_bytes']:
  reason='deadline_or_storage'
  emergency(c,wrapper_trigger=dict(reason=reason,measurement=inspect(r),at=time.time()))
  if ticks(p.pid)==rec['start_ticks']:os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:
   if ticks(p.pid)==rec['start_ticks']:os.killpg(p.pid,signal.SIGKILL)
  break
 time.sleep(2)
code=p.wait();log.close()
(r/'public'/f'RECOVERY_{mode.upper()}_WRAPPER.json').write_text(json.dumps(dict(exit_code=code,started=start,ended=time.time(),wall_process_seconds=time.time()-start,stop_reason=reason,timeout_absent=ticks(p.pid) is None),indent=2))
print(json.dumps(dict(mode=mode,exit_code=code)),flush=True)
sys.exit(code)
