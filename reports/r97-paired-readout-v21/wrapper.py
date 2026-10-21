"""Own process group, absolute deadline and bounded-output guard independent of SSH."""
import json,os,subprocess,time,signal,sys
from pathlib import Path
r=Path('/tmp/n97').resolve();c=json.loads((r/'private/config.json').read_text());mode=os.environ['MODE']
def ticks(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def usage():return sum(max(p.stat().st_size,p.stat().st_blocks*512) for p in (r,*r.rglob('*')) if not p.is_symlink())
start=time.time();limit=min(1200 if mode in ('audit','score','qualify') else 10000,int(c['deadline']-start-30));assert limit>10
if mode in ('qualify','formal'):
 raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
 row=[s.split(',') for s in raw.splitlines() if int(s.split(',')[0])==0][0]
 assert row[1].strip()==c['gpus'][0]['uuid'] and int(row[2])>=10240,'GPU UUID/free-memory'
log=(r/'private'/f'{mode}.log').open('xb')
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
p=subprocess.Popen(['timeout','-k','5',str(limit),'/tmp/q85m/bin/python','/tmp/e97.py'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
rec=dict(pid=p.pid,start_ticks=ticks(p.pid),mode=mode,started=start,timeout_seconds=limit)
(r/'private'/f'{mode}_timeout.json').write_text(json.dumps(rec))
reason=None
while p.poll() is None:
 if time.time()>c['deadline']-20 or usage()>=c['remote_normal_cap_bytes']:
  reason='deadline_or_storage'
  if ticks(p.pid)==rec['start_ticks']:os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=5)
  except subprocess.TimeoutExpired:
   if ticks(p.pid)==rec['start_ticks']:os.killpg(p.pid,signal.SIGKILL)
  break
 time.sleep(2)
code=p.wait();log.close()
(r/'public'/f'{mode.upper()}_WRAPPER.json').write_text(json.dumps(dict(exit_code=code,started=start,ended=time.time(),wall_process_seconds=time.time()-start,stop_reason=reason,timeout_absent=ticks(p.pid) is None),indent=2))
print(json.dumps(dict(mode=mode,exit_code=code)),flush=True)
sys.exit(code)
