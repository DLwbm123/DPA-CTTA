"""Neutral entry; settings are read from private files, never visible arguments."""
import json,os,sys,runpy,time,signal,resource
from pathlib import Path
r=Path('/tmp/n97').resolve();c=json.loads((r/'private/config.json').read_text())
mode=os.environ.get('MODE','audit')
e=json.loads(Path('/tmp/n85m/private/env.json').read_text())
e.update(CONFIG=str(r/'private/config.json'),ENTRY='/tmp/e97.py',PYTHONDONTWRITEBYTECODE='1',
         TMPDIR=str(r/'private'),XDG_CACHE_HOME=str(r/'private/cache'),MPLCONFIGDIR=str(r/'private/matplotlib'),
         CUDA_VISIBLE_DEVICES='0' if mode in ('qualify','formal') else '')
os.environ.update(e);sys.path[:0]=[str(r),e['PYTHONPATH']]
resource.setrlimit(resource.RLIMIT_FSIZE,(262144,262144))
seconds=max(1,int(c['deadline']-time.time()-60))
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('absolute deadline')))
signal.alarm(seconds)
def usage():
 total=0
 for p in (r,*r.rglob('*')):
  try:
   st=p.lstat()
   if not __import__('stat').S_ISLNK(st.st_mode):total+=max(st.st_size,st.st_blocks*512)
  except FileNotFoundError:pass  # Dependency import creates and removes transient paths.
 return total
def write(name,obj):
 data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
 assert len(data)<262144 and usage()+len(data)+8192<c['remote_normal_cap_bytes'],'storage reserve'
 p=r/name;tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(data);tmp.replace(p)
write('private/'+mode+'_process.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],started=time.time(),mode=mode))
start=time.time();cpu=time.process_time()
try:
 if mode=='audit':
  import audit
  result=audit.run(c,write)
 elif mode in ('qualify','formal'):
  import worker
  result=worker.run(c,write,mode)
 elif mode=='score':
  import score
  result=score.run(c,write)
 else:raise ValueError('unregistered mode')
 write('public/'+mode.upper()+'_EXIT.json',dict(status='COMPLETE',wall_seconds=time.time()-start,CPU_process_seconds=time.process_time()-cpu,result=result))
except BaseException as exc:
 write('public/'+mode.upper()+'_EXIT.json',dict(status='FAILED',wall_seconds=time.time()-start,CPU_process_seconds=time.process_time()-cpu,error=type(exc).__name__,reason=str(exc)))
 raise
