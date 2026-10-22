"""Neutral entry; settings are read from private files, never visible arguments."""
import json,os,sys,runpy,time,signal,resource
from pathlib import Path
r=Path('/tmp/n107').resolve();c=json.loads((r/'private/config.json').read_text())
c.update(json.loads((r/'recovery/AMENDMENT.json').read_text())['config_overrides'])
mode=os.environ.get('MODE','audit')
e=json.loads(Path('/tmp/n85m/private/env.json').read_text())
e.update(CONFIG=str(r/'private/config.json'),ENTRY='/tmp/e107r.py',PYTHONDONTWRITEBYTECODE='1',
         TMPDIR=str(r/'private'),XDG_CACHE_HOME=str(r/'private/cache'),MPLCONFIGDIR=str(r/'private/matplotlib'),
         CUDA_VISIBLE_DEVICES='0' if mode in ('qualify','formal') else '')
os.environ.update(e);sys.path[:0]=[str(r/'recovery'),str(r),e['PYTHONPATH']]
from codec import usage as disk_usage,guard_write,emergency
resource.setrlimit(resource.RLIMIT_FSIZE,(c['remote_cap_bytes'],c['remote_cap_bytes']))
seconds=int(c['deadline']-time.time()-60);assert seconds>0
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('absolute deadline')))
signal.alarm(seconds)
signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('wrapper termination')))
def write(name,obj):
 if name.startswith('private/formal') or name.startswith('private/score') or name.startswith('private/selfcheck'):name=name.replace('private/','private/recovery_',1)
 if name.startswith('public/FORMAL') or name.startswith('public/SCORE_') or name=='public/SELFCHECK_EXIT.json':name=name.replace('public/','public/RECOVERY_',1)
 data=(json.dumps(obj,indent=2,allow_nan=False)+'\n').encode()
 assert len(data)<4*1024**2
 guard_write(c,len(data),name)
 p=r/name;tmp=p.with_name(p.name+'.tmp');tmp.write_bytes(data);tmp.replace(p)
write('private/'+mode+'_process.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],started=time.time(),mode=mode))
start=time.time();cpu=time.process_time()
try:
 if mode=='selfcheck':
  import method,score
  method.self_check();score.self_check();__import__('post_audit').self_check()
  result={'status':'PASS'}
 elif mode in ('qualify','formal'):
  import worker
  result=worker.run(c,write,mode)
 elif mode=='post_audit':
  import post_audit
  result=post_audit.run(c,write)
 elif mode=='score':
  import score
  result=score.run(c,write)
 else:raise ValueError('unregistered mode')
 write('public/'+mode.upper()+'_EXIT.json',dict(status='COMPLETE',wall_seconds=time.time()-start,CPU_process_seconds=time.process_time()-cpu,result=result))
except BaseException as exc:
 emergency(c,entry_failure=dict(mode=mode,status='FAILED',wall_seconds=time.time()-start,CPU_process_seconds=time.process_time()-cpu,error=type(exc).__name__,reason=str(exc)[:2048]))
 try:write('public/'+mode.upper()+'_EXIT.json',dict(status='FAILED',wall_seconds=time.time()-start,error=type(exc).__name__,reason=str(exc)[:2048]))
 except Exception as receipt_error:emergency(c,receipt_error=str(receipt_error)[:2048])
 raise
