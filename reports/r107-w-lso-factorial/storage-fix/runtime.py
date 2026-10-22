"""Prepared final-run supervisor. Refuses to run without a new explicit authorization."""
import gzip,json,os,signal,subprocess,sys,threading,time,types,traceback
from pathlib import Path
ROOT=Path('/tmp/n107').resolve()
sys.path.insert(0,str(ROOT/'storage-fix'))
from arena import Arena,usage,NORMAL
POOL=ROOT/'private/final_output.arena'
DEADLINE=1792751614.42385

def arena():return Arena(POOL,ROOT)
def authorized():
    a=arena().inspect()['state'].get('authorization',{})
    required=dict(user_approved=True,max_updates=2477,max_backwards=2477,max_forwards=22317,deadline=DEADLINE,remote_cap_bytes=10485760)
    if any(a.get(k)!=v for k,v in required.items()):raise PermissionError('new human approval of 2477 updates/backwards and 22317 forwards required')
    if time.time()>=DEADLINE-120:raise TimeoutError('original deadline')

def setup():
    e=json.loads(Path('/tmp/n85m/private/env.json').read_text());e.update(PYTHONDONTWRITEBYTECODE='1',TMPDIR=str(ROOT/'private'),XDG_CACHE_HOME=str(ROOT/'private/cache'),MPLCONFIGDIR=str(ROOT/'private/matplotlib'),CUDA_CACHE_DISABLE='1')
    os.environ.update(e);sys.path[:0]=[str(ROOT/'storage-fix'),str(ROOT),e['PYTHONPATH']]
    bundle=json.loads(gzip.decompress(arena().get('code/final_bundle.json.gz')))
    for name in ('io_adapter','worker','score'):
        module=types.ModuleType(name);module.__file__='<'+name+'>';sys.modules[name]=module
        exec(compile(bundle[name],'<'+name+'>','exec'),module.__dict__)
    return sys.modules['io_adapter']

def stage():
    authorized();mode=os.environ['FINAL_STAGE'];os.environ['CUDA_VISIBLE_DEVICES']='0' if mode=='formal' else ''
    io=setup();c=json.loads((ROOT/'private/config.json').read_text());c.update(max_updates=2477,max_backwards=2477,max_forwards=22317)
    signal.signal(signal.SIGTERM,lambda *_:(_ for _ in ()).throw(TimeoutError('supervisor termination')))
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('original deadline')));signal.alarm(max(1,int(DEADLINE-time.time()-60)))
    start=time.time();io.write_json('private/final_'+mode+'_process.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],started=start))
    try:
        result=sys.modules['worker'].run(c,io.write_json,'formal') if mode=='formal' else sys.modules['score'].run(c,io.write_json)
        io.write_json('public/FINAL_'+mode.upper()+'_EXIT.json',dict(status='COMPLETE',seconds=time.time()-start,result=result))
    except BaseException as e:
        io.emergency(c,stage=mode,error=type(e).__name__,reason=str(e)[:512],traceback=traceback.format_exc()[-1024:]);raise

def bootstrap():
    return "import sys,gzip,json;from pathlib import Path;sys.path.insert(0,'/tmp/n107/storage-fix');from arena import Arena;a=Arena('/tmp/n107/private/final_output.arena','/tmp/n107');b=json.loads(gzip.decompress(a.get('code/final_bundle.json.gz')));exec(compile(b['runtime'],'<runtime>','exec'));stage()"

def controller():
    authorized();a=arena()
    if a.inspect()['state'].get('final_started'):raise RuntimeError('one final launch only')
    assert a.inspect()['state'].get('validation',{}).get('status')=='PASS'
    assert usage(ROOT)+32768<NORMAL
    assert len(list((ROOT/'jobs').glob('*/COMPLETE.json')))==11
    assert DEADLINE-time.time()>4500
    # GPU readiness is checked before consuming the one-launch marker.
    c=json.loads((ROOT/'private/config.json').read_text())
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free','--format=csv,noheader,nounits'],text=True)
    row=[v.split(',') for v in rows.splitlines() if v.split(',')[0].strip()=='0'][0]
    assert row[1].strip()==c['gpus'][0]['uuid'] and int(row[2])>=10240
    a.set_state('final_started',dict(at=time.time(),deadline=DEADLINE))
    io=setup()
    for mode in ('formal','score'):
        authorized()
        limit=min(3600,int(DEADLINE-time.time()-60));assert limit>120
        env=dict(os.environ,FINAL_STAGE=mode,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
        p=subprocess.Popen(['timeout','-k','5',str(limit),'/tmp/q85m/bin/python','-'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True,env=env)
        p.stdin.write(bootstrap().encode());p.stdin.close();io.write_json('private/final_'+mode+'_timeout.json',dict(pid=p.pid,started=time.time(),timeout_seconds=limit))
        def drain():
            tail=b''
            while True:
                data=p.stdout.read(1024)
                if not data:break
                tail=(tail+data)[-1024:];a.set_state('log:'+mode,tail.decode(errors='replace'))
        thread=threading.Thread(target=drain);thread.start()
        while p.poll() is None:
            if time.time()>DEADLINE-20 or usage(ROOT)>=NORMAL:
                a.set_state('supervisor_stop',dict(at=time.time(),usage=usage(ROOT)));os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
                break
            time.sleep(1)
        code=p.wait();thread.join();io.write_json('public/FINAL_'+mode.upper()+'_WRAPPER.json',dict(exit_code=code,ended=time.time(),timeout_absent=not Path('/proc/'+str(p.pid)).exists()))
        if code:
            io.write_json('public/FINAL_CONTROLLER_EXIT.json',dict(status='FAILED',stage=mode,exit_code=code));return
    io.write_json('public/FINAL_CONTROLLER_EXIT.json',dict(status='COMPLETE',at=time.time()))
