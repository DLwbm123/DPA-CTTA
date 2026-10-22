"""Actual-NFS pressure check. No model forward, backward, labels or old-file writes."""
import json,os,sys,threading,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from arena import Arena,CapacityError,SIZE,START,BANK,PREFIX,MAGIC,usage,NORMAL
r=Path('/tmp/n107').resolve();path=r/'private/final_output.arena'
assert not path.exists(),'one validation allocation; retain previous receipt'
peak={'root_usage':usage(r),'arena_allocated':0,'samples':0};stop=threading.Event()
def observe():
    while not stop.is_set():
        u=usage(r);peak['root_usage']=max(peak['root_usage'],u);peak['samples']+=1
        if path.exists():peak['arena_allocated']=max(peak['arena_allocated'],path.stat().st_blocks*512)
        stop.wait(.02)
t=threading.Thread(target=observe);t.start();start=time.time()
try:
    a,creation=Arena.create(path,r)
    fixed_size=path.stat().st_size
    # Fill almost the complete payload capacity with incompressible bytes.
    n=(SIZE-START-4096)//4
    for i in range(4):a.put('test/payload'+str(i),os.urandom(n))
    before=a.inspect()
    try:a.put('test/too_large',bytes(8192))
    except CapacityError:pass
    else:raise AssertionError('oversized append admitted')
    after=a.inspect();assert before['cursor']==after['cursor'] and path.stat().st_size==fixed_size
    # All remaining real updates plus scorer/control writes, with no append growth.
    for i in range(384):
        a.set_state('progress',dict(update=i,forwards=i*9,trace='x'*1500))
        if usage(r)>=NORMAL:raise AssertionError('NFS allocation exceeded normal cap')
    committed=a.inspect();assert committed['state']['progress']['update']==383
    # Interrupt an inactive bank before its footer: the prior committed bank wins.
    with a.locked() as f:
        h=a._load(f);next_generation=h['generation']+1
        os.pwrite(f.fileno(),PREFIX.pack(MAGIC,next_generation,100),next_generation%2*BANK);os.fsync(f.fileno())
    assert a.inspect()['generation']==committed['generation']
    a.set_state('failure',dict(status='FAILED_TEST',updates=192,backwards=192,error='x'*2048))
    assert a.inspect()['state']['failure']['updates']==192
    assert path.stat().st_size==SIZE
    # Remove only this test's logical objects. The extent remains fully allocated.
    with a.locked() as f:
        h=a._load(f);a._commit(f,dict(h,cursor=START,objects={},state={'phase':'VALIDATED_EMPTY','model_calls':0}))
    result=dict(status='PASS',creation=creation,post_stress=a.inspect(),pressure_payload_bytes=n*4,metadata_commits=386,capacity_rejection_before_write=True,interrupted_bank_recovered=True,failure_state_saved=True,physical_extent_retained=True,old_assets_modified=False,model_forwards=0,training_updates=0,backwards=0,labels_read=0,wall_seconds=time.time()-start)
except BaseException as e:
    result=dict(status='FAILED',error=type(e).__name__,reason=str(e),wall_seconds=time.time()-start)
finally:
    stop.set();t.join();result['observed_peak']=peak
print(json.dumps(result),flush=True)
if result['status']!='PASS':raise SystemExit(1)
