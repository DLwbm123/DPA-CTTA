import base64,json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
payload=base64.b64encode((r/'final_bundle.json.gz').read_bytes()).decode()
script='''import os,sys,json,gzip,lzma,types,base64
from pathlib import Path
sys.path.insert(0,'/tmp/n107/storage-fix');from arena import Arena,usage
r=Path('/tmp/n107').resolve();a=Arena(r/'private/final_output.arena',r)
h=a.inspect();assert not h['state'].get('authorization',{}).get('user_approved') and not h['state'].get('final_started') and all(k.startswith(('code/','validation/')) or k=='public/validation_result.json' for k in h['objects'])
with a.locked() as f:
 old=a._load(f);a._commit(f,dict(old,cursor=32768,objects={},state={'phase':'VALIDATED_EMPTY'}))
a.put('code/final_bundle.json.gz',base64.b64decode(PAYLOAD))
b=json.loads(gzip.decompress(a.get('code/final_bundle.json.gz')));before=a.inspect()
mod=types.ModuleType('io_adapter');exec(compile(b['io_adapter'],'<io_adapter>','exec'),mod.__dict__);sys.modules['io_adapter']=mod
with mod.PackedWriter(r/'validation/raw.xz',{}) as w:
 w.write(bytes(65536));w.write(bytes([255])*65536);assert w.tell()==131072
assert lzma.decompress(a.get('validation/raw.xz'))==bytes(65536)+bytes([255])*65536
with mod.TraceWriter(r/'validation/trace.gz',{}) as t:t.write('test trace\\n')
assert gzip.decompress(a.get('validation/trace.gz'))==b'test trace\\n'
mod.write_json('private/validation_state.json',dict(updates=192));assert mod.read_json(r/'private/validation_state.json')['updates']==192
mod.write_json('public/validation_result.json',dict(result='fixture'));assert mod.read_json(r/'public/validation_result.json')['result']=='fixture'
cursor=a.inspect()['cursor']
try:mod.put({},r/'validation/too_large',bytes(131073))
except mod.CapacityError:pass
else:raise AssertionError('category quota not enforced')
assert a.inspect()['cursor']==cursor
for i in range(48):a.put('validation/catalog'+str(i),b'x')
a.set_state('log:formal','x'*1024);a.set_state('log:score','x'*1024)
mod.emergency({},worker_state=dict(updates=192,forwards=1728,completed_jobs=['x'*30]*12,reason='x'*512),traceback='x'*1024,reason='x'*512)
sm=types.ModuleType('score');exec(compile(b['score'],'<score>','exec'),sm.__dict__);sm.self_check()
rm=types.ModuleType('runtime');exec(compile(b['runtime'],'<runtime>','exec'),rm.__dict__)
try:rm.authorized()
except PermissionError:blocked=True
else:raise AssertionError('unapproved model replay was not blocked')
peak=a.inspect();assert peak['root_usage']<9961472 and peak['allocated_bytes']==4194304
with a.locked() as f:
 h=a._load(f);a._commit(f,dict(h,cursor=before['cursor'],objects=before['objects'],state=before['state']))
a.set_state('validation',dict(status='PASS',extent=4194304,models=0,labels=0,authorization='NOT_GRANTED'))
print(json.dumps(dict(status='PASS',output_routing_roundtrip=True,metadata_pressure_catalog_objects=48,failed_state_retained=True,score_selfcheck=True,authorization_blocks_launch=blocked,model_forwards=0,backwards=0,updates=0,labels_read=0,before=before,peak=peak['root_usage'],after=a.inspect())))
'''.replace('PAYLOAD',repr(payload))
out=subprocess.check_output(['ssh','my-gpu','PYTHONDONTWRITEBYTECODE=1 /tmp/q85m/bin/python -'],input=script.encode());d=json.loads(out)
(r/'INTEGRATION_VALIDATION.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k not in ('before','after')}));print('payload_free',d['after']['payload_free'])
