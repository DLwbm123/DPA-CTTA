"""R107 output routing: immutable old files plus one preallocated arena."""
import gzip,io,json,lzma
from pathlib import Path
import numpy as np
from arena import Arena,CapacityError,usage
ROOT=Path('/tmp/n107').resolve()
POOL=ROOT/'private/final_output.arena'

def arena():return Arena(POOL,ROOT)
def key(path):
    p=Path(path)
    return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)

def read_json(path):
    name=key(path);a=arena();h=a.inspect()
    if 'json:'+name in h['state']:return h['state']['json:'+name]
    if name in h['objects']:return json.loads(gzip.decompress(a.get(name)))
    return json.loads((ROOT/name).read_text())

def write_json(name,obj):
    name=key(name)
    if (ROOT/name).exists():raise FileExistsError('preserve existing '+name)
    if name.endswith(('_state.json','_process.json','_timeout.json','SCORE_PROGRESS.json','RUN_STATE.json')):
        arena().set_state('json:'+name,obj)
    else:put({},name,gzip.compress(json.dumps(obj,allow_nan=False).encode(),mtime=0))

LIMITS={'raw':262144,'known_post':1040144,'last_post':131072,'metrics':458752,'other':131072}
def group(name):
    if name.endswith('/K.bits.xz'):return 'raw'
    if name.endswith('/K.post.xor.xz'):return 'last_post' if '_final/' in name else 'known_post'
    if name.endswith('/values.private.npz.xz'):return 'metrics'
    return 'other'

def put(c,path,data):
    name=key(path)
    if (ROOT/name).exists():raise FileExistsError('preserve existing '+name)
    a=arena();g=group(name)
    used=sum(v['length'] for k,v in a.inspect()['objects'].items() if not k.startswith('code/') and group(k)==g)
    if used+len(data)>LIMITS[g]:raise CapacityError('reserved '+g+' payload limit; no bytes written')
    a.put(name,data)

def open_read(path):
    name=key(path);a=arena()
    return a.open_read(name) if name in a.inspect()['objects'] else (ROOT/name).open('rb')

def exists(path):return key(path) in arena().inspect()['objects'] or (ROOT/key(path)).exists()

def emergency(c,**event):
    a=arena();prior=a.inspect()['state'].get('failure',{});prior.update(event)
    a.set_state('failure',prior)

class PackedWriter:
    def __init__(self,path,c):self.path=path;self.c=c;self.encoder=lzma.LZMACompressor(preset=6);self.buffer=io.BytesIO();self.count=0
    def write(self,b):
        self.count+=len(b);self.buffer.write(self.encoder.compress(b))
        if self.buffer.tell()>2*1024**2:raise CapacityError('RAM encoded stream exceeds arena')
    def tell(self):return self.count
    def flush(self):pass
    def __enter__(self):return self
    def __exit__(self,t,v,tb):
        if t is None:
            self.buffer.write(self.encoder.flush());put(self.c,self.path,self.buffer.getvalue())
        self.buffer.close()

class TraceWriter:
    def __init__(self,path,c):self.path=path;self.c=c;self.buffer=io.StringIO()
    def write(self,s):
        if self.buffer.tell()+len(s)>256*1024:raise CapacityError('bounded trace RAM')
        self.buffer.write(s)
    def flush(self):pass
    def __enter__(self):return self
    def __exit__(self,t,v,tb):
        put(self.c,self.path,gzip.compress(self.buffer.getvalue().encode(),mtime=0));self.buffer.close()

def unpack(b):
    assert len(b)==65536
    return np.unpackbits(np.frombuffer(b,dtype=np.uint8)).reshape(2,512,512).astype(bool)
def xor(a,b):
    assert len(a)==len(b)==65536
    return np.bitwise_xor(np.frombuffer(a,np.uint8),np.frombuffer(b,np.uint8)).tobytes()
def self_check():
    x=bytes(65536);y=bytes([255])*65536
    assert xor(x,xor(x,y))==y
    assert np.packbits(unpack(y)).tobytes()==y
