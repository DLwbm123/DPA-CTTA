"""Lossless packed-mask storage; cap actual allocation before compressed writes."""
import io,lzma,stat,json,os,time,fcntl
from pathlib import Path
import numpy as np
WIDTH=65536

def inspect(root):
    rows=[]
    for p in (Path(root),*Path(root).rglob('*')):
        try:
            s=p.lstat()
            if not stat.S_ISLNK(s.st_mode):
                rows.append(dict(path=str(p.relative_to(root)),size=s.st_size,allocated=s.st_blocks*512,charged=max(s.st_size,s.st_blocks*512)))
        except FileNotFoundError:pass
    return dict(usage=sum(x['charged'] for x in rows),largest=sorted(rows,key=lambda x:x['charged'],reverse=True)[:24])

def usage(root):return inspect(root)['usage']

class StorageLimit(RuntimeError):pass

def emergency(c,**event):
    # Preallocated fixed-size receipt uses the existing 0.5 MiB terminal reserve.
    p=Path(c['output_root'])/'private/RECOVERY_EMERGENCY.json'
    with p.open('r+b',buffering=0) as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX)
        previous=json.loads(f.read());previous.update(event,at=time.time())
        data=json.dumps(previous,allow_nan=False).encode()
        assert len(data)<=32768,'emergency receipt size'
        f.seek(0);f.write(data.ljust(32768,b' '));os.fsync(f.fileno())
        fcntl.flock(f.fileno(),fcntl.LOCK_UN)

def guard_write(c,n,label):
    observed=inspect(c['output_root']);observed.update(next_bytes=n,reserved_growth=2*n+8192,limit=c['remote_normal_cap_bytes'],operation=label)
    if observed['usage']+observed['reserved_growth']>=observed['limit']:
        emergency(c,storage_trigger=observed)
        raise StorageLimit(json.dumps(observed,separators=(',',':')))


def put(c,p,data):
    guard_write(c,len(data),str(Path(p).relative_to(c['output_root'])))
    with Path(p).open('xb') as f:f.write(data)

class PackedWriter:
    def __init__(self,p,c):
        self.c=c;self.f=Path(p).open('xb');self.encoder=lzma.LZMACompressor(preset=6);self.count=0
    def _write(self,b):
        if not b:return
        guard_write(self.c,len(b),str(Path(self.f.name).relative_to(self.c['output_root'])))
        self.f.write(b);self.f.flush()
    def write(self,b):self.count+=len(b);self._write(self.encoder.compress(b))
    def tell(self):return self.count
    def flush(self):self.f.flush()
    def __enter__(self):return self
    def __exit__(self,t,v,tb):
        try:
            if t is None:self._write(self.encoder.flush())
        finally:self.f.close()

def unpack(b):
    assert len(b)==WIDTH
    return np.unpackbits(np.frombuffer(b,dtype=np.uint8)).reshape(2,512,512).astype(bool)

def xor(a,b):
    assert len(a)==len(b)==WIDTH
    return np.bitwise_xor(np.frombuffer(a,np.uint8),np.frombuffer(b,np.uint8)).tobytes()

def self_check():
    a=np.zeros((2,512,512),bool);a[:,10:80,30:90]=True;b=a.copy();b[1,20:30,40:50]=False
    x=np.packbits(a).tobytes();y=np.packbits(b).tobytes()
    assert np.array_equal(unpack(lzma.decompress(lzma.compress(x,preset=6))),a)
    assert xor(x,lzma.decompress(lzma.compress(xor(x,y),preset=6)))==y
