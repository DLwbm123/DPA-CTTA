"""Lossless packed-mask storage; cap actual allocation before compressed writes."""
import io,lzma,stat
from pathlib import Path
import numpy as np
WIDTH=65536

def usage(root):
    total=0
    for p in (Path(root),*Path(root).rglob('*')):
        try:
            st=p.lstat()
            if not stat.S_ISLNK(st.st_mode):total+=max(st.st_size,st.st_blocks*512)
        except FileNotFoundError:pass
    return total

def put(c,p,data):
    assert usage(c['output_root'])+2*len(data)+8192<c['remote_normal_cap_bytes'],'allocated storage reserve'
    with Path(p).open('xb') as f:f.write(data)

class PackedWriter:
    def __init__(self,p,c):
        self.c=c;self.f=Path(p).open('xb');self.encoder=lzma.LZMACompressor(preset=6);self.count=0
    def _write(self,b):
        if not b:return
        assert usage(self.c['output_root'])+2*len(b)+8192<self.c['remote_normal_cap_bytes'],'packed storage reserve'
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
