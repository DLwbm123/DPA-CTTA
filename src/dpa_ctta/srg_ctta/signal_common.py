import json
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def unpack(p):
    b=Path(p).read_bytes()
    if len(b)!=65536:raise ValueError('registered packed endpoint geometry')
    return np.unpackbits(np.frombuffer(b,np.uint8)).reshape(2,512,512).astype(bool)
def identity(pid):
    p=Path('/proc')/str(pid)/'stat'
    if not p.exists():return None
    return int(p.read_text().rsplit(')',1)[1].split()[19])
def retired(p):
    r=read(p)
    return not r['active'] and r['exit_code']==0 and identity(r['pid'])!=r['start_ticks']
