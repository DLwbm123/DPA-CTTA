import os, sys, json, gzip
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())

def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def unpack(p):
    p=Path(p);raw=p.read_bytes() if p.exists() else gzip.decompress(Path(str(p)+'.gz').read_bytes())
    return np.unpackbits(np.frombuffer(raw,np.uint8)).reshape(2,512,512).astype(bool)

def appearances(x):
    import torch
    import torch.nn.functional as F
    axis=torch.arange(-3,4,dtype=x.dtype);k=torch.exp(-axis.square()/2);k/=k.sum();kernel=(k[:,None]*k[None,:]).expand(3,1,7,7)
    return (x.clone(),x.pow(.7).clamp(0,1),(x*x.new_tensor((1.15,1.,.85)).view(1,3,1,1)).clamp(0,1),F.conv2d(F.pad(x,(3,)*4,mode='reflect'),kernel,groups=3).clamp(0,1))

def aggregate(a):
    a=np.asarray(a,float);contents=[]
    for row in a:
        ok=np.isfinite(row);contents.append(float(np.average(row[ok],weights=np.tile([2,1,1,1],len(row)//4)[ok])) if ok.any() else None)
    valid=[v for v in contents if v is not None];ci=None
    if len(valid)==8:
        ix=np.random.default_rng(20261018).integers(0,8,(2000,8));ci=np.quantile(np.asarray(contents)[ix].mean(1),[.025,.975]).tolist()
    return dict(mean=float(np.mean(valid)) if valid else None,ci95=ci,defined_inputs=int(np.isfinite(a).sum()),defined_contents=len(valid))

def audit(c,panel):
    root=Path(c['root']).resolve();old=Path(c['sealed_root']).resolve();data=Path(c['source_root']).resolve();models=Path(c['model_root']).resolve();checkpoint=Path(c['checkpoint_path']).resolve()
    allowed={Path(p['image_path']).resolve() for p in panel};bits={(old/'inputs'/f'{i}_{k}_{a}.bits').resolve() for i in range(8) for k in range(4) for a in ('SOURCE','TOPO_SMOOTH')}
    bits |= {Path(str(p)+'.gz') for p in bits}
    prior={Path(c['prior_root'])/'private/measurement'/f'{o}_{i}_{k}.npz' for o in range(2) for i in range(8) for k in range(4)}
    prior |= {Path(c['bn_prior_root'])/'private/measurement'/f'{o}_{i}_{k}.npz' for o in range(2) for i in range(8) for k in range(4)}
    prior |= {Path(c['repair_prior_root'])/'private/measurement'/f'{o}_{i}_{k}.npz' for o in range(2) for i in range(8) for k in range(4)}
    prefixes=[Path(sys.prefix).resolve(),Path(sys.base_prefix).resolve(),Path(os.environ['DPA_CTTA_BASE_ROOT']).resolve()]
    def hook(event,args):
        if event.startswith('socket.'):raise PermissionError('network denied')
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        p=Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(data) and p not in allowed:raise PermissionError('unregistered data')
        if p.is_relative_to(models) and p!=checkpoint:raise PermissionError('unregistered model')
        if p.is_relative_to(old) and p not in bits and not p.is_relative_to(old/'code'):raise PermissionError('old metadata or evaluation')
        if p.is_relative_to(root/'scorer'):raise PermissionError('GT stage')
        if str(p).startswith('/remote-home') and not (p.is_relative_to(root) or p in prior or p in allowed or p in bits or p==checkpoint or p.is_relative_to(old/'code') or any(p.is_relative_to(v) for v in prefixes)):raise PermissionError('outside assets')
        if args[2]&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC) and not p.is_relative_to(root):raise PermissionError('outside write')
    sys.addaudithook(hook)
    for p in [old/'scorer/panel.json',models/'forbidden.pth',data/'unregistered.png']:
        try:p.read_bytes()
        except PermissionError:pass
        else:raise AssertionError('access gate')
    import socket
    try:socket.socket()
    except PermissionError:pass
    else:raise AssertionError('socket gate')
    return 4
