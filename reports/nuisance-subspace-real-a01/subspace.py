"""Current-image nuisance channel subspace; original weights/buffers stay fixed."""
import numpy as np
RANK=8

def projector(cov):
    v=np.linalg.eigh(cov)[1][:,-RANK:]
    return v@v.T

def removals(f,probes):
    shape=f.shape;x=f.reshape(shape[0],-1).astype(np.float64);xc=x-x.mean(1,keepdims=True)
    ds=[p.reshape(x.shape).astype(np.float64)-x for p in probes]
    d=np.concatenate([v-v.mean(1,keepdims=True) for v in ds],axis=1)
    pn=projector(d@d.T/d.shape[1]);pc=projector(xc@xc.T/xc.shape[1])
    q=np.linalg.qr(np.random.default_rng(20261021).normal(size=(x.shape[0],RANK)))[0];pr=q@q.T
    rn=pn@xc;energy=np.linalg.norm(rn);outs=[];ratios=[]
    assert energy>1e-8 and np.linalg.norm(d)>1e-8
    for p in (pn,pc,pr):
        r=p@xc;n=np.linalg.norm(r);assert n>1e-8
        scale=energy/n;assert scale<=100,'degenerate energy matched control'
        rr=r*scale;assert np.isclose(np.linalg.norm(rr),energy,rtol=1e-10)
        outs.append(rr.reshape(shape).astype(np.float32));ratios.append(scale)
    diag=np.array([energy,np.linalg.norm(xc),np.linalg.norm(d),np.linalg.norm(pn-pr),np.linalg.norm(pn-pc),*ratios],np.float64)
    return outs,diag

def variants(model,x):
    import torch
    captured=[]
    def capture(m,a,o):captured.append(o.detach().cpu().numpy()[0].copy())
    h=model.res.layer2.register_forward_hook(capture)
    try:
        z=model(x)[0][0].cpu().numpy()
        for v in (x.pow(.7),(x*x.new_tensor([1.15,1.,.85]).view(1,3,1,1)).clamp(0,1)):
            v=(v-v.min())/(v.max()-v.min());model(v)
    finally:h.remove()
    assert len(captured)==3 and captured[0].shape[0]==128
    rs,diag=removals(captured[0],captured[1:]);zs=[z]
    for r in rs:
        delta=torch.from_numpy(r)[None].to(x.device)
        def intervene(m,a,o):return o-delta
        h=model.res.layer2.register_forward_hook(intervene)
        try:zs.append(model(x)[0][0].cpu().numpy())
        finally:h.remove()
    zs=np.stack(zs);assert np.isfinite(zs).all()
    return zs,diag

def check():
    rng=np.random.default_rng(3);f=rng.normal(size=(16,11,13)).astype(np.float32);ps=[f+rng.normal(size=f.shape).astype(np.float32)*.1 for _ in range(2)]
    rs,d=removals(f,ps)
    assert len(rs)==3 and all(np.isfinite(r).all() for r in rs)
    assert all(np.allclose(r.mean((1,2)),0,atol=1e-7) for r in rs)
    assert np.allclose([np.linalg.norm(r) for r in rs],d[0],rtol=1e-6)
    p=projector(np.eye(16));assert np.allclose(p@p,p) and np.isclose(np.trace(p),RANK)
