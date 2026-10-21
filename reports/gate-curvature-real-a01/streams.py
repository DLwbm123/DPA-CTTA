"""Antithetic finite-scale gate residual; no claim of a nonzero classical ReLU Hessian."""
import numpy as np


def decode(m,deep,skips):
    import torch.nn.functional as F
    h=F.relu(deep)
    h=m.up1(h,skips[3]);h=m.up2(h,skips[2]);h=m.up3(h,skips[1]);h=m.up4(h,skips[0])
    return m.seg_head(F.relu(m.bnout(m.up5(h))))


def variants(m,x,verify=False):
    import torch
    captures=[]
    def capture(mod,args,out):captures.append((out[0].clone(),[f.clone() for f in out[1]]))
    hook=m.res.register_forward_hook(capture)
    try:
        z0=m(x)[0]
        a=x.pow(.9);a=(a-a.min())/(a.max()-a.min());za=m(a)[0]
        b=x.pow(1.1);b=(b-b.min())/(b.max()-b.min());zb=m(b)[0]
    finally:hook.remove()
    (d0,s0),(da,_),(db,_)=captures
    if verify:assert torch.equal(decode(m,d0,s0),z0),'native decoder interface mismatch'
    delta=(da-db)*.5
    original_norm=torch.linalg.vector_norm(d0)
    raw_norm=torch.linalg.vector_norm(delta)
    delta=delta*torch.clamp(.1*original_norm/raw_norm.clamp_min(1e-12),max=1.)
    # Same global energy, fixed direction, independent of image/labels and probe direction.
    rng=np.random.default_rng(20261022)
    random=torch.from_numpy(rng.standard_normal(tuple(d0.shape),dtype=np.float32)).to(d0.device)
    random=random*torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(random).clamp_min(1e-12)
    smooth=(decode(m,d0+delta,s0)+decode(m,d0-delta,s0))*.5
    random_smooth=(decode(m,d0+random,s0)+decode(m,d0-random,s0))*.5
    corrected=2*z0-smooth;null=2*z0-random_smooth
    ensemble=(z0+za+zb)/3
    zs=torch.cat((z0,corrected,smooth,ensemble,null),dim=0).cpu().numpy()
    assert np.isfinite(zs).all()
    assert torch.linalg.vector_norm(delta)<=.10001*original_norm
    assert torch.allclose(torch.linalg.vector_norm(random),torch.linalg.vector_norm(delta),rtol=1e-5,atol=1e-5)
    diag=np.array([float((corrected-z0).abs().mean()),float((corrected-null).abs().mean()),float(raw_norm),float(torch.linalg.vector_norm(delta))])
    return zs,diag


def check():
    v=np.array([-2.,.1,3.]);d=np.array([.2,.3,.1])
    f=lambda x:2*x+3
    assert np.allclose(2*f(v)-(f(v+d)+f(v-d))/2,f(v))
    f=lambda x:np.maximum(x,0)
    correction=2*f(v)-(f(v+d)+f(v-d))/2
    assert np.isclose(correction[0],f(v)[0]) and np.isclose(correction[2],f(v)[2])
    assert not np.isclose(correction[1],f(v)[1])
