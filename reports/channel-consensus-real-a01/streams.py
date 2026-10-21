"""Nonlinear per-channel consensus with an unchanged affine source readout."""
import numpy as np


def middle(a,b,c):
    import torch
    return torch.maximum(torch.minimum(a,b),torch.minimum(torch.maximum(a,b),c))

def variants(m,x,verify=False):
    import torch
    feats=[]
    hook=m.seg_head.register_forward_pre_hook(lambda mod,args:feats.append(args[0].clone()))
    try:
        z0=m(x)[0]
        a=x.pow(.9);a=(a-a.min())/(a.max()-a.min());za=m(a)[0]
        b=x.pow(1.1);b=(b-b.min())/(b.max()-b.min());zb=m(b)[0]
    finally:hook.remove()
    assert len(feats)==3
    f0,fa,fb=feats
    main=m.seg_head(middle(f0,fa,fb))
    null=m.seg_head(middle(f0,torch.roll(fa,fa.shape[-1]//2,-1),torch.roll(fb,fb.shape[-1]//2,-1)))
    mean=(z0+za+zb)/3
    if verify:
        check_mean=m.seg_head(torch.stack(feats).mean(0))
        assert torch.allclose(check_mean,mean,rtol=1e-5,atol=1e-4),'affine mean parity'
    med=middle(z0,za,zb)
    zs=torch.cat((z0,main,med,mean,null)).cpu().numpy()
    assert np.isfinite(zs).all()
    diag=np.array([float((main-z0).abs().mean()),float((main-med).abs().mean()),float((main-null).abs().mean()),float((middle(f0,fa,fb)-f0).abs().mean())])
    return zs,diag


def check():
    f=np.array([[0.,2.],[1.,0.],[2.,1.]])
    w=np.array([1.,2.])
    assert np.allclose(f.mean(0)@w,(f@w).mean())
    assert not np.isclose(np.median(f,axis=0)@w,np.median(f@w))
