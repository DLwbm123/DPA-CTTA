"""Frozen single-block reuse, with residual replay and position controls."""
import numpy as np


def altered(m,x,block,mode):
    import torch
    assert block.downsample is None and block.stride==1
    visits=[]
    def apply(mod,args,out):
        assert out.shape==args[0].shape
        visits.append(1)
        if mode=='reuse':
            # Calling forward directly bypasses the hook: exactly one extra block.
            return mod.forward(out)
        if mode=='replay':return torch.relu(2*out-args[0])
        if mode=='identity':return out
        raise ValueError(mode)
    hook=block.register_forward_hook(apply)
    try:z=m(x)[0]
    finally:hook.remove()
    assert len(visits)==1
    return z


def variants(m,x,verify=False):
    import torch
    z0=m(x)[0]
    early=m.res.layer1[0];late=m.res.layer4[-1]
    if verify:assert torch.equal(altered(m,x,early,'identity'),z0)
    single=altered(m,x,early,'reuse')
    replay=altered(m,x,early,'replay')
    deeper=altered(m,x,late,'reuse')
    probe=x.pow(.9);probe=(probe-probe.min())/(probe.max()-probe.min())
    aug=m(probe)[0]
    main=(z0+single)*.5
    zs=torch.cat((z0,main,single,(z0+replay)*.5,(z0+deeper)*.5,(z0+aug)*.5)).cpu().numpy()
    assert np.isfinite(zs).all()
    diag=np.array([float((single-z0).abs().mean()),float((single-replay).abs().mean()),float((main-z0).abs().mean())])
    return zs,diag


def check():
    # Composition differs from replaying the old residual, even without learning.
    f=lambda x:np.maximum(.5*x+1,0)
    x=np.array([-3.,0.,4.]);y=f(x)
    assert not np.array_equal(f(y),np.maximum(2*y-x,0))
