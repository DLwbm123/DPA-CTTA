"""Factorial pairing of deep/skip streams from the same image's two appearances."""
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
        v=x.pow(.7);v=(v-v.min())/(v.max()-v.min());z1=m(v)[0]
    finally:hook.remove()
    assert len(captures)==2
    (d0,s0),(d1,s1)=captures
    if verify:assert torch.equal(decode(m,d0,s0),z0),'native decoder interface mismatch'
    a=decode(m,d0,s1);b=decode(m,d1,s0)
    shift=lambda ss:[torch.roll(f,f.shape[-1]//2,dims=-1) for f in ss]
    an=decode(m,d0,shift(s1));bn=decode(m,d1,shift(s0))
    intact=(z0+z1)*.5;cross=(z0+z1+a+b)*.25;null=(z0+z1+an+bn)*.25
    zs=torch.cat((z0,cross,intact,null),dim=0).cpu().numpy()
    assert np.isfinite(zs).all()
    diag=np.array([float((cross-intact).abs().mean()),float((cross-null).abs().mean())])
    return zs,diag

def check():
    # Factorial interaction must vanish for an additive separable decoder.
    d=np.array([1.,3.]);s=np.array([2.,7.]);z=d[:,None]+s[None,:]
    assert np.isclose(z.mean(),(z[0,0]+z[1,1])*.5)
    z=d[:,None]*s[None,:];assert not np.isclose(z.mean(),(z[0,0]+z[1,1])*.5)
