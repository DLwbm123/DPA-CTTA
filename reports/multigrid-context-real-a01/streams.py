"""Coarse semantic context with fine-grid skip evidence, all original weights frozen."""
import numpy as np


def high_appearances(x):
    import torch
    import torch.nn.functional as F
    axis=torch.arange(-6,7,dtype=x.dtype);k=torch.exp(-axis.square()/8);k/=k.sum();kernel=(k[:,None]*k[None,:]).expand(3,1,13,13)
    return (x.clone(),x.pow(.7).clamp(0,1),(x*x.new_tensor((1.15,1.,.85)).view(1,3,1,1)).clamp(0,1),F.conv2d(F.pad(x,(6,)*4,mode='reflect'),kernel,groups=3).clamp(0,1))


def decode(m,deep,skips):
    import torch.nn.functional as F
    h=F.relu(deep)
    h=m.up1(h,skips[3]);h=m.up2(h,skips[2]);h=m.up3(h,skips[1]);h=m.up4(h,skips[0])
    return m.seg_head(F.relu(m.bnout(m.up5(h))))


def variants(m,x,high,verify=False):
    import torch
    import torch.nn.functional as F
    captures=[]
    def capture(mod,args,out):captures.append((out[0].clone(),[f.clone() for f in out[1]]))
    hook=m.res.register_forward_hook(capture)
    down=lambda z:F.interpolate(z,size=(512,512),mode='area')
    try:
        z0=m(x)[0];low,sl=captures.pop()
        if verify:assert torch.equal(decode(m,low,sl),z0)
        del sl
        zh=m(high)[0];dh,sh=captures.pop()
        if verify:assert torch.equal(decode(m,dh,sh),zh)
        context=F.interpolate(low,size=dh.shape[-2:],mode='bilinear',align_corners=False)
        multigrid=down(decode(m,context,sh))
        shifted=down(decode(m,torch.roll(context,context.shape[-1]//2,dims=-1),sh))
        high_logits=down(zh)
        del zh,dh,sh
        up=F.interpolate(x,size=(1024,1024),mode='bicubic',align_corners=False).clamp(0,1)
        zu=m(up)[0];du,su=captures.pop()
        if verify:assert torch.equal(decode(m,du,su),zu)
        resample=down(decode(m,context,su))
    finally:hook.remove()
    assert not captures
    ensemble=(z0+high_logits)*.5
    zs=torch.cat((z0,multigrid,high_logits,ensemble,resample,shifted),dim=0).cpu().numpy()
    assert np.isfinite(zs).all()
    diag=np.array([float((multigrid-resample).abs().mean()),float((multigrid-shifted).abs().mean())])
    return zs,diag


def check():
    import torch
    import torch.nn.functional as F
    x=torch.ones(1,2,16,16)
    y=F.interpolate(x,size=(32,32),mode='bilinear',align_corners=False)
    assert torch.equal(y,torch.ones(1,2,32,32))
    assert torch.equal(F.interpolate(y,size=(16,16),mode='area'),x)
    assert all(t.shape==(1,3,32,32) for t in high_appearances(torch.ones(1,3,32,32)))
