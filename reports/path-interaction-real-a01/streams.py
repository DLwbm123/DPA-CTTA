"""Fixed inclusion-exclusion of native nonlinear path evidence."""
import numpy as np

class Memory:
    def __init__(self): self.ids=[]
    def add(self,i,proposals):
        assert i not in self.ids and len(proposals)==4
        self.ids.append(i)

def variants(m,x,memory):
    import torch
    import torch.nn.functional as F
    captures={}
    def save(name):
        def hook(mod,args,out): captures[name]=out.clone()
        return hook
    handles=[getattr(m,n).register_forward_hook(save(n)) for n in ('up4','up5','bnout')]
    try:z=m(x)[0]
    finally:
        for h in handles:h.remove()
    assert set(captures)=={'up4','up5','bnout'}
    h=captures['up4'];conv=m.up5;bn=m.bnout
    assert conv.groups==1 and h.shape[1]==256 and m.up4.tr_conv.out_channels==m.up4.x_conv.out_channels==128
    def component(v,w):
        return F.conv_transpose2d(v,w,None,conv.stride,conv.padding,conv.output_padding,conv.groups,conv.dilation)
    deep=component(h[:,:128],conv.weight[:128]);skip=component(h[:,128:],conv.weight[128:])
    bias=conv.bias.view(1,-1,1,1) if conv.bias is not None else 0
    assert torch.allclose(deep+skip+bias,captures['up5'],rtol=2e-5,atol=2e-4),'native path decomposition'
    scale=bn.weight.view(1,-1,1,1)/torch.sqrt(bn.running_var.view(1,-1,1,1)+bn.eps)
    d=deep*scale;s=skip*scale;u=captures['bnout']
    baseline=(bias-bn.running_mean.view(1,-1,1,1))*scale+bn.bias.view(1,-1,1,1)
    assert torch.allclose(baseline+d+s,u,rtol=2e-5,atol=2e-4),'BN path decomposition'
    zd=m.seg_head(F.relu(baseline+d));zs=m.seg_head(F.relu(baseline+s))
    zbase=m.seg_head(F.relu(baseline))
    additive=zd+zs-zbase
    residual=additive-z
    reverse=z-residual
    shuffled=z+torch.roll(residual,residual.shape[-1]//2,dims=-1)
    logits=torch.cat((z,additive,zd,zs,reverse,shuffled),0).cpu().numpy()
    assert np.isfinite(logits).all()
    diag=np.array([float(residual.abs().mean()),float((additive-shuffled).abs().mean()),float((deep+skip+bias-captures['up5']).abs().max()),float((baseline+d+s-u).abs().max())])
    return logits,diag,None

def check():
    # Mixed finite difference cancels for affine readouts; ReLU can produce either sign.
    b=np.array([1.,-1.,0.]);d=np.array([-2.,1.,2.]);s=np.array([-2.,1.,-1.])
    affine=lambda x:2*x+3
    assert np.array_equal(affine(b+d)+affine(b+s)-affine(b),affine(b+d+s))
    f=lambda x:np.maximum(x,0)
    add=f(b+d)+f(b+s)-f(b);native=f(b+d+s)
    assert np.any(native-add>0) and np.any(native-add<0)
    assert np.array_equal(f(b+d)+f(b)-f(b),f(b+d))
    residual=add-native
    assert np.isclose(np.linalg.norm(np.roll(residual,1)),np.linalg.norm(residual))
