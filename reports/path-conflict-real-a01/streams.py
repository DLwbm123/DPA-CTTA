"""Cancel only the skip component opposed to native deep-path evidence."""
import numpy as np

class Memory:
    def __init__(self): self.ids=[]
    def add(self,i,proposals):
        assert i not in self.ids and len(proposals)==4
        self.ids.append(i)

def directions(d,s,u):
    import torch
    norm=torch.linalg.vector_norm(d,dim=1,keepdim=True)
    unit=torch.where(norm>0,d/norm.clamp_min(1e-12),torch.zeros_like(d))
    # The minimum correction that makes skip's contextual projection nonnegative.
    amount=(-torch.sum(s*unit,dim=1,keepdim=True)).clamp_min(0)
    residual=amount*unit
    def reorient(v):
        n=torch.linalg.vector_norm(v,dim=1,keepdim=True)
        fallback=torch.zeros_like(v);fallback[:,0]=1
        unit_v=torch.where(n>1e-12,v/n.clamp_min(1e-12),fallback)
        return amount*unit_v
    radial=reorient(u)
    shuffled=reorient(torch.roll(residual,residual.shape[-1]//2,dims=-1))
    for control in (radial,shuffled):
        assert torch.allclose(torch.linalg.vector_norm(control,dim=1),torch.linalg.vector_norm(residual,dim=1),rtol=2e-5,atol=2e-5)
    return residual,radial,shuffled,amount

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
    # Preserve native numerical output; use split convolutions only for the correction.
    residual,radial,shuffled,amount=directions(d,s,u)
    main=m.seg_head(F.relu(u+residual));mag=m.seg_head(F.relu(u+radial))
    reverse=m.seg_head(F.relu(u-residual));null=m.seg_head(F.relu(u+shuffled))
    logits=torch.cat((z,main,mag,reverse,null),0).cpu().numpy()
    assert np.isfinite(logits).all()
    diag=np.array([float((amount>0).float().mean()),float(amount.mean()),float((main-z).abs().mean()),float((main-null).abs().mean()),float((deep+skip+bias-captures['up5']).abs().max())])
    return logits,diag,None

def check():
    # Projection preserves the orthogonal skip component and is not mere scaling.
    d=np.array([[1.,0.],[0.,2.],[0.,0.],[1.,1.]])
    s=np.array([[-2.,3.],[4.,1.],[2.,-3.],[-2.,0.]])
    norm=np.linalg.norm(d,axis=1,keepdims=True);unit=np.divide(d,norm,out=np.zeros_like(d),where=norm>0)
    coeff=(s*unit).sum(1,keepdims=True);r=np.maximum(-coeff,0)*unit
    assert np.all(((s+r)*unit).sum(1)>=-1e-12)
    assert np.allclose((s+r)-((s+r)*unit).sum(1,keepdims=True)*unit,s-coeff*unit)
    assert np.array_equal(r[1:3],np.zeros((2,2)))
    assert np.array_equal(s[0]+r[0],[0.,3.])
    assert np.linalg.norm(r[0])==np.linalg.norm(-r[0])
