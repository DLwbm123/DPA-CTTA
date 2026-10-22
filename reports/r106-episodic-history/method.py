"""Only remove W/LSO; retain original six-weak and keyed-strong machinery."""
import torch
import numpy as np
from torch.nn import functional as F
from dpa_ctta.r79_mechanism_control.method import Host as Previous
from dpa_ctta.r37_cw_orientation.method import Host as LossHost
from dpa_ctta.b1_host import finite

class Plain(Previous):
    def __init__(self,state,cell,seed,identity,**kwargs):
        assert cell in ('C1','C2','DUP')
        self.cell=cell
        super().__init__(state,'NATIVE' if cell=='C1' else 'VERTEX',seed,identity,**kwargs)

    def _criterion(self,z,q):
        assert len(self.weak)==6 and torch.equal(torch.stack(self.weak).sigmoid().mean(0).to(q),q)
        assert not q.requires_grad
        self.target_cache=(q.detach().cpu().clone(),None)
        if self.cell=='C1':
            self._tag(z,'original',1.)
            loss=F.binary_cross_entropy_with_logits(z,q)
        else:
            assert self.strong is not None
            self.extra=True
            try:
                other=self.native.model(self.strong if self.cell=='DUP' else torch.flip(self.strong,(-1,)))[0]
                if self.cell!='DUP':other=torch.flip(other,(-1,))
            finally:self.extra=False;self.strong=None
            self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
            loss=(F.binary_cross_entropy_with_logits(z,q)+F.binary_cross_entropy_with_logits(other,q))*.5
        self.diag.update(joint_loss=float(loss.detach()),pseudo_loss=float(loss.detach()),LSO=0.)
        finite(loss);return loss

class DuplicateFull(Previous):
    def __init__(self,state,seed,identity,**kwargs):
        super().__init__(state,'VERTEX',seed,identity,**kwargs)
    def _criterion(self,z,q):
        assert self.strong is not None
        self.extra=True
        try:other=self.native.model(self.strong)[0]
        finally:self.extra=False;self.strong=None
        self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
        loss=(LossHost._criterion(self,z,q)+LossHost._criterion(self,other,q))*.5
        self.diag['joint_loss']=float(loss.detach());return loss

def plain_oracle(host):
    def criterion(z,q):
        host.extra=True
        try:b=torch.flip(host.native.model(torch.flip(host.strong,(-1,)))[0],(-1,))
        finally:host.extra=False;host.strong=None
        host._tag(z,'vertex0',.5);host._tag(b,'vertex1',.5)
        loss=sum(F.binary_cross_entropy_with_logits(v,q,reduction='sum')/q.numel() for v in (z,b))/2
        host.diag['joint_loss']=float(loss.detach());return loss
    return criterion

def close_enough(a,b):
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        return torch.allclose(a,b,atol=2e-6,rtol=1e-4) if a.is_floating_point() else torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(close_enough(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(close_enough(x,y) for x,y in zip(a,b))
    if isinstance(a,float):return abs(a-b)<=2e-6+1e-4*abs(b)
    return a==b

def self_check():
    torch.manual_seed(102)
    z=torch.randn(1,2,9,9,requires_grad=True);q=torch.rand_like(z)
    a=F.binary_cross_entropy_with_logits(z,q);b=(F.binary_cross_entropy_with_logits(z,q)+F.binary_cross_entropy_with_logits(z,q))*.5
    assert torch.equal(a,b) and torch.equal(torch.autograd.grad(a,z,retain_graph=True)[0],torch.autograd.grad(b,z)[0])
    assert close_enough({'x':torch.ones(3),'n':2},{'x':torch.ones(3),'n':2})
    assert not close_enough(torch.zeros(3),torch.ones(3))


def reset_episode(h,initial):
    import copy
    current=h.snapshot();reset=copy.deepcopy(initial)
    for key in ('rng','native_rng'):reset['base'][key]=current['base'][key]
    h.restore(reset)
    actual=h.snapshot()
    assert close_enough(actual,reset), 'incomplete episode restore'
    assert h.native.steps==0 and not h.native.base.state, 'Adam not empty'
