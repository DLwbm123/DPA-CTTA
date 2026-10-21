"""Immutable six-view source target; original C2 update and readout."""
import math
import torch
from torch.nn import functional as F
from plain import Plain,close_enough
from dpa_ctta.b1_host import official,finite
from dpa_ctta.hosts.vptta import model_input_from_pixels
from dpa_ctta.r19_model_only.method import equal

def mix(q0,qt,cell):
    if cell=='STATIC':return q0
    if cell=='HALF':return (q0+qt)*.5
    if cell=='CURRENT':return qt
    assert cell=='RELIABLE'
    def confidence(q):
        p=q.clamp(1e-7,1-1e-7)
        return (1+(p*p.log()+(1-p)*torch.log1p(-p))/math.log(2)).clamp(0,1)+1e-6
    a,b=confidence(q0),confidence(qt)
    return (a*q0+b*qt)/(a+b)

class Anchored(Plain):
    def __init__(self,state,cell,seed,identity,**kw):
        assert cell in ('STATIC','HALF','RELIABLE','CURRENT')
        self.anchor_cell=cell;self.q0=None
        super().__init__(state,'C2',seed,identity,**kw)
        self.initial=self.snapshot()
    @torch.no_grad()
    def source_target(self,x,verify=False):
        saved=self.snapshot()
        try:
            self.restore(self.initial);self.collect=False
            z=model_input_from_pixels(x,'fundus').to(self.device)
            api=official().Rotate_and_Flip()
            logits=[self.native.model(z)[0].detach()]
            for factor in range(5):logits.append(api.inverse(self.native.model(api(z,factor))[0],factor).detach())
            q=torch.stack(logits).sigmoid().mean(0).detach()
        finally:self.restore(saved)
        if verify:assert equal(saved,self.snapshot()),'source observation changed persistent state'
        finite(q);return q
    def keyed_step(self,x,raw):
        self.q0=self.source_target(x)
        try:return super().keyed_step(x,raw)
        finally:self.q0=None
    def _criterion(self,z,q):
        assert len(self.weak)==6 and torch.equal(torch.stack(self.weak).sigmoid().mean(0).to(q),q)
        assert self.q0 is not None and not self.q0.requires_grad and not q.requires_grad
        target=mix(self.q0.to(q),q,self.anchor_cell).detach();finite(target)
        assert bool(((target>=0)&(target<=1)).all())
        self.target_cache=(target.cpu().clone(),None)
        self.extra=True
        try:other=torch.flip(self.native.model(torch.flip(self.strong,(-1,)))[0],(-1,))
        finally:self.extra=False;self.strong=None
        self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
        loss=(F.binary_cross_entropy_with_logits(z,target)+F.binary_cross_entropy_with_logits(other,target))*.5
        self.diag.update(joint_loss=float(loss.detach()),pseudo_loss=float(loss.detach()),LSO=0.,source_weight_cell=self.anchor_cell,target_shift=float((target-q).abs().mean()))
        finite(loss);return loss

def self_check():
    p=torch.tensor([0.,.1,.5,.9,1.]);q=1-p
    for cell in ('STATIC','HALF','RELIABLE','CURRENT'):
        a=mix(p,q,cell);assert torch.isfinite(a).all() and bool(((a>=0)&(a<=1)).all())
        assert torch.allclose(mix(p,p,cell),p,atol=1e-7)
    assert torch.allclose(mix(p,q,'RELIABLE'),mix(q,p,'RELIABLE'),atol=1e-7)
    assert mix(torch.tensor([.99]),torch.tensor([.5]),'RELIABLE').item()>.98
