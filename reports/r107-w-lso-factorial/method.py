"""Matched functional W/LSO ablations; keep each branch's endogenous q and history."""
import torch
from torch.nn import functional as F
from dpa_ctta.r79_mechanism_control.method import Host as Previous
from dpa_ctta.r20_model_only_search.method import weights_for
from dpa_ctta.r36_incremental_modules.method import local_structure_loss
from dpa_ctta.r85_loss_aggregation.method import combine
from dpa_ctta.b1_host import finite
CELLS=('C2','W_ONLY','LSO_ONLY','VV')

def loss_at(z,q,w,cell):
    b=(F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).mean() if cell in ('W_ONLY','VV') else F.binary_cross_entropy_with_logits(z,q)
    s=local_structure_loss(z,q,w)[0] if cell in ('LSO_ONLY','VV') else z.new_zeros(())
    return b+.1*s if cell in ('LSO_ONLY','VV') else b,b,s

class Factorial(Previous):
    def __init__(self,state,cell,seed,identity,**kw):
        assert cell in CELLS;self.cell=cell
        super().__init__(state,'VERTEX',seed,identity,**kw)
    def _criterion(self,z,q):
        pred=torch.stack(self.weak).sigmoid();assert len(self.weak)==6 and torch.equal(pred.mean(0).to(q),q)
        w=weights_for(pred,.1,2.).to(q) if self.cell in ('W_ONLY','VV') else torch.ones_like(q)
        self.target_cache=(q.detach().cpu().clone(),w.detach().cpu().clone())
        assert self.strong is not None
        self.extra=True
        try:t=torch.flip(self.native.model(torch.flip(self.strong,(-1,)))[0],(-1,))
        finally:self.extra=False;self.strong=None
        self._tag(z,'vertex0',.5);self._tag(t,'vertex1',.5)
        a,b0,s0=loss_at(z,q,w,self.cell);b,b1,s1=loss_at(t,q,w,self.cell);loss=(a+b)*.5
        self.diag.update(joint_loss=float(loss.detach()),BCE=float(((b0+b1)*.5).detach()),LSO=float(((s0+s1)*.5).detach()),weight_mean=float(w.mean()),weight_std=float(w.std()),q_source='own_current',W_source='own_current' if self.cell in ('W_ONLY','VV') else 'ones')
        finite(loss);return loss

def oracle(host,cell):
    def criterion(z,q):
        pred=torch.stack(host.weak).sigmoid();assert torch.equal(pred.mean(0).to(q),q)
        w=weights_for(pred,.1,2.).to(q) if cell=='W_ONLY' else torch.ones_like(q)
        host.target_cache=(q.detach().cpu().clone(),w.detach().cpu().clone());host.extra=True
        try:t=torch.flip(host.native.model(torch.flip(host.strong,(-1,)))[0],(-1,))
        finally:host.extra=False;host.strong=None
        host._tag(z,'vertex0',.5);host._tag(t,'vertex1',.5)
        if cell=='W_ONLY':
            loss=((F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).mean()+(F.binary_cross_entropy_with_logits(t,q,reduction='none')*w).mean())*.5
        else:
            # Reliability is neutral; q-derived orientation/support and global denominator stay intact.
            loss=((F.binary_cross_entropy_with_logits(z,q)+.1*local_structure_loss(z,q,w)[0])+(F.binary_cross_entropy_with_logits(t,q)+.1*local_structure_loss(t,q,w)[0]))*.5
        host.diag['joint_loss']=float(loss.detach());return loss
    return criterion

def self_check():
    torch.manual_seed(107)
    z=torch.randn(1,2,32,32,requires_grad=True);t=torch.randn_like(z,requires_grad=True);q=torch.rand_like(z);w=torch.rand_like(z)+.1
    for cell in CELLS:
        ww=w if cell in ('W_ONLY','VV') else torch.ones_like(w)
        got=(loss_at(z,q,ww,cell)[0]+loss_at(t,q,ww,cell)[0])*.5
        if cell=='C2':expected=(F.binary_cross_entropy_with_logits(z,q)+F.binary_cross_entropy_with_logits(t,q))*.5
        elif cell=='VV':expected=combine(z,t,q,w,'VV')[0]
        elif cell=='W_ONLY':expected=((F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).mean()+(F.binary_cross_entropy_with_logits(t,q,reduction='none')*w).mean())*.5
        else:expected=((F.binary_cross_entropy_with_logits(z,q)+.1*local_structure_loss(z,q,torch.ones_like(w))[0])+(F.binary_cross_entropy_with_logits(t,q)+.1*local_structure_loss(t,q,torch.ones_like(w))[0]))*.5
        assert torch.equal(got,expected)
        assert all(torch.equal(a,b) for a,b in zip(torch.autograd.grad(got,(z,t),retain_graph=True),torch.autograd.grad(expected,(z,t),retain_graph=True)))
    print('FOUR_FUNCTIONAL_LOSSES_AND_LOGIT_GRADIENTS_PASS',flush=True)
