"""Original two views, current q/whole W; independently choose loss aggregation."""
import torch
from torch.nn import functional as F
from ..r79_mechanism_control.method import Host as Previous,SEEDS,style
from ..r20_model_only_search.method import weights_for
from ..r36_incremental_modules.method import local_structure_loss,smooth,structure
from ..b1_host import finite
CELLS=('VV','VB','BV','BB')


def components(z,q,w):
    return (F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).mean(),local_structure_loss(z,q,w)[0]


def channel_components(z,q,w):
    # Channel contributions SUM to the original objective; never renormalize a channel.
    b=(F.binary_cross_entropy_with_logits(z,q,reduction='none')*w).sum((0,2,3))/z.numel()
    po,_=structure(smooth(F.avg_pool2d(z.sigmoid(),4)))
    to,tm=structure(smooth(F.avg_pool2d(q.detach(),4)))
    tn=tm/tm.amax((-2,-1),keepdim=True).clamp_min(1e-6)
    weight=(tn*F.avg_pool2d(w.detach(),4)*(tm>2e-6)).detach()
    s=((1-(po*to).sum(2).clamp(-1,1))*weight).sum((0,2,3))/weight.sum().clamp_min(1e-6)
    return b,s


def combine(z,t,q,w,cell):
    b0,s0=components(z,q,w);b1,s1=components(t,q,w);bm,sm=components((z+t)*.5,q,w)
    bv=(b0+b1)*.5;sv=(s0+s1)*.5
    # Preserve the retained endpoints' actual floating point order.
    loss=((b0+.1*s0)+(b1+.1*s1))*.5 if cell=='VV' else bm+.1*sm if cell=='BB' else (bv if cell[0]=='V' else bm)+.1*(sv if cell[1]=='V' else sm)
    return loss,dict(BCE=float((bv if cell[0]=='V' else bm).detach()),LSO=float((sv if cell[1]=='V' else sm).detach()),D_B=float((bv-bm).detach()),D_S=float((sv-sm).detach()))


class Host(Previous):
    def __init__(self,state,cell,seed,identity,**kwargs):
        if cell not in CELLS:raise ValueError('unregistered loss cell')
        self.cell=cell;self.probe=False;self.component_gradients={}
        super().__init__(state,'VERTEX' if cell=='VV' else 'BARY',seed,identity,**kwargs)

    def _criterion(self,z,q):
        if self.strong is None or len(self.weak)!=6:raise ValueError('original two strong/six weak contract')
        pred=torch.stack(self.weak).sigmoid();w=weights_for(pred,.1,2.).to(q)
        if not torch.equal(pred.mean(0).to(q),q):raise ValueError('current q provenance')
        self.target_cache=(q.detach().cpu().clone(),w.detach().cpu().clone())
        self.extra=True
        try:other=torch.flip(self.native.model(torch.flip(self.strong,(-1,)))[0],(-1,))
        finally:self.extra=False;self.strong=None
        if self.arm=='VERTEX':self._tag(z,'vertex0',.5);self._tag(other,'vertex1',.5)
        else:self._tag(z,'mixed0',1.);self._tag(other,'mixed1',1.)
        if self.probe:
            before=self.native.counts.copy()
            b0,s0=channel_components(z,q,w);b1,s1=channel_components(other,q,w);bm,sm=channel_components((z+other)*.5,q,w)
            vals={'B_V':(b0+b1)*.5,'B_B':bm,'S_V':(s0+s1)*.5,'S_B':sm}
            self.component_gradients={}
            for name,v in vals.items():
                for ch in (0,1):
                    if self.charge:self.charge('diagnostic_backward')
                    g=torch.autograd.grad(v[ch],self.native.params,retain_graph=True)
                    finite(g);self.component_gradients[name+str(ch)]=torch.cat([x.detach().flatten().cpu() for x in g])
            # Observer hook counters belong to the native one-update contract; physical
            # diagnostic traversals are separately charged outside every snapshot.
            self.native.counts=before
        loss,parts=combine(z,other,q,w,self.cell)
        self.diag.update(loss_parts=parts,joint_loss=float(loss.detach()),q_source='current',W_source='current_whole')
        finite(loss);return loss


def oracle(host,cell):
    """Independent direct original formulas; no use of combine/channel helpers."""
    def criterion(z,q):
        w=weights_for(torch.stack(host.weak).sigmoid(),.1,2.).to(q)
        host.extra=True
        try:t=torch.flip(host.native.model(torch.flip(host.strong,(-1,)))[0],(-1,))
        finally:host.extra=False;host.strong=None
        def b(v):return (F.binary_cross_entropy_with_logits(v,q,reduction='none')*w).mean()
        def s(v):return local_structure_loss(v,q,w)[0]
        if cell=='VV':return ((b(z)+.1*s(z))+(b(t)+.1*s(t)))*.5
        if cell=='BB':return b((z+t)*.5)+.1*s((z+t)*.5)
        return ((b(z)+b(t))*.5 if cell[0]=='V' else b((z+t)*.5))+.1*((s(z)+s(t))*.5 if cell[1]=='V' else s((z+t)*.5))
    return criterion


def self_check():
    from ..r79_mechanism_control.method import self_check as previous
    previous();torch.manual_seed(85)
    z=torch.randn(1,2,32,32,requires_grad=True);t=torch.randn_like(z,requires_grad=True);q=torch.rand_like(z);w=torch.rand_like(z)+.1
    losses={c:combine(z,t,q,w,c)[0] for c in CELLS}
    for value in (z,t):
        gs={c:torch.autograd.grad(v,value,retain_graph=True)[0] for c,v in losses.items()}
        assert torch.allclose(gs['VV']-gs['VB']-gs['BV']+gs['BB'],torch.zeros_like(value),atol=2e-6,rtol=1e-4)
    for a,b in zip(channel_components(z,q,w),components(z,q,w)):
        assert torch.allclose(a.sum(),b,atol=2e-6,rtol=1e-4)
    gap=combine(z,t,q,w,'VV')[1]['D_B'];gap2=combine(z,t,1-q,w,'VV')[1]['D_B']
    assert gap>=-2e-6 and abs(gap-gap2)<2e-6
    equal_losses=[combine(z,z,q,w,c)[0] for c in CELLS]
    assert all(torch.allclose(v,equal_losses[0],atol=2e-6,rtol=1e-4) for v in equal_losses)
    assert 1152+48*3+2304==3600
    print('LOSS_FACTORIAL_CHANNEL_DENOMINATOR_JENSEN_GRADIENT_CPU_PASS',flush=True)
