"""Four-element flip-group loss average with original targets and one Adam step."""
import types
import torch
from torch.nn import functional as F
from plain import Plain,close_enough
from dpa_ctta.r79_mechanism_control.method import Host as Previous
from dpa_ctta.r85_loss_aggregation.method import components
from dpa_ctta.r20_model_only_search.method import weights_for
from dpa_ctta.b1_host import finite
from dpa_ctta.host_diagnostic import rng,restore

def four_step(n,x):
    assert x.device.type=='cpu' and x.dtype==torch.float32 and x.ndim==4 and len(x)==1 and not x.requires_grad
    finite(x);restore(n.rng);before=n.counts.copy();n.last={}
    data={'data':x.numpy().copy()};original=x.to(n.device).clone()
    n.opt.cal_consis_loss(data);n.base.step()
    with torch.no_grad():logits=n.model(original)[0].detach()
    finite(n.base.state);finite(n.params);n.steps+=1;n.rng=rng()
    assert all(int(n.base.state[p]['step'])==n.steps for p in n.params)
    assert all(p._version==v and p.grad is None for p,v in n.frozen.values())
    counts={k:v-before[k] for k,v in n.counts.items()}
    assert counts==dict(forwards=11,backwards=1,base_adam=1,perturb=0,restore=0),counts
    return logits,dict(counts=counts,adam_step=n.steps,lr=float(n.base.param_groups[0]['lr']),diagnostics=n.last.copy())

class Four(Previous):
    def __init__(self,state,cell,seed,identity,**kw):
        assert cell in ('C4','V4','DUPC','DUPV')
        self.cell=cell
        super().__init__(state,'VERTEX',seed,identity,**kw)
        self.native.normalized_step=types.MethodType(four_step,self.native)
    def _criterion(self,z,q):
        pred=torch.stack(self.weak).sigmoid()
        assert len(pred)==6 and torch.equal(pred.mean(0).to(q),q) and not q.requires_grad
        full=self.cell in ('V4','DUPV');w=weights_for(pred,.1,2.).to(q) if full else None
        self.target_cache=(q.detach().cpu().clone(),None if w is None else w.detach().cpu().clone())
        assert self.strong is not None
        transforms=((-1,),(),(-1,)) if self.cell.startswith('DUP') else ((-1,),(-2,),(-2,-1))
        values=[z];self.extra=True
        try:
            for axes in transforms:
                inp=torch.flip(self.strong,axes) if axes else self.strong
                v=self.native.model(inp)[0];values.append(torch.flip(v,axes) if axes else v)
        finally:self.extra=False;self.strong=None
        losses=[]
        for i,v in enumerate(values):
            self._tag(v,'vertex'+str(i),.25)
            if full:
                b,l=components(v,q,w);losses.append(b+.1*l)
            else:losses.append(F.binary_cross_entropy_with_logits(v,q))
        loss=((losses[0]+losses[1])+(losses[2]+losses[3]))*.25
        self.diag.update(joint_loss=float(loss.detach()),four_view=True,full_W_LSO=full)
        finite(loss);return loss

def self_check():
    torch.manual_seed(104)
    a=torch.randn(1,2,9,9,requires_grad=True);b=torch.randn_like(a,requires_grad=True);q=torch.rand_like(a)
    x=F.binary_cross_entropy_with_logits(a,q);y=F.binary_cross_entropy_with_logits(b,q)
    two=(x+y)*.5;four=((x+y)+(x+y))*.25
    assert torch.equal(two,four)
    for ga,gb in zip(torch.autograd.grad(two,(a,b),retain_graph=True),torch.autograd.grad(four,(a,b))):assert torch.equal(ga,gb)
    for axes in ((-1,),(-2,),(-2,-1)):assert torch.equal(a,torch.flip(torch.flip(a,axes),axes))
