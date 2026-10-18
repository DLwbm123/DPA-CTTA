"""DSR supervision transferred to two nested sigmoid segmentation outputs."""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as Retained, candidates
from ..b1_host import official
from ..hosts.vptta import model_input_from_pixels
from ..r7_shared.context import tensor_digest

ARMS=('NATIVE_BN','NATIVE_DEC','DSR_BN','DSR_DEC')


def hard_loss(z,q,threshold):
    q=q.detach();positive=q>=threshold;negative=q<=.3;valid=positive|negative
    p=z.sigmoid();pn=(1-p+.05).clamp(max=1.)
    terms=torch.where(positive,F.softplus(-z),-(1-pn).square()*pn.clamp_min(1e-8).log())
    return (terms*valid).sum()/valid.sum().clamp_min(1),positive,negative


def discrepancy(z,q):
    q=q.detach().clamp(1e-6,1-1e-6)
    return (q*(q.log()-F.logsigmoid(z))+(1-q)*((1-q).log()-F.logsigmoid(-z))).mean().detach()


class Host(Retained):
    def __init__(self,state,arm,seed,identity,**kwargs):
        if arm not in ARMS:raise ValueError('unknown registered arm')
        self.arm=arm;self.dsr=arm.startswith('DSR')
        config=copy.deepcopy(candidates()[3])
        if self.dsr:
            config.update(id=arm,family='control',params={},module=None,
                          components={'T':dict(teacher_mix=1.,ema_alpha=.999)})
        super().__init__(state,config,seed,identity,**kwargs)
        n=self.native;self.bn_names=list(n.names)
        if arm.endswith('DEC'):
            extra=[(k,p) for k,p in n.model.named_parameters() if k in ('up5.weight','up5.bias')]
            if [k for k,p in extra]!=['up5.weight','up5.bias'] or sum(p.numel() for k,p in extra)!=32800:
                raise ValueError('verified final decoder scope required')
            for k,p in extra:
                p.requires_grad_(True);n.params.append(p);n.names.append(k);n.base.param_groups[0]['params'].append(p)
                del n.frozen[k]
            n.opt.param_groups=n.base.param_groups
            self.fixed_digest=tensor_digest([(k,v) for k,v in n.model.state_dict().items() if k not in n.names])
        self.threshold=torch.full((1,2,1,1),.8,device=self.device)
        self.reference=None;self.warm=[];self.pending=None

    @torch.no_grad()
    def _prepare_teacher(self,x):
        if not self.dsr:return
        inp=model_input_from_pixels(x,'fundus').to(self.device);transform=official().Rotate_and_Flip()
        predictions=[self.teacher(inp)[0].sigmoid()]
        for k in range(5):
            predictions.append(transform.inverse(self.teacher(transform(inp,k))[0],k).sigmoid())
        self.q_teacher=torch.stack(predictions).mean(0).detach()

    def _criterion(self,z,q_native):
        if not self.dsr:return super()._criterion(z,q_native)
        q=self.q_teacher.to(z).detach();d=float(discrepancy(z,q));old=self.reference
        route=self.visits>=20 and old is not None and d>1.2*old
        hard,pos,neg=hard_loss(z,q,self.threshold);soft=F.binary_cross_entropy_with_logits(z,q)
        next_threshold=self.threshold.clone()
        if self.visits>=20:
            for ch in range(2):
                grid=q[0,ch,8::16,8::16];foreground=grid[grid>=.5]
                if foreground.numel():
                    estimate=torch.quantile(foreground,.95).clamp(.6,.95)
                    next_threshold[0,ch].mul_(.9).add_(estimate,alpha=.1)
        self.pending=(d,next_threshold)
        self.diag.update(soft_loss=float(soft.detach()),hard_loss=float(hard.detach()),hard_active=bool(route),
            discrepancy=d,reference_before=old,positive_threshold=self.threshold.flatten().tolist(),
            positive_fraction=pos.float().mean((0,2,3)).tolist(),negative_fraction=neg.float().mean((0,2,3)).tolist(),
            teacher_forwards=6)
        return soft+.25*int(route)*hard

    def _lr(self,opt,args,kwargs):
        super()._lr(opt,args,kwargs)
        ps=dict(self.native.model.named_parameters())
        self.diag['BN_only_gradient_norm']=float(torch.stack([ps[k].grad.norm() for k in self.bn_names]).norm())
        self.diag['decoder_gradient_norm']=float(torch.stack([ps[k].grad.norm() for k in self.native.names if k not in self.bn_names]).norm()) if self.arm.endswith('DEC') else 0.

    def step(self,x):
        z,detail=super().step(x)
        if self.pending is not None:
            d,threshold=self.pending
            if self.visits<=20:
                self.warm.append(d)
                if self.visits==20:self.reference=sum(self.warm)/20
            else:self.reference=.999*self.reference+.001*d
            self.threshold.copy_(threshold);self.pending=None
        return z,detail

    def snapshot(self):
        s=super().snapshot()
        s['dsr_state']=dict(threshold=self.threshold.detach().cpu().clone(),reference=self.reference,warm=self.warm.copy())
        return s

    def restore(self,s):
        super().restore(s);v=s['dsr_state'];self.threshold.copy_(v['threshold'].to(self.threshold))
        self.reference=v['reference'];self.warm=v['warm'].copy();self.pending=None


def self_check():
    torch.manual_seed(77)
    z=torch.randn(1,2,4,4,requires_grad=True);q=torch.full_like(z,.5,requires_grad=True)
    loss,positive,negative=hard_loss(z,q,torch.tensor(.8));loss.backward()
    assert torch.equal(z.grad,torch.zeros_like(z)) and q.grad is None and not positive.any() and not negative.any()
    z=torch.tensor([[[[0.,0.,0.]]]],requires_grad=True);q=torch.tensor([[[[.9,.1,.5]]]],requires_grad=True)
    loss,p,n=hard_loss(z,q,torch.tensor(.8));loss.backward()
    assert z.grad[0,0,0,0]<0 and z.grad[0,0,0,1]>0 and z.grad[0,0,0,2]==0 and q.grad is None
    assert discrepancy(z,q)>=0
    # Exercise causal routing and threshold updates without a model or target data.
    h=object.__new__(Host);h.dsr=True;h.device=torch.device('cpu');h.visits=0
    h.threshold=torch.full((1,2,1,1),.8);h.reference=None;h.warm=[];h.pending=None;h.diag={}
    h.q_teacher=torch.full((1,2,32,32),.9)
    for i in range(20):
        h.visits=i;h._criterion(torch.zeros_like(h.q_teacher,requires_grad=True),None)
        assert not h.diag['hard_active'] and torch.equal(h.threshold,torch.full_like(h.threshold,.8))
        h.warm.append(h.pending[0])
    h.reference=sum(h.warm)/20;h.visits=20
    h._criterion(torch.full_like(h.q_teacher,-3.,requires_grad=True),None)
    assert h.diag['hard_active'] and h.reference==sum(h.warm)/20
    assert torch.equal(h.threshold,torch.full_like(h.threshold,.8)) and (h.pending[1]>h.threshold).all()
    print('DSR_LOSS_AND_CAUSAL_ROUTING_SELF_CHECK_PASS',flush=True)
