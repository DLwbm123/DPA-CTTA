"""MARS-inspired dense independent-Bernoulli loss, without anatomy or source priors."""
import math
import torch
from torch.nn import functional as F
from .method import Host as AffineHost, entropy, l2, normalize_parameters

KINDS=('MARS','MASK_HARD','FIXED_RHO','ENT_NORM')


def masked_views(x,seed):
    if x.ndim!=4 or x.shape[0]!=1 or x.shape[2]%4 or x.shape[3]%4:
        raise ValueError('one image divisible into 4x4 patches required')
    g=torch.Generator(device='cpu').manual_seed(seed)
    total=(x.shape[2]//4)*(x.shape[3]//4)
    for _ in range(2):
        rho=float(torch.rand((),generator=g)*.2+.2)
        selected=torch.randperm(total,generator=g)[:int(total*rho)]
        mask=torch.ones(total,dtype=x.dtype)
        mask[selected]=0
        mask=mask.reshape(1,1,x.shape[2]//4,x.shape[3]//4).repeat_interleave(4,2).repeat_interleave(4,3).to(x)
        yield x*mask,rho,float((mask==0).float().mean())


def soft_target(hard,rho,kind):
    softness=rho if kind=='MARS' else (.3 if kind=='FIXED_RHO' else 0.)
    return hard*(1-softness)+(1-hard)*softness


class Host(AffineHost):
    def __init__(self,base,kind):
        if kind not in KINDS:raise ValueError('unknown frozen loss')
        super().__init__(base,'ENT_NORM');self.kind=kind

    def step(self,x,xw,xs,z0,seed):
        self.opt.zero_grad(set_to_none=True)
        z=self.forward(x);pre=z.detach().clone()
        pixel_entropy=entropy(pre.sigmoid())*math.log(2)
        gate=pixel_entropy<=pixel_entropy.mean((-2,-1),keepdim=True)
        hard=(pre>=0).to(z)
        base_loss=(entropy(z.sigmoid())*math.log(2)).mean()
        if not torch.isfinite(base_loss):raise ValueError('nonfinite entropy')
        base_loss.backward()
        rank_value=0.;ratios=[];forwards=2;backwards=1
        if self.kind!='ENT_NORM':
            for view,rho,effective in masked_views(x,seed):
                logits=self.forward(view)
                values=F.binary_cross_entropy_with_logits(logits,soft_target(hard,rho,self.kind),reduction='none')
                rank=(values*gate).sum()/gate.sum().clamp_min(1)
                if not torch.isfinite(rank):raise ValueError('nonfinite ranking loss')
                (.2*rank/2).backward()
                rank_value+=float(rank.detach())/2
                ratios.append(dict(drawn=rho,effective=effective))
                forwards+=1;backwards+=1
        if any(p.grad is None for p in self.params) or not all(bool(torch.isfinite(p.grad).all()) for p in self.params):
            raise ValueError('invalid affine gradients')
        before=[p.detach().clone() for p in self.params]
        self.opt.step()
        with torch.no_grad():
            raw,actual,cosine=normalize_parameters(self.params,before)
            post=self.forward(x).detach()
        self.total_steps+=1;self.segment_steps+=1
        if raw<=1e-12:raise ValueError('zero update proposal')
        names=dict(self.model.named_parameters());buffers=dict(self.model.named_buffers())
        if any(names[n]._version!=v for n,v in self.fixed.items()):raise ValueError('frozen parameter changed')
        if any(not torch.equal(v,buffers[n]) for n,v in self.buffers.items()):raise ValueError('BN statistics changed')
        if any(int(self.opt.state[p]['step'])!=self.segment_steps for p in self.params):raise ValueError('Adam age differs')
        log=dict(loss=float(base_loss.detach())+.2*rank_value,entropy=float(base_loss.detach()),ranking=rank_value,
                 anchor=0.,anchor_gradient_norm=0.,raw_step_norm=raw,actual_step_norm=actual,direction_cosine=cosine,
                 norm_matched=abs(actual-.01)<=5e-7,zero_proposal=False,gate_fraction=float(gate.float().mean()),mask_ratios=ratios)
        return pre,post,gate,pre,dict(steps=[log],total_affine_norm=l2([p-v for p,v in zip(self.params,self.initial)]),
            optimizer_age=self.segment_steps,forwards=forwards,optimizer_steps=1,loss_backward_calls=backwards,
            diagnostic_anchor_gradient_calls=0)
