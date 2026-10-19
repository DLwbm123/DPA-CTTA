"""HRAC-inspired stop-gradient mirror distillation for two Bernoulli heads."""
import math
import torch
from torch.nn import functional as F
from .method import Host as AffineHost, entropy, l2, normalize_parameters

KINDS=('FLIP','ROLL','SOURCE_FLIP','ENT_NORM')


def aligned_teacher(logits,rolled=False):
    teacher=logits.detach().flip(-1)
    return teacher.roll(teacher.shape[-1]//2+1,-1) if rolled else teacher


def bernoulli_kl(student,teacher):
    if student.shape!=teacher.shape or student.ndim!=4 or student.shape[:2]!=(1,2):
        raise ValueError('independent two-head one-image logits required')
    target=teacher.detach().double();q=target.sigmoid();z=student.double()
    return (q*(F.logsigmoid(target)-F.logsigmoid(z))+
        (1-q)*(F.logsigmoid(-target)-F.logsigmoid(-z))).mean()


class Host(AffineHost):
    def __init__(self,base,kind):
        if kind not in KINDS:raise ValueError('unknown correspondence control')
        super().__init__(base,'ENT_NORM');self.kind=kind
        # Frozen source is used only by the predeclared teacher control.
        self.source=base

    def step(self,x,xw,xs,z0,seed):
        self.opt.zero_grad(set_to_none=True)
        if self.kind=='ENT_NORM':teacher=None
        else:
            with torch.no_grad():
                model=self.source if self.kind=='SOURCE_FLIP' else self.model
                out=model(x.flip(-1));zt=out[0] if isinstance(out,(tuple,list)) else out
                teacher=aligned_teacher(zt,self.kind=='ROLL')
        z=self.forward(x);pre=z.detach().clone()
        loss=(entropy(z.sigmoid())*math.log(2)).mean() if teacher is None else bernoulli_kl(z,teacher)
        if not torch.isfinite(loss) or (teacher is not None and float(loss.detach())<=1e-12):
            raise ValueError('nonfinite or degenerate correspondence signal; no fallback')
        loss.backward()
        if any(p.grad is None for p in self.params) or not all(bool(torch.isfinite(p.grad).all()) for p in self.params):
            raise ValueError('invalid affine gradients')
        gradient=l2([p.grad for p in self.params])
        if gradient<=1e-12:raise ValueError('zero affine gradient; no arbitrary normalized step')
        before=[p.detach().clone() for p in self.params];self.opt.step()
        with torch.no_grad():
            raw,actual,cosine=normalize_parameters(self.params,before)
            post=self.forward(x).detach()
        if raw<=1e-12:raise ValueError('zero parameter proposal')
        self.total_steps+=1;self.segment_steps+=1
        names,buffers=dict(self.model.named_parameters()),dict(self.model.named_buffers())
        if any(names[n]._version!=v for n,v in self.fixed.items()):raise ValueError('fixed parameter changed')
        if any(not torch.equal(v,buffers[n]) for n,v in self.buffers.items()):raise ValueError('BN buffers changed')
        if any(int(self.opt.state[p]['step'])!=self.segment_steps for p in self.params):raise ValueError('Adam age differs')
        with torch.no_grad():
            after=float(bernoulli_kl(post,teacher)) if teacher is not None else float((entropy(post.sigmoid())*math.log(2)).mean())
            disagreement=float((z.detach().sigmoid()-teacher.sigmoid()).abs().mean()) if teacher is not None else 0.
        log=dict(loss=float(loss.detach()),entropy=float((entropy(pre.sigmoid())*math.log(2)).mean()),ranking=0.,anchor=0.,anchor_gradient_norm=0.,raw_step_norm=raw,actual_step_norm=actual,direction_cosine=cosine,norm_matched=abs(actual-.01)<=5e-7,zero_proposal=False,gate_fraction=1.,affine_gradient_norm=gradient,teacher_disagreement=disagreement,fixed_teacher_loss_post=after)
        return pre,post,torch.ones_like(pre,dtype=torch.bool),pre,dict(steps=[log],total_affine_norm=l2([p-v for p,v in zip(self.params,self.initial)]),optimizer_age=self.segment_steps,forwards=2+(teacher is not None),optimizer_steps=1,loss_backward_calls=1,diagnostic_anchor_gradient_calls=0)
