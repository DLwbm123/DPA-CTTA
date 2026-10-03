"""EVA-0-inspired two-forward feature adapter, independent of flip teachers."""
import hashlib
import math
import torch

SEED=20261003
COMBINATIONS=((1e-3,1e-4),(1e-3,5e-4),(3e-3,1e-4),(3e-3,5e-4))
KINDS=('Z_PROBE','Z_CORE','Z_BOUND')

def directions(seed,order,step,w,kind):
    key=f'{seed}|{order}|{step}'.encode();gen=torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8],'little')%(2**63-1))
    u=torch.randint(0,2,w.shape,generator=gen).float()*2-1
    if kind=='Z_BOUND' and step%4==0 and float(w.norm())>0:
        v=2*torch.rand(w.shape,generator=gen)
        u=math.sqrt(w.numel())*(-w)/(w.norm()+1e-6)*v
    return u

def losses(zp,zm,roi=None):
    if zp.shape!=zm.shape or zp.shape[1]!=2:raise ValueError('two independent Bernoulli channels required')
    bar=(zp+zm)/2;roi=torch.ones_like(bar,dtype=torch.bool) if roi is None else roi.expand_as(bar)
    values=[]
    for z in (zp,zm):
        cells=[]
        for c in range(2):
            rbar=(bar[:,c][roi[:,c]].square().mean()+1e-12).sqrt()
            rz=(z[:,c][roi[:,c]].square().mean()+1e-12).sqrt()
            s=z[:,c]*rbar/rz;p=s.sigmoid();entropy=-(p*torch.log(p.clamp_min(1e-6))+(1-p)*torch.log((1-p).clamp_min(1e-6)))
            foreground=(bar[:,c].sigmoid()>=.5).detach();groups=[]
            for group in (foreground,~foreground):
                valid=group&roi[:,c]
                if valid.any():groups.append(entropy[valid].mean())
            if not groups:raise ValueError('empty effective ROI')
            cells.append(torch.stack(groups).mean())
        values.append(torch.stack(cells).mean())
    return values

def update(w,u,lp,lm,mu,eta,kind):
    if kind=='Z_PROBE':return torch.zeros_like(w),0.
    g=((lp-lm)/(2*mu))*u
    if not torch.isfinite(g).all():raise ValueError('nonfinite zero-order gradient')
    norm=g.norm();g=g/max(float(norm),1.);w=w-eta*g
    if kind=='Z_BOUND':
        w=w*(1-1e-3);rms=w.norm()/math.sqrt(w.numel())
        if rms>.05:w=w*(.05/rms)
    if not torch.isfinite(w).all():raise ValueError('nonfinite adapter')
    return w,float(norm)

class Adapter:
    def __init__(self,segmenter,kind,mu,eta,seed=SEED,order=0):
        if kind not in KINDS or (mu,eta) not in COMBINATIONS:raise ValueError('unregistered Z condition')
        self.seg=segmenter;self.kind=kind;self.mu=mu;self.eta=eta;self.seed=seed;self.order=order
        self.w=torch.zeros(512);self.inject=self.w;self.visits=0
        self.handle=dict(segmenter.model.named_modules())['up3'].register_forward_hook(self._hook)

    def _hook(self,module,args,h):
        if h.shape[1]!=256:raise ValueError('adapter layer channels')
        if not torch.isfinite(self.inject).all():raise ValueError('nonfinite injection')
        if not self.inject.any():return h
        a,b=self.inject.to(h).chunk(2)
        rms=(h.detach().float().square().mean((-2,-1),keepdim=True)+1e-12).sqrt().to(h)
        return (1+a[None,:,None,None])*h+b[None,:,None,None]*rms

    @torch.no_grad()
    def step(self,image):
        step=self.visits+1;u=directions(self.seed,self.order,step,self.w,self.kind)
        center=self.w.clone();before=self.seg.forwards
        try:
            self.inject=center+self.mu*u;zp=self.seg(image).float()
            self.inject=center-self.mu*u;zm=self.seg(image).float()
        finally:self.inject=center
        if self.seg.forwards-before!=2:raise ValueError('Z must perform exactly two forwards')
        if not torch.isfinite(zp).all() or not torch.isfinite(zm).all():raise ValueError('nonfinite probes')
        output=(zp+zm)/2
        lp,lm=losses(zp,zm);new,gnorm=update(center,u,lp,lm,self.mu,self.eta,self.kind)
        self.w=new;self.inject=new;self.visits=step
        return output,dict(visit=step,loss_plus=float(lp),loss_minus=float(lm),gradient_norm=gnorm,
                          adapter_rms=float(new.norm()/math.sqrt(new.numel())),updates=int(self.kind!='Z_PROBE'),
                          full_forwards=2,output_before_update=True,update_skipped=bool(self.kind!='Z_PROBE' and gnorm==0))

    def snapshot(self):
        return dict(w=self.w.clone(),visits=self.visits,kind=self.kind,mu=self.mu,eta=self.eta,seed=self.seed,order=self.order)

    def restore(self,x):
        if any(x[k]!=getattr(self,k) for k in ('kind','mu','eta','seed','order')) or x['w'].shape!=(512,) or not torch.isfinite(x['w']).all():raise ValueError('Z snapshot identity')
        self.w=x['w'].clone();self.inject=self.w;self.visits=x['visits']

    def close(self):self.handle.remove()
