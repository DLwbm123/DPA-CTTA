"""Model-only native GraTa candidate, reversible validation, and bounded FIFO."""
import copy, hashlib, io, random
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
from ..b1_host import Host as Native
from ..host_diagnostic import rng, restore as restore_native_rng
from ..hosts.vptta import model_input_from_pixels
from ..r8_ba.rng import capture, restore
from ..r7_shared.context import tensor_digest


def clone(x):
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:clone(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return type(x)(clone(v) for v in x)
    return copy.deepcopy(x)


def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b


def snapshot(n):
    external=capture();restore_native_rng(n.rng);nr=capture();restore(external)
    return dict(parameters={name:clone(p) for name,p in n.model.named_parameters() if name in n.names},
        buffers={name:clone(p) for name,p in n.model.named_buffers()},
        gradients={name:None if p.grad is None else clone(p.grad) for name,p in n.model.named_parameters()},
        adam=clone(n.base.state_dict()),grata=clone(n.opt.state_dict()),steps=n.steps,last=copy.deepcopy(n.last),counts=n.counts.copy(),
        rng=external,native_rng=nr,modes={name:m.training for name,m in n.model.named_modules()})


def restore_state(n,s):
    ps=dict(n.model.named_parameters());bs=dict(n.model.named_buffers())
    with torch.no_grad():
        for name,p in s['parameters'].items():ps[name].copy_(p.to(ps[name]))
        for name,p in s['buffers'].items():bs[name].copy_(p.to(bs[name]))
    for name,p in ps.items():p.grad=None if s['gradients'][name] is None else s['gradients'][name].to(p).clone()
    n.base.load_state_dict(copy.deepcopy(s['adam']));n.opt.load_state_dict(copy.deepcopy(s['grata']))
    n.steps=s['steps'];n.last=copy.deepcopy(s['last']);n.counts=s['counts'].copy()
    for name,m in n.model.named_modules():m.training=s['modes'][name]
    restore(s['native_rng']);n.rng=rng();restore(s['rng'])


def state_digest(s):
    # Diagnostic-only digest; never restored as a physical cost counter.
    b=io.BytesIO();torch.save(s,b);return hashlib.sha256(b.getvalue()).hexdigest()


@torch.no_grad()
def readonly(n,pixels):
    """Same train/current-statistics BN semantics, discard all forward side effects."""
    before=snapshot(n);hooks=[(h,h.features) for h in getattr(n.model,"feature_hooks",[]) if hasattr(h,"features")]
    try:return n.model(model_input_from_pixels(pixels,'fundus').to(n.device))[0].detach().float().cpu()
    finally:
        restore_state(n,before)
        for h,v in hooks:h.features=v


def probability(z):return z.float().sigmoid().clamp(1e-6,1-1e-6)


def regions(z):
    h=probability(z)>=.5
    if not torch.isfinite(z).all() or (h.sum((-2,-1))<16).any() or h.all():return None
    x=h.float();d=F.max_pool2d(x,5,1,2)>0;e=1-F.max_pool2d(F.pad(1-x,(2,2,2,2),value=1),5,1)>0
    return [e,d&~e,~d]


def divergence(p,q):
    p=p.clamp(1e-6,1-1e-6);q=q.clamp(1e-6,1-1e-6);m=(p+q)*.5
    def kl(x,y):return x*torch.log(x/y)+(1-x)*torch.log((1-x)/(1-y))
    return .5*(kl(p,m)+kl(q,m))/np.log(2)


def validation(n,x,support):
    p=[probability(readonly(n,x.pow(g))) for g in (.95,1.05)]
    if support is None:return None
    js=divergence(*p);out=[]
    for c in range(2):
        xs=[js[0,c][r[0,c]].mean() for r in support if r[0,c].any()]
        if not xs or not all(torch.isfinite(v) for v in xs):return None
        out.append(float(torch.stack(xs).mean()))
    return out


def accepts(z0,z1,v0,v1):
    if v0 is None or v1 is None or not torch.isfinite(z1).all():return False,'unavailable'
    a0=(probability(z0)>=.5).sum((-2,-1));a1=(probability(z1)>=.5).sum((-2,-1));ratio=(a1+1)/(a0+1)
    if not ((ratio>=.5)&(ratio<=2)).all():return False,'area'
    if any(b>a+1e-6 for a,b in zip(v0,v1)) or np.mean(v0)-np.mean(v1)<=1e-6:return False,'validation'
    return True,'accepted'


def reliable(q):
    r=[q>=.9,q<=.1]
    return r if all((a.sum((-2,-1))>=16).all() for a in r) else None


def brier_history(n,item):
    p=probability(readonly(n,item['x']));error=(p-item['q']).square()
    return [float(torch.stack([error[0,c][r[0,c]].mean() for r in item['support']]).mean()) for c in range(2)]


class Host:
    def __init__(self,state,arm,identity,device='cuda:0',model=None,diagnostics=True):
        if arm not in ('G','G_HALF','G_VAL','G_MEM'):raise ValueError('unregistered arm')
        random.seed(20260907);np.random.seed(20260907);torch.manual_seed(20260907)
        self.native=Native('G',state=state,device=device,model=model);self.arm=arm;self.factor=.5 if arm=='G_HALF' else 1.
        self.lr_hook=self.native.base.register_step_pre_hook(self._scale)
        self.visits=0;self.memory=deque();self.diagnostics=diagnostics;self.last_outputs={}
        self.context={'sha256':identity};self.frozen=tensor_digest([(k,v) for k,v in self.native.model.state_dict().items() if k not in self.native.names])
    def _scale(self,opt,args,kwargs):
        for g in opt.param_groups:g['lr']=g['lr']*self.factor
    def check_frozen(self,boundary=False):
        if boundary and tensor_digest([(k,v) for k,v in self.native.model.state_dict().items() if k not in self.native.names])!=self.frozen:raise ValueError('non-BN backbone changed')
    def snapshot(self):return dict(context=self.context,visits=self.visits,native=snapshot(self.native),memory=clone(list(self.memory)))
    def restore(self,s):
        if s['context']!=self.context:raise ValueError('snapshot context mismatch')
        restore_state(self.native,s['native']);self.visits=s['visits'];self.memory=deque(clone(s['memory']))
    def step(self,x):
        n=self.native;t=self.visits+1;self.last_outputs={};pre=snapshot(n);expired=0
        while self.memory and t-self.memory[0]['time']>=64:self.memory.popleft();expired+=1
        item=self.memory[0] if self.memory and self.arm=='G_MEM' else None
        diagnostic=self.arm=='G' and self.diagnostics and (t-1)%16==0
        inspect=self.arm in ('G_VAL','G_MEM') or diagnostic
        z0=readonly(n,x);support=regions(z0) if inspect else None
        v0=validation(n,x,support) if inspect else None
        r0=brier_history(n,item) if item is not None else None
        numerical_failure=False
        try:z1,detail=n.step(x);z1=z1.detach().float().cpu();post=snapshot(n)
        except ValueError as error:
            if self.arm not in ('G_VAL','G_MEM') or 'nonfinite' not in str(error):raise
            advanced=capture();restore_state(n,pre);restore(advanced);n.rng=rng();post=snapshot(n);z1=z0;detail={'nonfinite_candidate':True};numerical_failure=True
        v1=validation(n,x,support) if inspect and not numerical_failure else None
        r1=brier_history(n,item) if item is not None and not numerical_failure else None
        accepted,reason=accepts(z0,z1,v0,v1) if self.arm in ('G_VAL','G_MEM') else (True,'native')
        veto=False
        if accepted and item is not None and (not all(np.isfinite(r1)) or any(b>a+1e-6 for a,b in zip(r0,r1))):accepted=False;reason='memory';veto=True
        self.last_outputs=dict(pre=z0,candidate=z1)
        diag=None
        if diagnostic:
            restore_state(n,pre);self.factor=.5
            try:zh,dh=n.step(x);zh=zh.detach().float().cpu();sh=snapshot(n);vh=validation(n,x,support)
            finally:self.factor=1.;restore_state(n,post)
            self.last_outputs['half']=zh
            af,rf=accepts(z0,z1,v0,v1);ah,rh=accepts(z0,zh,v0,vh)
            diag=dict(position=t,V_skip=v0,V_full=v1,V_half=vh,accept_full=af,accept_half=ah,reason_full=rf,reason_half=rh,
                state_pre=state_digest(pre),state_full=state_digest(post),state_half=state_digest(sh),augmentation='same cloned native RNG; main trajectory always full')
        if not accepted:
            restore_state(n,pre)
            # Rejection restores Adam and all persistent model state, but consumes one native augmentation draw.
            restore(post['native_rng']);n.rng=rng();restore(post['rng']);out=z0
        else:out=z1
        writes=evicted=0
        if self.arm=='G_MEM' and accepted:
            q=probability(out);rs=reliable(q)
            if rs is not None:
                self.memory.append(dict(x=x.clone(),q=q.clone(),support=rs,time=t));writes=1
                while len(self.memory)>8:self.memory.popleft();evicted+=1
        self.visits=t
        return out,dict(visit=t,state_committed=True,accepted=accepted,reason=reason,V_minus=v0,V_plus=v1,native=detail,
            memory_empty=item is None,memory_age=None if item is None else t-item['time'],history_before=r0,history_after=r1,
            memory_veto=veto,writes=writes,evictions=evicted,expired=expired,memory_size=len(self.memory),diagnostic=diag)
    def close(self):
        self.lr_hook.remove()
        for h in self.native.handles:h.remove()
