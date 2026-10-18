"""Reliability-TTA-inspired binary routing; OD/OC remain independent Bernoulli heads."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F

ARMS=('ENT','RG','STRAT','RANDOM','ALL','GLOBAL','NOANCHOR','ENT_NORM')
NORMALIZED=('STRAT','RANDOM','ALL','GLOBAL','NOANCHOR','ENT_NORM')


def entropy(p):
    q=p.clamp(1e-6,1-1e-6)
    return -(q*q.log()+(1-q)*(1-q).log())/math.log(2)


def sobel(p):
    k=p.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])/8
    x=F.conv2d(F.pad(p,(1,)*4,mode='replicate'),k[None,None].expand(p.shape[1],1,3,3),groups=p.shape[1])
    y=F.conv2d(F.pad(p,(1,)*4,mode='replicate'),k.T[None,None].expand(p.shape[1],1,3,3),groups=p.shape[1])
    return (x.square()+y.square()+1e-12).sqrt()


@torch.no_grad()
def uncertainty(p,pw,ps):
    bd=(sobel(pw)-sobel(ps)).abs().flatten(2)
    quantile=torch.quantile(bd,bd.new_tensor([.05,.95]),dim=-1)
    bd=((bd-quantile[0,...,None])/(quantile[1]-quantile[0])[...,None].clamp_min(1e-6)).clamp(0,1).reshape_as(p)
    return (entropy(p)+(pw-ps).abs()+bd)/3


@torch.no_grad()
def strata(source):
    fg=source>=.5
    outer=F.max_pool2d(fg.float(),7,1,3)>0
    inner=F.max_pool2d((~fg).float(),7,1,3)==0
    band=outer&~inner
    return (fg&~band,~fg&~band,band)


@torch.no_grad()
def select(u,regions,arm,seed):
    if arm in ('ENT','ENT_NORM','ALL'): return torch.ones_like(u,dtype=torch.bool)
    if arm=='RG': return u<=torch.quantile(u.flatten(2),.5,dim=-1)[...,None,None]
    out=torch.zeros_like(u,dtype=torch.bool)
    if arm=='GLOBAL': regions=(torch.ones_like(u,dtype=torch.bool),)
    generator=torch.Generator(device='cpu').manual_seed(seed)
    for region in regions:
        for c in range(u.shape[1]):
            idx=region[0,c].flatten().nonzero().flatten()
            if not len(idx): continue
            keep=max(1,len(idx)//2)
            if arm=='RANDOM': order=torch.randperm(len(idx),generator=generator,device='cpu').to(idx.device)
            else: order=torch.argsort(u[0,c].flatten()[idx],stable=True)
            out[0,c].flatten()[idx[order[:keep]]]=True
    return out


def reduce_region(values,mask,regions,balanced):
    weights=torch.stack(regions if balanced else (torch.ones_like(mask),)).to(values)*mask[None]
    counts=weights.sum((-2,-1))
    valid=counts>0
    means=(values[None]*weights).sum((-2,-1))/counts.clamp_min(1)
    return (means*valid).sum()/valid.sum().clamp_min(1)


def l2(tensors):
    return float(torch.cat([v.detach().reshape(-1) for v in tensors]).double().norm())


@torch.no_grad()
def normalize_parameters(params,before,target=.01):
    delta=[(p-v).double() for p,v in zip(params,before)]
    raw=l2(delta)
    if raw<=1e-12:return raw,0.,None
    desired=[d*(target/raw) for d in delta]
    for _ in range(8):
        for p,v,d in zip(params,before,desired):p.copy_(v.double()+d)
        actual_delta=[(p-v).double() for p,v in zip(params,before)]
        actual=l2(actual_delta)
        if abs(actual-target)<=5e-7:break
        if actual<=1e-12:raise ValueError('nonzero proposal lost to parameter precision')
        # Accumulate the correction in the FP64 intended offset so a quantization
        # plateau cannot silently discard each successive radius correction.
        desired=[d*(target/actual) for d in desired]
    if abs(actual-target)>5e-7:
        # A uniform scale can jump over the target when many FP32 coordinates
        # cross a rounding threshold together. Distribute one-ULP corrections.
        flat=torch.cat([p.detach().flatten() for p in params])
        origin=torch.cat([v.flatten() for v in before])
        direction=torch.cat([d.flatten() for d in delta])
        for _ in range(4):
            toward=origin if actual>target else torch.where(direction>=0,torch.full_like(flat,float('inf')),torch.full_like(flat,-float('inf')))
            candidate=torch.nextafter(flat,toward)
            change=(candidate.double()-origin.double()).square()-(flat.double()-origin.double()).square()
            wanted=target*target-actual*actual
            valid=(change*wanted>0)&(direction!=0)&torch.isfinite(candidate)
            ids=valid.nonzero().flatten()
            if not len(ids):break
            order=torch.argsort(change[ids].abs(),stable=True);ids=ids[order]
            cumulative=change[ids].abs().cumsum(0)
            k=min(int(torch.searchsorted(cumulative,abs(wanted))),len(ids)-1)
            counts=(k,k+1)
            count=min(counts,key=lambda n:abs((float(cumulative[n-1]) if n else 0)-abs(wanted)))
            flat[ids[:count]]=candidate[ids[:count]]
            offset=0
            for p in params:
                p.copy_(flat[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
            actual_delta=[(p-v).double() for p,v in zip(params,before)];actual=l2(actual_delta)
            if abs(actual-target)<=5e-7:break
    if abs(actual-target)>5e-7:raise ValueError(f'actual affine step budget failed: raw={raw:.12g}, actual={actual:.12g}')
    dot=float(torch.cat([d.flatten() for d in delta]).dot(torch.cat([d.flatten() for d in actual_delta])))
    cosine=dot/(raw*actual)
    if cosine<.999999:raise ValueError('rounded proposal direction changed')
    return raw,actual,cosine


def objective(z,pw,ps,z0,mask,regions,arm):
    p=z.sigmoid();balanced=arm in ('STRAT','RANDOM','ALL','GLOBAL','NOANCHOR')
    target=(pw+ps).detach()/2
    ent=reduce_region(entropy(p),mask,regions,balanced)
    cons=reduce_region((p-target).square(),mask,regions,balanced)
    near=F.max_pool2d(mask.float(),3,1,1)>0
    bd=reduce_region((sobel(p)-sobel(target)).abs(),near,regions,balanced)
    p0=z0.detach().sigmoid()
    kl=p0*(F.logsigmoid(z0.detach())-F.logsigmoid(z))+(1-p0)*(F.logsigmoid(-z0.detach())-F.logsigmoid(-z))
    anchor=reduce_region(kl,~mask,regions,balanced)
    loss=ent if arm in ('ENT','ENT_NORM') else ent+cons+bd+(0 if arm=='NOANCHOR' else 1)*anchor
    return loss,dict(entropy=ent,consistency=cons,boundary=bd,anchor=anchor)


def intensity_views(pixels,seed):
    g=torch.Generator(device='cpu').manual_seed(seed)
    weak=pixels.pow(.95)
    strong=(pixels.pow(1.05)+.01*torch.randn(pixels.shape,generator=g,dtype=pixels.dtype)).clamp(0,1)
    return weak,strong


class Host:
    def __init__(self,base,arm):
        if arm not in ARMS: raise ValueError('unknown routing arm')
        self.arm=arm;self.model=copy.deepcopy(base).eval().requires_grad_(False)
        self.names=[];self.params=[]
        for name,m in self.model.named_modules():
            if isinstance(m,nn.BatchNorm2d):
                if type(m) is not nn.BatchNorm2d or not m.track_running_stats or m.running_mean is None:
                    raise ValueError('standard frozen-statistics BN required')
                for key,p in m.named_parameters(recurse=False):
                    p.requires_grad_(True);self.names.append(name+'.'+key);self.params.append(p)
        if not self.params: raise ValueError('normalization-affine subset required')
        self.initial=[p.detach().clone() for p in self.params]
        self.fixed={n:p._version for n,p in self.model.named_parameters() if not p.requires_grad}
        self.buffers={n:b.detach().clone() for n,b in self.model.named_buffers()}
        self.total_steps=0;self.reset()

    def reset(self):
        with torch.no_grad():
            for p,x in zip(self.params,self.initial): p.copy_(x);p.grad=None
        self.opt=torch.optim.Adam(self.params,lr=1e-4,betas=(.9,.999),eps=1e-8,weight_decay=0)
        self.segment_steps=0

    def forward(self,x):
        out=self.model(x)
        return out[0] if isinstance(out,(tuple,list)) else out

    def step(self,x,xw,xs,z0,seed):
        regions=strata(z0.sigmoid());logs=[];pre=None
        for step in range(2):
            self.opt.zero_grad(set_to_none=True)
            with torch.no_grad(): pw=self.forward(xw).sigmoid();ps=self.forward(xs).sigmoid()
            z=self.forward(x)
            if pre is None: pre=z.detach().clone()
            u=uncertainty(z.detach().sigmoid(),pw,ps)
            mask=select(u,regions,self.arm,seed+step*1000003)
            if int(mask.sum())<128: raise ValueError('fixed keep quota unexpectedly degenerate')
            loss,terms=objective(z,pw,ps,z0,mask,regions,self.arm)
            anchor_grads=torch.autograd.grad(terms['anchor'],self.params,retain_graph=True)
            anchor_norm=l2(anchor_grads)
            if not torch.isfinite(loss): raise ValueError('nonfinite unlabeled objective')
            loss.backward()
            if any(p.grad is None for p in self.params) or not bool(torch.stack([torch.isfinite(p.grad).all() for p in self.params]).all()): raise ValueError('invalid affine gradient')
            before=[p.detach().clone() for p in self.params];self.opt.step()
            with torch.no_grad():
                if self.arm in NORMALIZED:raw,actual,cosine=normalize_parameters(self.params,before)
                else:raw=actual=l2([p-v for p,v in zip(self.params,before)]);cosine=None
            self.total_steps+=1;self.segment_steps+=1
            count=torch.stack(regions).sum((-2,-1))
            kept=(torch.stack(regions)&mask[None]).sum((-2,-1))
            fractions=torch.where(count>0,kept/count.clamp_min(1),-1).squeeze(1).cpu().tolist()
            coverage=[[v if v>=0 else None for v in row] for row in fractions]
            logs.append(dict(loss=float(loss.detach()),**{k:float(v.detach()) for k,v in terms.items()},anchor_gradient_norm=anchor_norm,raw_step_norm=raw,actual_step_norm=actual,direction_cosine=cosine,norm_matched=bool(abs(actual-.01)<=5e-7) if self.arm in NORMALIZED else None,zero_proposal=bool(raw<=1e-12),gate_fraction=float(mask.float().mean()),stratum_gate_fraction=coverage))
        with torch.no_grad(): post=self.forward(x).detach()
        names=dict(self.model.named_parameters())
        if any(names[n]._version!=v for n,v in self.fixed.items()): raise ValueError('non-affine parameters changed')
        if not bool(torch.stack([(v==dict(self.model.named_buffers())[n]).all() for n,v in self.buffers.items()]).all()): raise ValueError('running statistics changed')
        if any(int(self.opt.state[p]['step'])!=self.segment_steps for p in self.params): raise ValueError('optimizer persistence failed')
        total_norm=l2([p-v for p,v in zip(self.params,self.initial)])
        return pre,post,mask.detach(),z.detach(),dict(steps=logs,total_affine_norm=total_norm,optimizer_age=self.segment_steps,forwards=7,optimizer_steps=2,diagnostic_anchor_gradient_calls=2,loss_backward_calls=2)
