"""Tensor-only source preparation. No filesystem loaders or target substitution."""
import functools
import hashlib
import math
import time
from dataclasses import dataclass
import torch
from .numerics import COUNTS, finite, seg_loss, predictive_basis, svd_basis, project

RANGES=((-0.1,.1),(.7,1.3),(.7,1.5),(.8,1.2),(.8,1.2),(.8,1.2),(0,1.5),(0,.03),(0,.25))
IDENTITY=(0.,1.,1.,1.,1.,1.,0.,0.,0.)

def seed(key): return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8],'big')%(2**63)
def generator(key): return torch.Generator().manual_seed(seed(key))

def split(groups,existing=None):
    if len(groups)!=len(set(groups)): raise ValueError('duplicate groups')
    if existing is None:
        ordered=sorted(groups,key=lambda x:hashlib.sha256(('R7_SOURCE_SPLIT_V1|'+x).encode()).digest())
        a,b=math.floor(.70*len(groups)),math.floor(.15*len(groups))
        existing=dict(fit=ordered[:a],cal=ordered[a:a+b],val=ordered[a+b:])
    if set(existing)!={'fit','cal','val'}: raise ValueError('source folds')
    flat=[g for values in existing.values() for g in values]
    if len(flat)!=len(set(flat)) or set(flat)!=set(groups): raise ValueError('fold overlap/coverage')
    if any(len(existing[f])<n for f,n in [('fit',32),('cal',4),('val',4)]): raise ValueError('source fold minimum')
    return {k:list(v) for k,v in existing.items()}

def anchors(fold):
    if fold not in ('fit','cal','val'): raise ValueError('source fold')
    values=[]
    for i in range(128 if fold=='fit' else 32):
        g=generator(f'R7_STYLE_V1|20260918|{fold}|{i}');x=torch.tensor(IDENTITY)
        if i:
            active=3 if fold=='val' else 1+(i-1)%2
            for j in torch.randperm(9,generator=g)[:active].tolist():
                lo,hi=RANGES[j]; x[j]=lo+(hi-lo)*torch.rand((),generator=g)
        values.append(x)
    return torch.stack(values)

def simulate(x,style,key):
    finite(x,style)
    if tuple(style.shape)!=(9,) or any(not lo<=float(v)<=hi for v,(lo,hi) in zip(style,RANGES)): raise ValueError('style range')
    if x.shape[0:2]!=(1,3) or x.dtype!=torch.float32 or x.device.type!='cpu': raise ValueError('CPU source RGB')
    b,c,g,r,gg,bb,sigma,noise,vignette=style.tolist()
    y=((x-.5)*c+.5+b).clamp(0,1).pow(g)*x.new_tensor([r,gg,bb])[None,:,None,None]
    if sigma>0:
        radius=math.ceil(3*sigma);k=torch.arange(-radius,radius+1,dtype=x.dtype)
        k=(-k.square()/(2*sigma*sigma)).exp(); k=k/k.sum()
        for horizontal in (True,False):
            weight=(k.view(1,1,1,-1) if horizontal else k.view(1,1,-1,1)).expand(3,-1,-1,-1)
            pad=(radius,radius,0,0) if horizontal else (0,0,radius,radius)
            y=torch.nn.functional.conv2d(torch.nn.functional.pad(y,pad,mode='reflect'),weight,groups=3)
    if noise:y=y+noise*torch.randn(y.shape,generator=generator(key),dtype=y.dtype)
    u=torch.linspace(-1,1,x.shape[-1]);v=torch.linspace(-1,1,x.shape[-2])
    return (y*(1-vignette*(v[:,None].square()+u[None,:].square())/2)).clamp(0,1)

@dataclass(frozen=True)
class Record:
    group: str
    fold: str
    image: torch.Tensor
    label: torch.Tensor
    origin: str = 'source'

class SourceData:
    def __init__(self,records,folds):
        self.folds=split([r.group for r in records],folds); self.records={r.group:r for r in records}
        for fold,groups in self.folds.items():
            for group in groups:
                r=self.records[group]
                if r.origin!='source' or r.fold!=fold: raise ValueError('source provenance/fold')
    def get(self,group,fold):
        if group not in self.folds[fold]: raise ValueError('source fold leakage')
        return self.records[group]

@dataclass(frozen=True)
class Oracles:
    fold: str
    values: torch.Tensor
    support_pairs: tuple
    def validate(self,data):
        n=128 if self.fold=='fit' else 32
        if self.fold not in data.folds or self.values.shape!=(1024,n) or len(self.support_pairs)!=n: raise ValueError('oracle dimensions/fold')
        finite(self.values)
        for pair in self.support_pairs:
            if len(pair)!=2 or len(set(pair))!=2 or not set(pair)<=set(data.folds[self.fold]): raise ValueError('oracle support identity')
        return self

def support_pair(data,fold,index):
    groups=data.folds[fold];g=generator(f'R7_ORACLE_PAIR|{fold}|{index}')
    return tuple(groups[i] for i in torch.randperm(len(groups),generator=g)[:2].tolist())

def oracle_one(segmenter,data,fold,index):
    pair=support_pair(data,fold,index);style=anchors(fold)[index]
    records=[data.get(group,fold) for group in pair]
    images=[simulate(r.image,style,f'R7_ORACLE_NOISE|{fold}|{index}|{r.group}') for r in records]
    v=torch.zeros(1024,requires_grad=True);opt=torch.optim.Adam([v],lr=.03,betas=(.9,.999),eps=1e-8,weight_decay=0)
    for _ in range(16):
        opt.zero_grad(set_to_none=True)
        loss=sum(seg_loss(segmenter(x,v),r.label) for x,r in zip(images,records))/2+1e-3*v.square().mean()
        finite(loss);loss.backward();COUNTS['source_backward_calls']+=1;finite(v.grad)
        opt.step();COUNTS['source_Adam']+=1;finite(v)
    return v.detach(),pair

def oracle_all(segmenter,data,fold):
    results=[oracle_one(segmenter,data,fold,i) for i in range(len(anchors(fold)))]
    return Oracles(fold,torch.stack([r[0] for r in results],1),tuple(r[1] for r in results)).validate(data)

def shared_basis(oracles,data):
    oracles.validate(data)
    if oracles.fold!='fit':raise ValueError('only fit oracle constructs basis')
    return svd_basis(oracles.values,32)

def pooled_jacobian(segmenter,data,groups):
    rows=[];readouts=[]
    for group in groups:
        r=data.get(group,'fit');v=torch.zeros(1024,requires_grad=True)
        z=segmenter(r.image,v);readout=torch.nn.functional.adaptive_avg_pool2d(z,(4,4)).flatten()
        readouts.append(readout.detach().double())
        for i in range(32):
            rows.append(torch.autograd.grad(readout[i],v,retain_graph=i<31)[0].detach().double());COUNTS['source_VJP']+=1
    return torch.stack(rows),torch.cat(readouts)

def a_basis(segmenter,data):
    groups=sorted(data.folds['fit'],key=lambda g:seed('R7_BASIS|'+g))
    if len(groups)<32: raise ValueError('32 disjoint fit groups required')
    jc,r=pooled_jacobian(segmenter,data,groups[:16]);jp,_=pooled_jacobian(segmenter,data,groups[16:32])
    b,cov=predictive_basis(jc,jp,r.sigmoid()*(1-r.sigmoid()))
    return b,dict(cov_groups=groups[:16],probe_groups=groups[16:32],covariance=cov)

def sequence(fold,step):
    bank=anchors(fold);gen=generator(f'R7_SCHEDULE|20260918|{fold}|{step}')
    a,b=torch.randperm(len(bank),generator=gen)[:2].tolist()
    if step%3==0:return [a,a,b,b]
    if step%3==2:return [a,b,a,b]
    scale=torch.tensor([hi-lo for lo,hi in RANGES]);chosen=[a]
    for _ in range(3):
        distances=(((bank-bank[chosen[-1]])/scale)**2).sum(1)
        chosen.append(min((j for j in range(len(bank)) if j not in chosen),key=lambda j:(float(distances[j]),j)))
    return chosen

def roles(data,fold,step,anchor_ids,oracles):
    # Exclude each proxy's support subjects from its query as well as its current
    # support. A small deterministic matching selects eight unique roles if possible.
    pool=data.folds[fold];g=generator(f'R7_ROLES|{fold}|{step}')
    ordered=[pool[i] for i in torch.randperm(len(pool),generator=g).tolist()]
    candidates=[ordered if slot%2==0 else [x for x in ordered if x not in oracles.support_pairs[anchor_ids[slot//2]]] for slot in range(8)]
    assigned={}
    def assign(slot,seen):
        for group in candidates[slot]:
            if group in seen: continue
            seen.add(group)
            if group not in assigned or assign(assigned[group],seen):assigned[group]=slot;return True
        return False
    if len(pool)>=8 and all(assign(i,set()) for i in range(8)):
        result=[None]*8
        for group,slot in assigned.items():result[slot]=group
        return result,False
    result=[]
    for t in range(4):
        support=next(x for x in ordered if t==0 or x!=result[-2])
        query=next(x for x in ordered if x!=support and x not in oracles.support_pairs[anchor_ids[t]])
        result.extend((support,query))
    return result,True

def terminal_step(fn):
    @functools.wraps(fn)
    def call(self,*args,**kwargs):
        if self.mode=='FAILED':raise RuntimeError('source trainer stopped at first failure')
        before=COUNTS.copy();started=time.monotonic()
        try:return fn(self,*args,**kwargs)
        except Exception as exc:
            self.mode='FAILED';self.first_error=dict(type=type(exc).__name__,message=str(exc),counts=dict(COUNTS-before),wall_seconds=time.monotonic()-started)
            raise
    return call

class SourceTrainer:
    def __init__(self,segmenter,method,data,oracles):
        self.segmenter=segmenter;self.method=method;self.data=data;self.oracles=oracles.validate(data)
        if oracles.fold!='fit':raise ValueError('fit trainer requires fit oracle')
        self.fit_steps=0;self.cal_steps=0;self.mode='FIT';self.gradients={};self.first_error=None
        method.requires_grad_(True);method.set_stage('fit')
        self.opt=torch.optim.AdamW([p for p in method.parameters() if p.requires_grad],lr=3e-4,weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    def episode(self,fold,step,oracles):
        oracles.validate(self.data)
        if oracles.fold!=fold:raise ValueError('oracle cross fold')
        ids=sequence(fold,step);rs,reuse=roles(self.data,fold,step,ids,oracles)
        state=self.method.initial();losses=[];queries=[]
        for t,aid in enumerate(ids):
            s,q=[self.data.get(g,fold) for g in rs[2*t:2*t+2]];style=anchors(fold)[aid]
            with torch.no_grad():
                _,raw0,e0=self.segmenter(s.image,observe=True)
                _,raw,e=self.segmenter(simulate(s.image,style,f'R7_SUPPORT|{fold}|{step}|{t}'),observe=True)
            # The clean target descriptor participates only as a detached target.
            clean=self.method.observe(raw0,e0)
            zstar=self.method.project(oracles.values[:,aid])
            state,aux=self.method.update(raw,e,state)
            z=self.segmenter(simulate(q.image,style,f'R7_QUERY|{fold}|{step}|{t}'),self.method.ambient(state))
            losses.append(seg_loss(z,q.label)+self.method.fit_loss(aux,clean,zstar,state))
            prob=z.detach().sigmoid();dice=(2*(prob*q.label).sum((0,2,3))+1e-6)/(prob.sum((0,2,3))+q.label.sum((0,2,3))+1e-6)
            queries.append(dict(seg_loss=float(seg_loss(z.detach(),q.label)),soft_Dice_OD_OC=dice.tolist(),proxy_mse=float((self.method.code(state).detach()-zstar).square().mean())))
        return torch.stack(losses).mean(),dict(query=queries,group_reuse=reuse,state=state)
    @terminal_step
    def fit_step(self):
        if self.mode!='FIT' or self.fit_steps>=1000:raise ValueError('fixed fit lifecycle')
        self.opt.zero_grad(set_to_none=True);loss,audit=self.episode('fit',self.fit_steps,self.oracles)
        finite(loss);loss.backward();COUNTS['source_backward_calls']+=1
        for n,p in self.method.named_parameters():
            if p.grad is not None:finite(p.grad);self.gradients[n]=float(p.grad.norm())
        torch.nn.utils.clip_grad_norm_([p for p in self.method.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
        self.opt.step();COUNTS['source_AdamW']+=1;finite(*self.method.parameters());self.fit_steps+=1
        return float(loss),audit
    def fit(self):
        while self.fit_steps<1000:self.fit_step()
    def start_calibration(self,oracles,*,procedural_micro=False):
        oracles.validate(self.data)
        if oracles.fold!='cal' or self.mode!='FIT' or (self.fit_steps!=1000 and not procedural_micro):raise ValueError('calibration lifecycle/fold')
        self.mode='CAL';self.cal_oracles=oracles;self.method.requires_grad_(False);self.method.set_stage('cal')
        for p in self.method.parameters():p.grad=None
        self.cal_opt=torch.optim.Adam([p for p in self.method.parameters() if p.requires_grad],lr=1e-3,betas=(.9,.999),eps=1e-8,weight_decay=0)
    @terminal_step
    def cal_step(self):
        if self.mode!='CAL' or self.cal_steps>=256:raise ValueError('fixed calibration lifecycle')
        step=self.cal_steps;ids=sequence('cal',step);rs,_=roles(self.data,'cal',step,ids,self.cal_oracles)
        state=self.method.initial();losses=[];self.cal_opt.zero_grad(set_to_none=True)
        for t,aid in enumerate(ids):
            r=self.data.get(rs[2*t],'cal');image=simulate(r.image,anchors('cal')[aid],f'R7_CAL|{step}|{t}')
            if self.method.group=='C':
                image=(image+.04*(step+1)/256*torch.randn(image.shape,generator=generator(f'R7_CAL_RGB|{step}|{t}'))).clamp(0,1)
            with torch.no_grad():_,raw,e=self.segmenter(image,observe=True)
            if self.method.group=='C':e=e+.02*(step+1)/256*torch.randn(e.shape,generator=generator(f'R7_CAL_TOKEN|{step}|{t}'))
            state,aux=self.method.update(raw,e,state)
            losses.append(self.method.cal_loss(aux,self.method.project(self.cal_oracles.values[:,aid]),state))
        loss=torch.stack(losses).mean();finite(loss);loss.backward();COUNTS['calibration_backward_calls']+=1
        for p in self.method.parameters():
            if p.requires_grad and p.grad is not None:finite(p.grad)
        self.cal_opt.step();COUNTS['calibration_Adam']+=1;finite(*self.method.parameters());self.cal_steps+=1
        return float(loss)
    def calibrate(self):
        while self.cal_steps<256:self.cal_step()
        self.method.requires_grad_(False);self.method.set_stage('online');self.mode='FROZEN'
        if self.method.group=='C':
            if getattr(self,'constant_phase',None):self.constant_phase()
            self.constant_variance()
    @torch.no_grad()
    def constant_variance(self):
        if self.method.group!='C' or self.mode!='FROZEN':raise ValueError('constant R source cal only after calibration')
        logs=[]
        for aid,style in enumerate(anchors('cal')):
            for group in self.data.folds['cal']:
                r=self.data.get(group,'cal');image=simulate(r.image,style,f'R7_CONST_R|{aid}|{group}')
                _,raw,e=self.segmenter(image,observe=True);logs.append(self.method.observe(raw,e)['R'].log().mean(0))
        self.method.set_constant_variance(torch.stack(logs).mean(0).exp(),'cal')
    @torch.no_grad()
    def validate(self,oracles):
        if self.mode!='FROZEN' or oracles.fold!='val':raise ValueError('held-out validation only')
        return [self.episode('val',i,oracles)[1]['query'] for i in range(64)]
