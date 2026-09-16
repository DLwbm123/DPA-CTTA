"""Current detached probabilities only; no identities, labels or future values."""
import math,random
from collections import deque
import torch

ARMS=('C','C_HALF','C_RANDOM','C_VERIFY')
PHYSICAL=dict(network_forwards=8,loss_backward_calls=1,adam_calls=1,jacobian_vjp_calls=0)


def finite(value):
    if isinstance(value,torch.Tensor):
        if not bool(torch.isfinite(value).all()):raise ValueError('nonfinite tensor')
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,(int,float)) and not math.isfinite(value):raise ValueError('nonfinite scalar')


def measurements(pre,q,trial,views):
    expected=(1,2,512,512)
    if any(p.shape!=expected or p.requires_grad for p in (pre,q,trial)) or views.shape!=(6,*expected) or views.requires_grad:raise ValueError('detached full grid required')
    values=[p.detach().to(device='cpu',dtype=torch.float64) for p in (pre,q,trial,views)]
    for p in values:
        finite(p)
        if bool(((p<0)|(p>1)).any()):raise ValueError('probability range')
    pre,q,trial,views=values
    def error(p):
        s=float((p-q).square().sum());return dict(sse=s,count=p.numel(),mean=s/p.numel())
    regions=[];delta=(trial-pre).square()
    for c,name in enumerate(('OD','OC')):
        for fg in (True,False):
            m=((q[:,c]>=.9)&(views[:,:,c]>=.5).all(0)) if fg else ((q[:,c]<=.1)&(views[:,:,c]<.5).all(0))
            n=int(m.sum());s=float(delta[:,c][m].sum())
            regions.append(dict(region=name+('_fg' if fg else '_bg'),count=n,sse=s,mean=s/n if n else None))
    valid=[r['mean'] for r in regions if r['count']]
    return dict(e_pre=error(pre),e_trial=error(trial),regions=regions,r=max(valid) if valid else None)


def quantile(history):
    if not history:return None
    s=sorted(history);h=(len(s)-1)*.90;i=math.floor(h)
    return s[i]+(h-i)*(s[min(i+1,len(s)-1)]-s[i])


class Rule:
    def __init__(self,arm,p_accept=None):
        if arm not in ARMS:raise ValueError('arm')
        if arm=='C_RANDOM' and (type(p_accept) not in (int,float) or not math.isfinite(p_accept) or not 0<=p_accept<=1):raise ValueError('frozen label-free calibration required')
        if arm!='C_RANDOM' and p_accept is not None:raise ValueError('p_accept is RANDOM only')
        self.arm=arm;self.p_accept=p_accept;self.history=deque(maxlen=128);self.random=random.Random(20260908);self.visit=0
    def decide(self,observation):
        finite(observation);self.visit+=1;past=list(self.history);r=observation['r'];q90=quantile(past)
        reason='warmup' if self.visit<=32 else 'empty_reliable_regions' if r is None else 'insufficient_history' if len(past)<32 else 'eligible'
        eligible=reason=='eligible'
        shadow=not eligible or (observation['e_trial']['mean']<=observation['e_pre']['mean']+1e-12 and r<=q90+1e-12)
        draw=self.random.random() if self.arm=='C_RANDOM' else None
        accept=True if self.arm in ('C','C_HALF') else shadow if self.arm=='C_VERIFY' else (not eligible or draw<self.p_accept)
        return dict(visit=self.visit,eligible=eligible,forced=not eligible,reason=reason,shadow_accept=bool(shadow),accept=bool(accept),past_count=len(past),q90_past=q90,random_draw=draw,p_accept=self.p_accept,**observation)
    def append(self,r):
        if r is not None:finite(r);self.history.append(r)
