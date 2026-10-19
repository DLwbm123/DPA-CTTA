"""Independent held-out intervention probes; no controller or label-based choices."""
import torch
from .spatial_gradient import Host as EntropyHost


def h(p):
    p=p.double().clamp(1e-12,1-1e-12)
    return -p*p.log()-(1-p)*(1-p).log()


def js(a,b):
    return (h((a+b)/2)-(h(a)+h(b))/2).clamp_min(0)


def reduce(values,source,balanced=True):
    # Equal head weight, then equal nonempty source-positive/negative regions.
    out=[]
    for ch in range(2):
        v=values[0,ch]
        masks=(source[0,ch]>=.5,source[0,ch]<.5) if balanced else (torch.ones_like(v,dtype=torch.bool),)
        out.append(torch.stack([v[m].mean() for m in masks if m.any()]).mean())
    return torch.stack(out)


def signals(before,after,anchor,source):
    if before.shape!=after.shape or before.shape!=anchor.shape or before.shape[0]!=4 or before.shape[1:3]!=(1,2):
        raise ValueError('four aligned independent Bernoulli probes required')
    if not all(bool(torch.isfinite(x).all()) for x in (before,after,anchor,source)):
        raise ValueError('nonfinite probe probability')
    result={}
    for balanced,prefix in ((True,''),(False,'full_')):
        def red(x):return reduce(x,source,balanced)
        db=red(h(before.mean(0))-h(before).mean(0))
        da=red(h(after.mean(0))-h(after).mean(0))
        ab=red(js(before,anchor).mean(0));aa=red(js(after,anchor).mean(0))
        dc=db-da;extra=(aa-ab).clamp_min(0)
        for name,value in dict(utility=dc-extra,consistency_gain=dc,negative_extra_drift=-extra,
                pre_entropy=red(h(before).mean(0)),pre_disagreement=db,pre_anchor_drift=ab,
                negative_function_change=-red((after-before).square().mean(0)).sqrt()).items():
            result[prefix+name]=value.tolist()
    result['valid_regions']=[int((source[0,c]>=.5).any())+int((source[0,c]<.5).any()) for c in range(2)]
    return result


def probe_views(x,seed):
    gen=torch.Generator(device='cpu').manual_seed(seed+700000)
    views=[]
    for _ in range(4):
        gamma=.95+.1*float(torch.rand((),generator=gen))
        flip=bool(torch.rand((),generator=gen)<.5)
        noise=torch.randn(x.shape,generator=gen,dtype=x.dtype).to(x.device)*.005
        y=(x.pow(gamma)+noise).clamp(0,1)
        views.append((y.flip(-1) if flip else y,flip))
    return views


class Host(EntropyHost):
    def __init__(self,base,kind):
        if kind!='ENT_NORM':raise ValueError('common entropy candidate only')
        super().__init__(base,kind)
        self.source_model=base

    @torch.no_grad()
    def probes(self,views,source=False):
        outputs=[]
        for x,flip in views:
            out=self.source_model(x)[0] if source else self.forward(x)
            if flip:out=out.flip(-1)
            outputs.append(out.sigmoid().double())
        return torch.stack(outputs)

    def step(self,x,xw,xs,z0,seed):
        views=probe_views(x,seed)
        before=self.probes(views);anchor=self.probes(views,True)
        out=super().step(x,xw,xs,z0,seed)
        after=self.probes(views)
        risk=signals(before,after,anchor,z0.sigmoid().double())
        out[-1]['probe_risk']=risk
        out[-1]['forwards']+=12
        return out
