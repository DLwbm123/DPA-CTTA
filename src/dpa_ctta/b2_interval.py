"""Detached six-view radii and Bernoulli interval projection; no global RNG use."""
import hashlib
import torch
import torch.nn.functional as F


@torch.no_grad()
def radius(q,s,arm,visit):
    if q.shape!=s.shape or q.ndim!=4 or len(q)!=1 or q.device.type!='cpu':raise ValueError('CPU single-image radius')
    if arm not in ('zero','I','U','S'):raise ValueError('radius arm')
    r=s.detach().clone() if arm!='zero' else torch.zeros_like(s)
    if arm in ('U','S'):
        for c in range(q.shape[1]):
            for b in (0,1):
                mask=(q[0,c]>=.5)==bool(b);v=s[0,c][mask]
                if not v.numel():continue
                if arm=='U':r[0,c][mask]=v.mean()
                elif v.numel()>1:
                    seed=int.from_bytes(hashlib.sha256(f'B2:20260907:{visit}:{c}:{b}'.encode()).digest()[:8],'big')
                    g=torch.Generator(device='cpu').manual_seed(seed)
                    r[0,c][mask]=v[torch.randperm(v.numel(),generator=g)]
    return r


@torch.no_grad()
def targets(predictions,arm,visit):
    q=predictions.mean(0);s=((predictions-q).square().mean(0)).sqrt()
    r=radius(q,s,arm,visit)
    return q,s,r,(q-r).clamp_min(0),(q+r).clamp_max(1)


def interval_kl(z,lower,upper):
    if z.shape!=lower.shape or z.shape!=upper.shape:raise ValueError('interval shape')
    if not all(torch.isfinite(t).all() for t in (z,lower,upper)):raise ValueError('nonfinite interval')
    if not ((lower>=0).all() and (upper<=1).all() and (lower<=upper).all()):raise ValueError('invalid interval')
    with torch.no_grad():
        p=z.detach().sigmoid();v=p.maximum(lower.detach()).minimum(upper.detach())
        active=(p<lower.detach())|(p>upper.detach())
        entropy=-(torch.xlogy(v,v)+torch.xlogy(1-v,1-v))
    raw=F.binary_cross_entropy_with_logits(z,v,reduction='none')-entropy
    loss=torch.where(active,raw,torch.zeros_like(raw)).mean()
    return loss,v,active,raw.detach().min()


@torch.no_grad()
def diagnostics(q,s,r,lo,hi,active):
    result=[];active=active.cpu()
    for c in range(q.shape[1]):
        fg=q[0,c]>=.5
        row=dict(channel=('OD','OC')[c],q_foreground=int(fg.sum()),pixels=fg.numel(),interval_width_mean=float((hi-lo)[0,c].mean()),crosses_half=int(((lo[0,c]<=.5)&(hi[0,c]>=.5)).sum()),partitions={})
        for key,t in [('s',s),('r',r)]:
            v=t[0,c].flatten();percentiles=torch.quantile(v,torch.tensor([.5,.9],dtype=v.dtype))
            row[key]=dict(mean=float(v.mean()),p50=float(percentiles[0]),p90=float(percentiles[1]),max=float(v.max()))
        for b in (0,1):
            m=fg==bool(b);row['partitions'][str(b)]=dict(active=int((active[0,c]&m).sum()),denominator=int(m.sum()))
        result.append(row)
    return result
