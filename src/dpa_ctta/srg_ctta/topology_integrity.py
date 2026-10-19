"""Frozen 2D adaptation of TopoSculpt probability/skeleton integrity."""
import numpy as np
import torch
from scipy.ndimage import binary_erosion,binary_dilation,generate_binary_structure
from .persistent_topology import critical_pairs,correction
CROSS=generate_binary_structure(2,1)
SQUARE=np.ones((3,3),bool)

def skeleton(mask):
    """Binary soft-skeleton limit: cross erosion, square opening, zero exterior.

Run to erosion exhaustion rather than an undocumented truncation count.
The support is detached, as it derives from a hard >.5 threshold.
"""
    image=np.asarray(mask,dtype=bool).copy()
    if image.ndim!=2:raise ValueError('2D skeleton')
    skel=np.zeros_like(image)
    for _ in range(sum(image.shape)+1):
        eroded=binary_erosion(image,structure=CROSS,border_value=0)
        opened=binary_dilation(eroded,structure=SQUARE,border_value=0)
        skel|=image & ~opened
        if not eroded.any():return skel
        image=eroded
    raise RuntimeError('erosion did not terminate')

def weighted_support(probability):
    support=np.stack([skeleton(p) for p in (probability.detach().cpu().numpy()>.5)])
    if any(not s.any() for s in support):raise ValueError('empty skeleton head; no fallback')
    return torch.as_tensor(support,device=probability.device,dtype=probability.dtype)

def integrity(current,previous,previous_support):
    """Mean of per-head natural probability MSE and weighted clDice term."""
    support=weighted_support(current)
    a=support*current;b=previous_support*previous
    sa=a.sum((-2,-1));sb=b.sum((-2,-1))
    if bool(((sa<=0)|(sb<=0)).any()):raise ValueError('zero weighted skeleton mass')
    precision=(a*previous).sum((-2,-1))/sa
    sensitivity=(b*current).sum((-2,-1))/sb
    denom=precision+sensitivity
    if bool((denom<=0).any()):raise ValueError('zero structural denominator')
    cldice=2*precision*sensitivity/denom
    mse=(current-previous).square().mean((-2,-1))
    value=(10000*mse+1000*(1-cldice)).mean()
    return value,support

def pairs_for_heads(probability,permuted=False):
    result=[]
    for p in probability.detach().cpu().numpy():
        finite,essential=critical_pairs(p)
        if permuted:
            size=p.size;shift=size//2+1
            finite=[(a+shift)%size for a in finite];essential=(essential+shift)%size
        result.append((finite,essential))
    return result

def objective(current,previous,previous_support,step,arm,cached):
    if arm not in ('TIB','PERMUTED','COR','INTEGRITY'):raise ValueError('unknown arm')
    refresh=step<=30 or step%3==0
    if arm!='INTEGRITY' and refresh:cached=pairs_for_heads(current,arm=='PERMUTED')
    cor=current.sum()*0 if arm=='INTEGRITY' else torch.stack([correction(p,q) for p,q in zip(current,cached)]).mean()
    if arm=='COR':com=current.sum()*0;support=previous_support
    else:com,support=integrity(current,previous,previous_support)
    weight=1. if step<=30 else .1
    loss=cor+weight*com
    if not bool(torch.isfinite(loss)):raise ValueError('nonfinite topology objective')
    return loss,cached,support,dict(correction=float(cor.detach()),integrity=float(com.detach()),integrity_weight=weight,ph_refresh=bool(refresh and arm!='INTEGRITY'))


def refine(model,x,arm,steps=90,progress=None):
    """One image, freshly reset caller model. No labels or persistent history."""
    model.eval().requires_grad_(True)
    params=list(model.parameters())
    opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01,betas=(.9,.999),eps=1e-8)
    with torch.no_grad():previous=model(x)[0][0].double().sigmoid().detach()
    previous_support=weighted_support(previous);cached=None;trace=[]
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        current=model(x)[0][0].double().sigmoid()
        loss,cached,support,d=objective(current,previous,previous_support,step,arm,cached)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,float('inf'),error_if_nonfinite=True)
        opt.step()
        if not all(bool(torch.isfinite(p).all()) for p in params):raise ValueError('nonfinite updated parameter')
        trace.append(dict(step=step,loss=float(loss.detach()),gradient_l2=float(norm),**d))
        previous=current.detach();previous_support=support.detach()
        if progress is not None and (step%10==0 or step==steps-1):progress(step)
    with torch.no_grad():z=model(x)[0].double()
    if not bool(torch.isfinite(z).all()):raise ValueError('nonfinite final prediction')
    opt_state={k:{a:(v.detach().cpu().clone() if torch.is_tensor(v) else v) for a,v in t.items()} for k,t in opt.state_dict()['state'].items()}
    return z,trace,opt_state
