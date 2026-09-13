"""T and G teachers; detached CPU geometry, full-resolution log-odds corrections."""
import time
import torch
from torch.nn import functional as F
from ..r1.region_memory import grid,tokens
from .kernels import (unit,density_pair,log_density,region_precision,grid_edges,
                      graph_weights,graph_refine,graph_energy,nested_bernoulli_projection)


def up(values, size):
    return F.interpolate(values.T.reshape(1,2,32,32),size=size,mode='bilinear',align_corners=False)


@torch.no_grad()
def correction(q, delta):
    d=up(delta.to(q),q.shape[-2:])
    refined=torch.sigmoid(torch.logit(q.clamp(1e-5,1-1e-5))+d)
    return torch.where(d==0,q,refined)


@torch.no_grad()
def density_target(q, raw_reference, snapshots, mode):
    start=time.monotonic();x=unit(raw_reference.detach().cpu()).double()
    delta=torch.zeros(1024,2,dtype=torch.float64);ready=[];evidence=[]
    for c in range(2):
        pair=snapshots[c*2:c*2+2];ready.append(all(s is not None for s in pair))
        if ready[-1]:
            covs=density_pair(pair[0][1],pair[1][1],mode)
            l0,m0=log_density(x,pair[0][0],covs[0]);l1,m1=log_density(x,pair[1][0],covs[1])
            supported=(torch.minimum(m0,m1)<=4)&(x.norm(dim=-1)>0)
            gap=(l1-l0)/32;delta[:,c]=torch.where(supported,.5*gap.clamp(-4,4),0)
            from ..host_diagnostic_analysis import distribution
            evidence.append(dict(channel=c,support_fraction=float(supported.double().mean()),
                density_difference_per_dim=distribution(gap.tolist())))
    result=correction(q,delta)
    return result,dict(pair_ready=ready,evidence=evidence,nonzero_grid_fraction=float((delta!=0).double().mean()),
        max_abs_logit_correction=float(delta.abs().max()),mean_abs_logit_correction=float(delta.abs().mean()),
        density_seconds=time.monotonic()-start)


@torch.no_grad()
def graph_target(q, raw, views, snapshots, mode):
    start=time.monotonic()
    if any(s is None for s in snapshots):return q,dict(ready=False,edges=0,iterations=0,graph_seconds=time.monotonic()-start)
    x=unit(raw.detach().cpu()).double();qg=grid(q.detach().cpu()).double()
    _,rel=tokens(qg,views);rel=rel.reshape(2,1024).T;a=1+9*rel.double()
    edges=grid_edges()
    weights=torch.zeros(len(edges),dtype=torch.float64) if mode=='ORDER' else graph_weights(x,qg,edges,
        [region_precision(s[1]) for s in snapshots] if mode=='PCA' else None)
    initial=nested_bernoulli_projection(qg,a)
    # Zero edges has an exact closed solution; avoids accumulating round-off drift.
    refined=initial if mode=='ORDER' or not bool(weights.any()) else graph_refine(qg,rel,edges,weights)
    delta=torch.logit(refined.clamp(1e-5,1-1e-5))-torch.logit(qg.clamp(1e-5,1-1e-5))
    full=correction(q,delta);wa=up(a.to(q),q.shape[-2:])
    flat=full[0].flatten(1).T
    result=nested_bernoulli_projection(flat,wa[0].flatten(1).T).T.reshape_as(q)
    return result,dict(ready=True,edges=len(edges),active_edges=int((weights>0).sum()),iterations=0 if mode=='ORDER' else 32,
        edge_weight_mean=float(weights.mean()),edge_weight_min=float(weights.min()),edge_weight_max=float(weights.max()),
        energy_before=float(graph_energy(initial,qg,a,edges,weights)),energy_after=float(graph_energy(refined,qg,a,edges,weights)),
        teacher_violations_before=int((q[:,1]>q[:,0]).sum()),teacher_violations_after=int((result[:,1]>result[:,0]).sum()),
        correction_fraction=float((q!=result).float().mean()),graph_seconds=time.monotonic()-start)
