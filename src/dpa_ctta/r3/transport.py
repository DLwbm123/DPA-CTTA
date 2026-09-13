"""Fit a current pre/post alignment; move all history before inserting post tokens."""
import time
import torch
from ..r1.recovery import local_seed
from .kernels import unit,procrustes_step


@torch.no_grad()
def commit(memory, pre, post, selected, visit, mode):
    started=time.monotonic();x=unit(pre.detach().cpu()).double();y=unit(post.detach().cpu()).double()
    active=[(r,ids) for r,ids in enumerate(selected) if len(ids)]
    unique=len(torch.cat([i for _,i in active]).unique()) if active else 0
    Q=torch.eye(32,dtype=torch.float64);xs=[];ys=[];weights=[];shuffled=0
    for r,ids in active:
        target=ids
        if mode=='SHUFFLE' and len(ids)>1:
            g=torch.Generator().manual_seed(local_seed('R3_M_SHUFFLE_'+str(r),visit))
            target=ids[torch.randperm(len(ids),generator=g)];shuffled+=int((target!=ids).sum())
        xs.append(x[ids]);ys.append(y[target]);weights.append(torch.full((len(ids),),1/(len(active)*len(ids)),dtype=torch.float64))
    svd_seconds=0.
    if mode!='IDPOST' and unique>=32:
        t=time.monotonic();Q=procrustes_step(torch.cat(xs),torch.cat(ys),torch.cat(weights));svd_seconds=time.monotonic()-t
    before=after=0.
    if active:
        X,Y,w=torch.cat(xs),torch.cat(ys),torch.cat(weights)
        before=float((w*(Y-X).square().sum(1)).sum());after=float((w*(Y-X@Q.T).square().sum(1)).sum())
    memory.transport(Q,visit)
    memory.merge([y[ids] for ids in selected],visit)
    return dict(distinct_positions=unique,paired_tokens=sum(len(i) for i in selected),shuffled_positions=shuffled,
        insufficient_support=unique<32,identity_policy=mode=='IDPOST',orthogonal_error=float((Q.T@Q-torch.eye(32)).norm()),
        rotation_from_identity=float((Q-torch.eye(32)).norm()),fit_before=before,fit_after=after,
        svd_seconds=svd_seconds,transport_seconds=time.monotonic()-started,frame_version=memory.frame_version)
