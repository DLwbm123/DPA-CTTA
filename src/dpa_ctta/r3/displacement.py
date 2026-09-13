"""Real feature-to-affine VJPs and a transform of one actual Adam proposal."""
import torch
from ..r1.recovery import local_seed
from .kernels import unit,constrain_displacement,norm_matched_displacement


def flatten(params):return torch.cat([p.detach().reshape(-1) for p in params])


@torch.no_grad()
def replace(params, vector):
    if vector.ndim!=1 or vector.numel()!=sum(p.numel() for p in params) or not torch.isfinite(vector).all():
        raise ValueError('affine replacement shape/finiteness')
    offset=0
    for p in params:
        p.copy_(vector[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()


def directions(snapshot, region, random):
    count=min(2,snapshot[1].shape[1])
    if not random:return snapshot[1][:,:count]
    g=torch.Generator().manual_seed(local_seed('R3_U_RAND_'+str(region),1))
    return torch.linalg.qr(torch.randn(32,2,generator=g,dtype=torch.float64),mode='reduced').Q[:,:count]


def probe_rows(raw, selected, snapshots, params, random=False, on_vjp=None):
    v=unit(raw);probes=[];definitions=[]
    for r,(ids,s) in enumerate(zip(selected,snapshots)):
        if s is None or not len(ids):continue
        basis=directions(s,r,random).to(v)
        for j in range(basis.shape[1]):
            probes.append(v[ids.to(v.device)].mean(0)@basis[:,j]);definitions.append((ids,basis[:,j].detach()))
    rows=[]
    for i,p in enumerate(probes):
        if on_vjp is not None:on_vjp()
        gs=torch.autograd.grad(p,params,retain_graph=i<len(probes)-1,create_graph=False,allow_unused=True)
        rows.append(torch.cat([(torch.zeros_like(x) if g is None else g).detach().reshape(-1) for x,g in zip(params,gs)]))
    A=torch.stack(rows) if rows else raw.new_empty((0,sum(p.numel() for p in params)))
    return A,definitions,[float(p.detach()) for p in probes]


@torch.no_grad()
def transform(before, params, rows, mode):
    proposal=flatten(params)-before
    protected,info=constrain_displacement(proposal,rows)
    applied=norm_matched_displacement(proposal,protected) if mode=='SCALE' else protected
    # Empty/zero A preserves the actual Adam candidate exactly, including rounding.
    replaced=bool(info['rows'])
    if replaced:replace(params,before+applied)
    actual=flatten(params)-before
    norms=rows.double().norm(dim=1);A=rows.double()[norms>1e-12]/norms[norms>1e-12,None]
    den=proposal.double().norm()*actual.double().norm()
    info.update(applied_norm=float(actual.double().norm()),actual_parameter_replacements=int(replaced),
        cosine=1. if float(den)==0 else float((proposal.double()@actual.double()/den).clamp(-1,1)),
        applied_linear_probe_norm=float((A@actual.double()).norm()),parameter_dim=before.numel(),jacobian_vjp_calls=len(rows))
    return info
