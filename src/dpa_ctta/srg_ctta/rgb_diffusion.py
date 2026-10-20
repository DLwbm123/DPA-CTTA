"""Fixed native RGB affinity/logit diffusion, not a trained SRDF model."""
import torch,numpy as np
from torch.nn import functional as F
from scipy.ndimage import binary_fill_holes
from .topology_projection import project,BG
ARMS=('SOURCE','SMOOTH','FILL_SOURCE','TOPO_SMOOTH','EDGE_LOGIT','ROLL_MATCH','UNIFORM_MATCH','UNIFORM_LOGIT')

def neighborhoods(x):
    h,w=x.shape[-2:];p=F.pad(x,(1,1,1,1),mode='reflect')
    return torch.stack([p[...,dy:dy+h,dx:dx+w] for dy in range(3) for dx in range(3)])

def affinities(x):
    x=x.double()
    if x.shape[:2]!=(1,3) or min(x.shape[-2:])<3 or not torch.isfinite(x).all():raise ValueError('finite native single RGB required')
    d=torch.cat(((x[...,1:,:]-x[...,:-1,:]).square().mean(1).flatten(),(x[...,:,1:]-x[...,:,:-1]).square().mean(1).flatten()))
    positive=d[d>0];scale=positive.median() if positive.numel() else x.new_tensor(0.)
    distances=(neighborhoods(x)-x).square().mean(2,keepdim=True)
    axis=x.new_tensor([1.,2.,1.]);kernel=(axis[:,None]*axis[None,:]).reshape(9,1,1,1,1)
    weights=kernel*(torch.exp(-distances/scale) if float(scale)>0 else torch.ones_like(distances))
    if not torch.isfinite(weights).all() or not (weights.sum(0)>0).all():raise ValueError('invalid affinity')
    return weights,scale

def install(source,candidate):
    z=source.clone();z[...,:-1,:-1]=candidate[...,:-1,:-1];return z

@torch.no_grad()
def outputs(z,x):
    z=z.double()
    if z.shape[:2]!=(1,2) or z.shape[-2:]!=x.shape[-2:] or not torch.isfinite(z).all():raise ValueError('independent native logits required')
    weights,scale=affinities(x);neighbors=neighborhoods(z)
    uniform=weights.new_tensor([1.,2.,1.,2.,4.,2.,1.,2.,1.]).reshape(9,1,1,1,1)
    edge=install(z,(weights*neighbors).sum(0)/weights.sum(0))
    raw_uniform=install(z,(uniform*neighbors).sum(0)/16)
    wrong=torch.roll(weights,z.shape[-1]//2+1,dims=-1)
    rolled=install(z,(wrong*neighbors).sum(0)/wrong.sum(0))
    target=(edge-z).square().sum(dim=(-2,-1),keepdim=True).sqrt()
    def match(v):
        delta=v-z;norm=delta.square().sum(dim=(-2,-1),keepdim=True).sqrt()
        if ((norm==0)&(target>0)).any():raise ValueError('nonzero target with zero control direction')
        ratio=torch.where(norm>0,target/torch.where(norm>0,norm,torch.ones_like(norm)),torch.zeros_like(norm))
        return z+ratio*delta
    roll_match=match(rolled);uniform_match=match(raw_uniform)
    axis=z.new_tensor([1,2,1])/4;kernel=(axis[:,None]*axis[None,:]).expand(2,1,3,3)
    smooth=F.conv2d(F.pad(z.sigmoid(),(1,1,1,1),mode='reflect'),kernel,groups=2)
    smooth=install(z,torch.logit(smooth.clamp(1e-12,1-1e-12)))
    src=(z.cpu().numpy()[0]>=0);sm=(smooth.cpu().numpy()[0]>=0)
    fill=np.stack([binary_fill_holes(a,structure=BG) for a in src]);topo=np.stack([project(a) for a in sm])
    binary=lambda a:torch.as_tensor(a,device=z.device,dtype=torch.float64).unsqueeze(0)*2-1
    out=dict(SOURCE=z,SMOOTH=smooth,FILL_SOURCE=binary(fill),TOPO_SMOOTH=binary(topo),EDGE_LOGIT=edge,ROLL_MATCH=roll_match,UNIFORM_MATCH=uniform_match,UNIFORM_LOGIT=raw_uniform)
    for name,v in out.items():
        if not torch.isfinite(v).all():raise ValueError('nonfinite output')
    norms=[(out[a]-z).square().sum(dim=(-2,-1),keepdim=True).sqrt() for a in ('ROLL_MATCH','UNIFORM_MATCH')]
    error=max(float((n-target).abs().max()) for n in norms)
    if error>1e-9*max(1.,float(target.max())):raise ValueError('output logit displacement norm mismatch')
    diag=dict(rgb_positive_neighbor_median=float(scale),edge_delta_norm_min=float(target.min()),edge_delta_norm_max=float(target.max()),control_norm_absolute_error=error,offcenter_weight_mean=float(weights[[0,1,2,3,5,6,7,8]].mean()),roll_vs_edge_weight_rms=float((wrong-weights).square().mean().sqrt()))
    return {a:out[a] for a in ARMS},diag
