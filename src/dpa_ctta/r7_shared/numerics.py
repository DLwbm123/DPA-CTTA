"""Differentiable float64 small matrices; no adaptive jitter or fallback."""
import math
from collections import Counter
import torch
from torch import nn
from torch.nn import functional as F

COUNTS = Counter()

def finite(*xs):
    for x in xs:
        if not isinstance(x, torch.Tensor) or not torch.isfinite(x).all():
            raise ValueError('non-finite tensor')

def shape(x, expected):
    finite(x)
    if tuple(x.shape) != tuple(expected): raise ValueError('tensor shape')

def solve(a, b):
    finite(a, b)
    if a.dtype != torch.float64 or not torch.allclose(a, a.T, rtol=1e-10, atol=1e-12):
        raise ValueError('symmetric float64 SPD required')
    COUNTS['cholesky_solves'] += 1
    v = torch.cholesky_solve(b[:, None] if b.ndim == 1 else b, torch.linalg.cholesky(a))
    finite(v)
    return v[:, 0] if b.ndim == 1 else v

def symmetric(a): return (a + a.T) / 2

def stable(w, rho=.95):
    COUNTS['spectral_norms'] += 1
    return rho * w / torch.linalg.matrix_norm(w, 2).clamp_min(1)

def mlp(a, b, c): return nn.Sequential(nn.Linear(a,b), nn.GELU(), nn.Linear(b,c))

def normalize(x, dim=-1): return x / x.norm(dim=dim, keepdim=True).clamp_min(1e-6)

def attention(x, book): return (normalize(x) @ normalize(book).T / .2).softmax(-1)

def variance(raw, maximum=10.): return 1e-4 + (maximum-1e-4) * raw.sigmoid()

def temperature(raw): return .25 + 3.75 * raw.sigmoid()

def temperature_initial(): return torch.tensor(math.log(.2/.8), dtype=torch.float32)

def coherence(rows):
    gram = normalize(rows) @ normalize(rows).T
    return gram[~torch.eye(len(rows),dtype=torch.bool,device=rows.device)].square().mean()

def gaussian_nll(target, m, p):
    e = target.double()-m
    return .5*(e @ solve(p,e) + 2*torch.linalg.cholesky(p).diagonal().log().sum() + len(m)*math.log(2*math.pi))/len(m)

def seg_loss(logits, labels):
    shape(labels, logits.shape)
    if ((labels < 0) | (labels > 1)).any(): raise ValueError('binary channel labels in [0,1]')
    p=logits.sigmoid(); axes=(0,2,3)
    dice=(2*(p*labels).sum(axes)+1e-6)/(p.sum(axes)+labels.sum(axes)+1e-6)
    return F.binary_cross_entropy_with_logits(logits,labels)+.5*(1-dice).mean()

def signed_columns(q):
    idx=q.abs().argmax(0); signs=q[idx,torch.arange(q.shape[1])].sign()
    return q * torch.where(signs == 0, torch.ones_like(signs), signs)

def svd_basis(v, rank=32):
    finite(v)
    v=v.double(); u,s,_=torch.linalg.svd(v,full_matrices=False); COUNTS['svd']+=1
    if torch.linalg.matrix_rank(v) < rank: raise ValueError('SOURCE_PREP_INCOMPLETE: rank deficient')
    return signed_columns(u[:,:rank])

def predictive_basis(jcov,jprobe,curvature,rank=16):
    jcov,jprobe,curvature=[x.double() for x in (jcov,jprobe,curvature)]
    finite(jcov,jprobe,curvature)
    if jcov.ndim!=2 or jprobe.ndim!=2 or jcov.shape[1]!=jprobe.shape[1] or curvature.shape!=(len(jcov),) or (curvature<0).any():
        raise ValueError('Jacobian/curvature dimensions')
    if rank < 1 or torch.linalg.matrix_rank(jprobe) < rank: raise ValueError('insufficient probe rank')
    eye=torch.eye(jcov.shape[1],dtype=torch.float64)
    cov=symmetric(solve(.01*eye+(jcov.T*curvature)@jcov/len(jcov),eye))
    e,u=torch.linalg.eigh(cov); COUNTS['eigh']+=1
    if (e<=0).any(): raise ValueError('nonpositive covariance')
    root=(u*e.sqrt())@u.T
    _,v=torch.linalg.eigh(symmetric(root@jprobe.T@jprobe@root)); COUNTS['eigh']+=1
    basis=root@signed_columns(v[:,-rank:].flip(1)); finite(basis)
    return basis,cov

def project(basis,v):
    finite(basis,v)
    if torch.linalg.matrix_rank(basis)!=basis.shape[1]: raise ValueError('basis rank')
    z=torch.linalg.lstsq(basis.double(),v.double()).solution
    return z,dict(residual_l2=float((basis@z-v).norm()),condition=float(torch.linalg.cond(basis)))
