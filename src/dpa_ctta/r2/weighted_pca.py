"""Exponentially weighted CPU sufficient statistics; no raw observation cache."""
import math
import torch
from ..r1.streaming_pca import StreamingSubspace, bank_audit

RHO = 2**(-1/128)


class WeightedSubspace(StreamingSubspace):
    def __init__(self, rho=RHO):
        super().__init__()
        if not 0 < rho <= 1: raise ValueError('invalid decay')
        self.rho = rho
        self.W = self.Q = 0.
        self.projector_change = None

    def covariance(self):
        divisor = self.W-self.Q/self.W if self.W else 0.
        if not math.isfinite(divisor) or divisor <= 0: raise ValueError('invalid weighted covariance divisor')
        cov = self.M2/divisor
        if not torch.isfinite(cov).all(): raise ValueError('nonfinite weighted covariance')
        return (cov+cov.T)/2

    @torch.no_grad()
    def merge(self, X, visit):
        if type(visit) is not int or visit != self.last_visit+1: raise ValueError('one merge per global visit required')
        if X.ndim != 2 or X.shape[1] != self.dim or not torch.isfinite(X).all(): raise ValueError('finite [n,32] vectors required')
        X = X.detach().to(device='cpu', dtype=torch.float64)
        self.last_visit = visit
        self.W *= self.rho; self.Q *= self.rho**2; self.M2 *= self.rho
        k = len(X)
        if k:
            mb = X.mean(0); centered = X-mb; m2b = centered.T@centered
            if self.W == 0: self.mean, self.M2 = mb.clone(), m2b.clone()
            else:
                delta = mb-self.mean; total = self.W+k
                self.M2 = self.M2+m2b+torch.outer(delta,delta)*(self.W*k/total)
                self.mean = self.mean+delta*(k/total)
            self.W += k; self.Q += k; self.n += k; self.images += 1; self.merges += 1
        if not math.isfinite(self.W+self.Q) or not torch.isfinite(self.mean).all() or not torch.isfinite(self.M2).all():
            raise ValueError('nonfinite weighted statistics')
        self.projector_change = None
        if not k or self.images < self.min_images or self.n < self.min_vectors or self.images % self.refresh: return
        self.eigh_calls += 1
        eig, vec = torch.linalg.eigh(self.covariance())
        if not torch.isfinite(eig).all() or not torch.isfinite(vec).all(): raise ValueError('nonfinite eigensystem')
        ids = torch.nonzero(eig > 1e-10).flatten().flip(0)[:self.rank]
        previous = self.U
        self.U = vec[:,ids].clone() if len(ids) else None
        if previous is not None and self.U is not None:
            self.projector_change = float((self.U@self.U.T-previous@previous.T).norm())
        self.center = self.mean.clone() if len(ids) else None
        self.eigenvalues = eig[ids].clone() if len(ids) else None
        self.version = visit


def audit(bank):
    weighted = isinstance(bank,WeightedSubspace)
    return dict(bank_audit(bank), raw_n=bank.n, images=bank.images,
                W=bank.W if weighted else float(bank.n), Q=bank.Q if weighted else float(bank.n),
                rho=bank.rho if weighted else 1.,
                projector_change=bank.projector_change if weighted else None)
