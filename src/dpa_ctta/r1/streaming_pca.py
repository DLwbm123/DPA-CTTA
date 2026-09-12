"""CPU float64 sufficient statistics; task-package covariance reference."""
import torch

class StreamingSubspace:
    """Cumulative CPU sufficient statistics with a lagged consumer snapshot.

    Caller must obtain the snapshot before processing the current item, and call
    merge once after that item's final prediction. No raw observations are stored.
    """
    def __init__(self, dim: int = 32, rank: int = 8, refresh: int = 16,
                 min_images: int = 16, min_vectors: int = 128):
        if not 0 < rank < dim or refresh < 1:
            raise ValueError("invalid subspace dimensions")
        self.dim, self.rank = dim, rank
        self.refresh, self.min_images, self.min_vectors = refresh, min_images, min_vectors
        self.n = self.images = self.version = self.merges = self.eigh_calls = 0
        self.last_visit = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.M2 = torch.zeros(dim, dim, dtype=torch.float64)
        self.center = self.U = self.eigenvalues = None

    @torch.no_grad()
    def merge(self, X: torch.Tensor, visit: int) -> None:
        if visit<=self.last_visit:raise ValueError('bank visit must advance')
        self.last_visit=visit
        if X.ndim != 2 or X.shape[1] != self.dim:
            raise ValueError("[n,d] input required")
        if not torch.isfinite(X).all():
            raise ValueError("nonfinite features")
        if len(X) == 0:
            return
        X = X.detach().to(device="cpu", dtype=torch.float64)
        k = len(X)
        mb = X.mean(0)
        xb = X - mb
        m2b = xb.T @ xb
        if self.n == 0:
            self.n = k
            self.mean, self.M2 = mb.clone(), m2b.clone()
        else:
            delta = mb - self.mean
            new_n = self.n + k
            self.M2 = self.M2 + m2b + torch.outer(delta, delta) * (self.n * k / new_n)
            self.mean = self.mean + delta * (k / new_n)
            self.n = new_n
        self.images += 1
        self.merges += 1
        if self.images % self.refresh or self.images < self.min_images or self.n < self.min_vectors:
            return
        cov = self.M2 / (self.n - 1)
        cov = (cov + cov.T) / 2
        self.eigh_calls += 1
        eig, vec = torch.linalg.eigh(cov)
        if not torch.isfinite(eig).all() or not torch.isfinite(vec).all():raise ValueError("nonfinite eigensystem")
        ids = torch.nonzero(eig > 1e-10).flatten().flip(0)[:self.rank]
        if not len(ids):
            self.center = self.U = self.eigenvalues = None
            self.version=visit
            return
        self.center = self.mean.clone()
        self.U = vec[:, ids].clone()
        self.eigenvalues = eig[ids].clone()
        self.version = visit

    def snapshot(self):
        if self.U is None:
            return None
        return self.center.clone(), self.U.clone(), self.version

def projection_residual(raw_features: torch.Tensor, center: torch.Tensor,
                        U: torch.Tensor) -> torch.Tensor:
    """Mean loss for one bank; caller defines per-bank equal weighting."""
    if raw_features.ndim != 2 or U.ndim != 2:
        raise ValueError("matrix features and basis required")
    if raw_features.shape[1] != U.shape[0] or center.shape != (U.shape[0],):
        raise ValueError("feature/basis shape mismatch")
    if not len(raw_features):
        raise ValueError("empty bank is omitted by caller")
    c = center.detach().to(raw_features)
    b = U.detach().to(raw_features)
    v = raw_features / raw_features.norm(dim=1, keepdim=True).clamp_min(1e-6)
    y = v - c
    residual = y - (y @ b) @ b.T
    denominator = y.square().sum(1).detach() + 1e-6
    return (residual.square().sum(1) / denominator).mean()

def bank_audit(bank):
    tensors=[v for v in vars(bank).values() if isinstance(v,torch.Tensor)]
    return dict(n=bank.n,contributing_images=bank.images,version=bank.version,ready=bank.U is not None,rank=0 if bank.U is None else bank.U.shape[1],merges=bank.merges,eigh_calls=bank.eigh_calls,state_bytes=sum(t.numel()*t.element_size() for t in tensors))
