"""R3 reference kernels. Pure CPU/tensor specifications, NOT repository hosts.

These are proposed checkpoint-only segmentation designs, not reproductions of
BayesTTA, KeepLoRA, LCA, MoIE or SPEGC. No data loading / network / CUDA code.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Sequence
import torch


def checked(x: torch.Tensor, ndim: int | None = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor) or not x.is_floating_point():
        raise TypeError("floating Tensor required")
    if ndim is not None and x.ndim != ndim:
        raise ValueError("tensor rank mismatch")
    if not torch.isfinite(x).all():
        raise ValueError("non-finite input")
    return x


def unit(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    checked(x, 2)
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


@torch.no_grad()
def shrink_covariance(cov: torch.Tensor, beta: float = .1,
                      floor: float = 1e-4) -> torch.Tensor:
    checked(cov, 2)
    if cov.shape[0] != cov.shape[1] or not 0 <= beta <= 1 or floor <= 0:
        raise ValueError("covariance / regularization")
    cov = (cov.double() + cov.double().T) / 2
    vals, vecs = torch.linalg.eigh(cov)
    scale = max(1., float(cov.norm()))
    if float(vals.min()) < -1e-8 * scale:
        raise ValueError("covariance is not PSD within round-off tolerance")
    vals = vals.clamp_min(0)
    tau = vals.mean().clamp_min(floor)
    vals = ((1-beta)*vals + beta*tau).clamp_min(floor)
    return (vecs * vals.unsqueeze(0)) @ vecs.T


@torch.no_grad()
def density_pair(cov0: torch.Tensor, cov1: torch.Tensor, mode: str,
                 rank: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """LR / DIAG / shared-isotropic prototype control, after common shrinkage."""
    c0, c1 = shrink_covariance(cov0), shrink_covariance(cov1)
    d = c0.shape[0]
    if c0.shape != c1.shape or not 0 < rank < d:
        raise ValueError("shape / rank")
    if mode == "ISO":
        tau = (c0.trace()+c1.trace())/(2*d)
        c = tau * torch.eye(d, dtype=c0.dtype, device=c0.device)
        return c, c.clone()
    if mode == "DIAG":
        return torch.diag(c0.diag()), torch.diag(c1.diag())
    if mode != "LR":
        raise ValueError("density mode")
    out = []
    for c in (c0, c1):
        vals, vecs = torch.linalg.eigh(c)
        residual = vals[:-rank].mean()
        vals = torch.cat([residual.expand(d-rank), vals[-rank:]])
        out.append((vecs * vals.unsqueeze(0)) @ vecs.T)
    return out[0], out[1]


@torch.no_grad()
def log_density(x: torch.Tensor, mean: torch.Tensor,
                cov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    checked(x, 2); checked(mean, 1); checked(cov, 2)
    d = x.shape[1]
    if mean.shape != (d,) or cov.shape != (d, d):
        raise ValueError("density dimensions")
    c = cov.to(device=x.device, dtype=torch.float64)
    l = torch.linalg.cholesky(c)
    diff = x.double()-mean.to(device=x.device, dtype=torch.float64)
    solved = torch.cholesky_solve(diff.T, l).T
    maha = (diff*solved).sum(-1).clamp_min(0)
    logdet = 2*torch.log(l.diag()).sum()
    # The shared -d/2 log(2*pi) constant cancels in the binary evidence ratio.
    return -.5*(maha+logdet), maha/d


@torch.no_grad()
def teacher_delta(x: torch.Tensor, means: Sequence[torch.Tensor],
                  covs: Sequence[torch.Tensor], *, ready: bool,
                  alpha: float = .5, per_dim_clip: float = 4.,
                  inlier_maha_per_dim: float = 4.) -> torch.Tensor:
    """Returns a LOW-RES log-odds correction, not a posterior guarantee.

    Caller upsamples this correction, NOT q, and preserves q exactly at zero
    correction. The caller rejects zero feature vectors before correction.
    """
    checked(x, 2)
    if not ready:
        return x.new_zeros(x.shape[0])
    if len(means) != 2 or len(covs) != 2:
        raise ValueError("two binary density models required")
    l0, m0 = log_density(x, means[0], covs[0])
    l1, m1 = log_density(x, means[1], covs[1])
    supported = torch.minimum(m0, m1) <= inlier_maha_per_dim
    supported &= x.norm(dim=-1) > 0
    delta = alpha*((l1-l0)/x.shape[1]).clamp(-per_dim_clip, per_dim_clip)
    return torch.where(supported, delta, 0).to(x.dtype)


@torch.no_grad()
def procrustes_step(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor,
                    tau: float = .01) -> torch.Tensor:
    """Column convention: y ~= Q x. x/y are rows. No centering/translation.

    Optimizes sum(normalized_weight * ||y-Qx||^2)+tau*||Q-I||_F^2.
    tau=0 is exposed for reference tests; production freezes tau=.01.
    """
    checked(x, 2); checked(y, 2); checked(weight, 1)
    if x.shape != y.shape or weight.shape != (len(x),) or tau < 0:
        raise ValueError("alignment dimensions")
    d = x.shape[1]
    eye = torch.eye(d, dtype=torch.float64, device=x.device)
    if len(x) == 0:
        return eye
    if (weight < 0).any() or float(weight.sum()) <= 0:
        raise ValueError("positive total weight required")
    w = weight.double()/weight.double().sum()
    cross = y.double().T @ (w[:, None]*x.double()) + tau*eye
    left, _, right_t = torch.linalg.svd(cross, full_matrices=False)
    return left @ right_t


@torch.no_grad()
def transport_state(mean: torch.Tensor, m2: torch.Tensor,
                    basis: torch.Tensor | None, Q: torch.Tensor):
    checked(mean, 1); checked(m2, 2); checked(Q, 2)
    # Caller also transports any cached snapshot mean, not only live mean/M2.
    mean2 = Q @ mean.double()
    m22 = Q @ m2.double() @ Q.T
    basis2 = None if basis is None else Q @ basis.double()
    return mean2, (m22+m22.T)/2, basis2


@torch.no_grad()
def constrain_displacement(delta: torch.Tensor, rows: torch.Tensor,
                           strength: float = 1.) -> tuple[torch.Tensor, dict]:
    """Proximal actual-step transform with normalized feature-Jacobian rows.

    argmin_d .5||d-delta||^2 + strength/2 ||A d||^2.
    This function does not estimate A, update Adam or claim global retention.
    """
    checked(delta, 1); checked(rows, 2)
    if rows.shape[1] != delta.numel() or strength < 0:
        raise ValueError("step dimensions / strength")
    rows = rows.to(device=delta.device, dtype=torch.float64)
    base = delta.double()
    norms = rows.norm(dim=1)
    A = rows[norms > 1e-12] / norms[norms > 1e-12, None]
    if not len(A) or strength == 0:
        return delta.clone(), {"rows": len(A), "base_norm": float(base.norm()),
                               "new_norm": float(base.norm()), "ratio": 1.}
    system = torch.eye(len(A), dtype=A.dtype, device=A.device) + strength*(A @ A.T)
    new = base - strength*A.T @ torch.linalg.solve(system, A @ base)
    ratio = float(new.norm()/base.norm()) if float(base.norm()) > 0 else 1.
    return new.to(delta.dtype), {
        "rows": len(A), "base_norm": float(base.norm()), "new_norm": float(new.norm()),
        "ratio": ratio, "probe_before": float((A@base).norm()),
        "probe_after": float((A@new).norm())}


@torch.no_grad()
def norm_matched_displacement(base: torch.Tensor, protected: torch.Tensor) -> torch.Tensor:
    checked(base, 1); checked(protected, 1)
    if base.shape != protected.shape:
        raise ValueError("shape")
    n = base.double().norm()
    if float(n) == 0:
        return base.clone()
    factor = (protected.double().norm()/n).clamp(0, 1)
    return base*factor.to(base.dtype)


def grid_edges(h: int = 32, w: int = 32, device="cpu") -> torch.Tensor:
    if h < 1 or w < 1:
        raise ValueError("grid size")
    ids = torch.arange(h*w, device=device).reshape(h, w)
    horizontal = torch.stack([ids[:, :-1].reshape(-1), ids[:, 1:].reshape(-1)], 1)
    vertical = torch.stack([ids[:-1, :].reshape(-1), ids[1:, :].reshape(-1)], 1)
    return torch.cat([horizontal, vertical])


@torch.no_grad()
def region_precision(cov: torch.Tensor) -> torch.Tensor:
    """LR8 covariance followed by trace-normalized precision, for graph metric."""
    c = shrink_covariance(cov)
    d = c.shape[0]; k = min(8, d-1)
    vals, vecs = torch.linalg.eigh(c)
    vals = torch.cat([vals[:-k].mean().expand(d-k), vals[-k:]])
    inverse = (vecs * vals.reciprocal().unsqueeze(0)) @ vecs.T
    return inverse/(inverse.trace()/d)


@torch.no_grad()
def graph_weights(z: torch.Tensor, q: torch.Tensor, edges: torch.Tensor,
                  precisions: Sequence[torch.Tensor] | None,
                  temperature: float = .1) -> torch.Tensor:
    checked(z, 2); checked(q, 2)
    if q.shape != (len(z), 2) or temperature <= 0:
        raise ValueError("graph shape/temperature")
    u, v = edges[:, 0], edges[:, 1]
    diff = z.double()[u]-z.double()[v]
    if precisions is None:
        dist = diff.square().sum(-1)
    else:
        if len(precisions) != 4:
            raise ValueError("four regional metrics")
        membership = torch.stack([1-q[:, 0], q[:, 0], 1-q[:, 1], q[:, 1]], 1).double()/2
        mixture = (membership[u]+membership[v])/2
        dist = torch.zeros(len(edges), dtype=torch.float64, device=z.device)
        for r, precision in enumerate(precisions):
            metric = precision.to(device=z.device, dtype=torch.float64)
            dist += mixture[:, r]*((diff @ metric)*diff).sum(-1)
    return torch.exp(-dist.clamp_min(0)/temperature).to(q.dtype)


@torch.no_grad()
def nested_bernoulli_projection(q: torch.Tensor, weight: torch.Tensor | None = None,
                                eps: float = 1e-5) -> torch.Tensor:
    """Forward-KL projection onto q_OC <= q_OD, independent Bernoulli channels.

    Solves sum_c weight_c KL(Ber(p_c)||Ber(q_c)). Not a softmax operation.
    """
    checked(q, 2)
    if q.shape[1] != 2 or not 0 < eps < .5:
        raise ValueError("OD/OC shape/epsilon")
    if weight is None:
        weight = torch.ones_like(q)
    checked(weight, 2)
    if weight.shape != q.shape or (weight <= 0).any():
        raise ValueError("positive matching weights required")
    safe = q.clamp(eps, 1-eps)
    logits = torch.logit(safe)
    pooled = torch.sigmoid((weight*logits).sum(-1)/weight.sum(-1))
    violation = safe[:, 1] > safe[:, 0]
    return torch.where(violation[:, None], pooled[:, None].expand_as(safe), safe)


def graph_energy(Q, q, weights_unary, edges, weights_edges, lam=.25, eps=1e-5):
    Q=Q.clamp(eps, 1-eps); q=q.clamp(eps, 1-eps)
    kl=Q*torch.log(Q/q)+(1-Q)*torch.log((1-Q)/(1-q))
    dist=(Q[edges[:, 0]]-Q[edges[:, 1]]).square().sum(-1)
    return (weights_unary*kl).sum()+lam*(weights_edges*dist).sum()


@torch.no_grad()
def graph_refine(q: torch.Tensor, reliable: torch.Tensor, edges: torch.Tensor,
                 weights: torch.Tensor, *, steps: int = 32, eta: float = .25,
                 lam: float = .25, eps: float = 1e-5) -> torch.Tensor:
    """Fixed-count mirror/proximal iterations. No learned graph, no backward.

    Minimize sum a KL(Q||q)+lam sum_(undirected edges) w||Q_u-Q_v||^2
    subject to 0<=OC<=OD<=1. Fixed 32 steps is NOT a convergence claim.
    """
    checked(q, 2); checked(weights, 1)
    if reliable.shape != q.shape or reliable.dtype != torch.bool:
        raise ValueError("reliability shape/type")
    if weights.shape != (len(edges),) or (weights < 0).any() or (weights > 1+1e-6).any():
        raise ValueError("bounded edge weights")
    if steps < 0 or eta <= 0 or lam < 0:
        raise ValueError("solver parameter")
    q=q.clamp(eps, 1-eps)
    a=1+9*reliable.to(q.dtype)
    Q=nested_bernoulli_projection(q, a, eps)
    u, v=edges[:, 0], edges[:, 1]
    for _ in range(steps):
        messages=2*lam*weights[:, None]*(Q[u]-Q[v])
        g=torch.zeros_like(Q)
        g.index_add_(0, u, messages);g.index_add_(0, v, -messages)
        unary=1+eta*a
        candidate=torch.sigmoid((torch.logit(Q.clamp(eps,1-eps))+eta*a*torch.logit(q)-eta*g)/unary)
        Q=nested_bernoulli_projection(candidate, unary, eps)
    return Q


@dataclass
class RouteEntry:
    descriptor_sum: torch.Tensor
    count: int = 0
    distance_mean: float = 0.
    distance_m2: float = 0.
    @property
    def centre(self):
        return self.descriptor_sum/self.descriptor_sum.norm().clamp_min(1e-12)


class DescriptorRouter:
    """Bounded online context cache. No real domain ID, no replay, no evictions.

    choose is read-only; newly created slots are pending until commit after output.
    The adaptation host separately stores BN/Adam and regional memories per slot.
    """
    def __init__(self, capacity: int = 3, min_observations: int = 16):
        if capacity < 1 or min_observations < 2:
            raise ValueError("router configuration")
        self.capacity=capacity;self.min_observations=min_observations
        self.entries: list[RouteEntry]=[]
    @torch.no_grad()
    def choose(self, descriptor: torch.Tensor) -> tuple[int, bool, float]:
        checked(descriptor, 1)
        if not self.entries:
            return 0, True, 0.
        d=descriptor.double()/descriptor.double().norm().clamp_min(1e-12)
        distances=torch.stack([(d-e.centre).square().sum() for e in self.entries])
        j=int(distances.argmin());distance=float(distances[j]);e=self.entries[j]
        std=(e.distance_m2/max(e.count-1,1))**.5
        threshold=max(1e-6,e.distance_mean+3*std)
        new=(len(self.entries)<self.capacity and e.count>=self.min_observations and distance>threshold)
        return (len(self.entries), True, distance) if new else (j, False, distance)
    @torch.no_grad()
    def commit(self, descriptor: torch.Tensor, index: int, created: bool, distance: float):
        checked(descriptor, 1)
        d=descriptor.double()/descriptor.double().norm().clamp_min(1e-12)
        if created:
            if index != len(self.entries) or index>=self.capacity:
                raise ValueError("creation order/capacity")
            self.entries.append(RouteEntry(d.clone(), 1, 0., 0.))
            return
        if index not in range(len(self.entries)) or not torch.isfinite(torch.tensor(distance)):
            raise ValueError("route commit")
        e=self.entries[index];e.count+=1;e.descriptor_sum+=d
        delta=distance-e.distance_mean;e.distance_mean+=delta/e.count
        e.distance_m2+=delta*(distance-e.distance_mean)


def recurring_stream(domain_items: Mapping[str, Sequence[str]], base_order: Sequence[str],
                     chunk: int = 64) -> list[str]:
    """Metadata-only, unique-content recurrence; never repeats a content ID."""
    if set(domain_items) != set(base_order) or len(base_order)!=len(set(base_order)) or chunk<=0:
        raise ValueError("domain mapping/order")
    all_items=[x for k in base_order for x in domain_items[k]]
    if len(all_items)!=len(set(all_items)):
        raise ValueError("duplicate content identities")
    positions={k:0 for k in base_order};result=[];r=0
    while any(positions[k]<len(domain_items[k]) for k in base_order):
        shift=r%len(base_order)
        order=list(base_order[shift:])+list(base_order[:shift])
        for k in order:
            lo=positions[k];hi=min(lo+chunk,len(domain_items[k]))
            result.extend(domain_items[k][lo:hi]);positions[k]=hi
        r+=1
    return result
