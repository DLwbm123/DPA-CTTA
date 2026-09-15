"""R4T graph-teacher mathematics, CPU only; no model, files, or target labels.

New project proposal, not a reproduction of SPEGC or classic random-walker
segmentation. The caller must retain raw current RGB and original six-view
probabilities, and must not return evaluator results to this module.
"""
from __future__ import annotations
import hashlib
import math
from typing import Dict, Tuple
import torch
from torch.nn import functional as F

MODES = ('G_BOUND', 'G_CONST', 'G_SHUFFLE', 'G_GLOBAL')


def _float_cpu(x: torch.Tensor, name: str) -> None:
    if not isinstance(x, torch.Tensor) or x.device.type != 'cpu' or not x.is_floating_point():
        raise ValueError(f'{name}: CPU floating tensor required')
    if not bool(torch.isfinite(x).all()):
        raise ValueError(f'{name}: nonfinite')


def reliability(q: torch.Tensor, views: torch.Tensor) -> torch.Tensor:
    """q [C,H,W], six probabilities [6,C,H,W]. Exact same 0.1/0.9 rule."""
    if views.shape != (6, *q.shape):
        raise ValueError('six aligned probability views required')
    hard = q >= .5
    return ((q <= .1) | (q >= .9)) & ((views >= .5) == hard[None]).all(0)


def contour_band(hard: torch.Tensor, radius: int = 2) -> torch.Tensor:
    """Both endpoints of real four-neighbour label changes; no artificial border."""
    if hard.dtype != torch.bool or hard.ndim != 3 or radius < 0:
        raise ValueError('CHW bool and nonnegative radius required')
    boundary = torch.zeros_like(hard)
    dh = hard[:, :, 1:] != hard[:, :, :-1]
    dv = hard[:, 1:, :] != hard[:, :-1, :]
    boundary[:, :, 1:] |= dh
    boundary[:, :, :-1] |= dh
    boundary[:, 1:, :] |= dv
    boundary[:, :-1, :] |= dv
    return F.max_pool2d(boundary[None].double(), 2 * radius + 1,
                        stride=1, padding=radius)[0].bool()


def _seed(visit: int, salt: str, seed: int) -> int:
    if type(visit) is not int or visit < 1:
        raise ValueError('visit must be positive integer')
    text = f'R4T_G|{seed}|{visit}|{salt}'.encode('utf-8')
    return int.from_bytes(hashlib.sha256(text).digest()[:8], 'big') % (2 ** 63 - 1)


@torch.no_grad()
def edge_weights(rgb_grid: torch.Tensor, mode: str, visit: int,
                 seed: int = 20260907, sigma: float = .5,
                 std_floor: float = .05, weight_floor: float = 1e-4
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return horizontal Hx(W-1) and vertical (H-1)xW conductances.

    Current-image channel standardisation is only used by the graph; it never
    replaces the segmentation model preprocessing. CONST matches orientation-
    specific weight sums. SHUFFLE preserves each orientation's weight multiset.
    """
    _float_cpu(rgb_grid, 'rgb_grid')
    if rgb_grid.ndim != 3 or rgb_grid.shape[0] != 3 or min(rgb_grid.shape[1:]) < 2:
        raise ValueError('3xHxW RGB grid required')
    if mode not in MODES or sigma <= 0 or std_floor <= 0 or not 0 <= weight_floor <= 1:
        raise ValueError('invalid graph settings')
    r = rgb_grid.detach().double()
    mean = r.mean((1, 2), keepdim=True)
    sd = ((r - mean).square().mean((1, 2), keepdim=True)).sqrt().clamp_min(std_floor)
    a = (r - mean) / sd
    dh = (a[:, :, 1:] - a[:, :, :-1]).square().sum(0)
    dv = (a[:, 1:, :] - a[:, :-1, :]).square().sum(0)
    wh = torch.exp(-dh / (2 * sigma ** 2)).clamp_min(weight_floor)
    wv = torch.exp(-dv / (2 * sigma ** 2)).clamp_min(weight_floor)
    if mode == 'G_CONST':
        wh = torch.full_like(wh, float(wh.mean()))
        wv = torch.full_like(wv, float(wv.mean()))
    elif mode == 'G_SHUFFLE':
        values = []
        for salt, w in [('h', wh), ('v', wv)]:
            gen = torch.Generator(device='cpu').manual_seed(_seed(visit, salt, seed))
            ix = torch.randperm(w.numel(), generator=gen)
            values.append(w.flatten()[ix].reshape(w.shape))
        wh, wv = values
    return wh, wv


def neighbours(x: torch.Tensor, wh: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(x)
    result[:, :, :-1] += wh[None] * x[:, :, 1:]
    result[:, :, 1:] += wh[None] * x[:, :, :-1]
    result[:, :-1, :] += wv[None] * x[:, 1:, :]
    result[:, 1:, :] += wv[None] * x[:, :-1, :]
    return result


def degree(wh: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
    h, wm1 = wh.shape
    return neighbours(torch.ones((1, h, wm1 + 1), dtype=wh.dtype), wh, wv)[0]


def energy(x: torch.Tensor, q: torch.Tensor, free: torch.Tensor,
           wh: torch.Tensor, wv: torch.Tensor, lam: float = 1.) -> float:
    unary = .5 * ((x - q).square() * free).sum()
    pair = .5 * lam * ((wh[None] * (x[:, :, 1:] - x[:, :, :-1]).square()).sum()
                       + (wv[None] * (x[:, 1:, :] - x[:, :-1, :]).square()).sum())
    return float(unary + pair)


@torch.no_grad()
def anchored_solve(q: torch.Tensor, free: torch.Tensor,
                   wh: torch.Tensor, wv: torch.Tensor,
                   lam: float = 1., steps: int = 64) -> Tuple[torch.Tensor, Dict]:
    """Fixed synchronous Jacobi steps for a strictly convex anchored graph energy.

    Values outside free stay exactly q. No node labels are rounded to 0/1.
    The positive unary remains for every free node, even without both seed classes.
    """
    _float_cpu(q, 'q'); _float_cpu(wh, 'wh'); _float_cpu(wv, 'wv')
    if q.ndim != 3 or free.shape != q.shape or free.dtype != torch.bool:
        raise ValueError('CHW q and same bool free mask required')
    _, h, w = q.shape
    if wh.shape != (h, w-1) or wv.shape != (h-1, w):
        raise ValueError('four-neighbour edge shapes')
    if lam < 0 or not math.isfinite(lam) or type(steps) is not int or steps < 0:
        raise ValueError('invalid solver setting')
    if bool(((q < 0) | (q > 1)).any()) or any(bool(((z < 0) | (z > 1)).any()) for z in (wh, wv)):
        raise ValueError('probabilities and weights must be in [0,1]')
    q = q.detach().double(); wh = wh.detach().double(); wv = wv.detach().double()
    free = free.detach()
    d = degree(wh, wv)
    x = q.clone()
    before = energy(x, q, free, wh, wv, lam)
    for _ in range(steps):
        proposal = (q + lam * neighbours(x, wh, wv)) / (1 + lam * d[None])
        x = torch.where(free, proposal, q)
    residual = ((1 + lam * d[None]) * x - lam * neighbours(x, wh, wv) - q)[free]
    info = dict(free_nodes=int(free.sum()), steps=steps,
                energy_before=before, energy_after=energy(x, q, free, wh, wv, lam),
                max_free_linear_residual=0. if not residual.numel() else float(residual.abs().max()),
                max_contraction=float((lam*d/(1+lam*d)).max()),
                exact_fixed_nodes=bool(torch.equal(x[~free], q[~free])))
    return x, info


@torch.no_grad()
def refine_target(rgb: torch.Tensor, q: torch.Tensor, views: torch.Tensor,
                  mode: str, visit: int, grid_size: int = 128,
                  radius: int = 2, lam: float = 1., steps: int = 64,
                  seed: int = 20260907) -> Tuple[torch.Tensor, Dict, Dict]:
    """Reference input: RGB [1,3,H,W], q [1,2,H,W], views [6,2,H,W].

    Production H=W=512, grid_size=128. Smaller sizes are a testing seam only.
    Returns detached teacher, public scalar audit, and transient evaluator masks.
    Transient masks must NOT be logged densely, retained, or returned to learning.
    """
    for name, x in [('rgb', rgb), ('q', q), ('views', views)]: _float_cpu(x, name)
    if mode not in MODES or q.ndim != 4 or q.shape[:2] != (1, 2):
        raise ValueError('invalid mode or single OD/OC q')
    h, w = q.shape[2:]
    if rgb.shape != (1,3,h,w) or views.shape != (6,2,h,w):
        raise ValueError('input shape mismatch')
    if h != w or grid_size < 2 or h % grid_size:
        raise ValueError('square integral pooling grid required')
    if any(bool(((x < 0) | (x > 1)).any()) for x in (rgb,q,views)):
        raise ValueError('raw RGB and probabilities must lie in [0,1]')
    if not torch.allclose(q[0], views.mean(0), atol=1e-6, rtol=0):
        raise ValueError('q must be the unmodified six-view mean')
    q0 = q.detach(); v0 = views.detach()
    factor = h // grid_size
    qg = F.avg_pool2d(q0.double(), factor, factor)[0]
    vg = F.avg_pool2d(v0.double(), factor, factor)
    rg = F.avg_pool2d(rgb.detach().double(), factor, factor)[0]
    reliable_g = reliability(qg, vg)
    band = contour_band(qg >= .5, radius)
    free = ~reliable_g if mode == 'G_GLOBAL' else band & ~reliable_g
    wh, wv = edge_weights(rg, mode, visit, seed)
    solved, info = anchored_solve(qg, free, wh, wv, lam, steps)
    correction = solved - qg
    up = F.interpolate(correction[None], size=(h,w), mode='bilinear', align_corners=False).to(q0)
    free_full = F.interpolate(free[None].float(), size=(h,w), mode='nearest').bool()
    reliable_full = reliability(q0[0], v0)[None]
    allowed = free_full & ~reliable_full
    target = torch.where(allowed, (q0 + up).clamp(0.,1.), q0)
    diff = target - q0
    info.update(mode=mode, grid=grid_size, potential_undirected_edges=int(wh.numel()+wv.numel()),
                reliable_grid_counts=[int(x.sum()) for x in reliable_g],
                free_grid_counts=[int(x.sum()) for x in free],
                allowed_full_counts=[int(x.sum()) for x in allowed[0]],
                qstar_changed_full_counts=[int((x!=0).sum()) for x in diff[0]],
                mean_abs_probability_correction=float(diff.abs().mean()),
                max_abs_probability_correction=float(diff.abs().max()),
                weight_h_mean=float(wh.mean()), weight_v_mean=float(wv.mean()),
                exact_outside_allowed=bool(torch.equal(target[~allowed],q0[~allowed])),
                exact_reliable_full=bool(torch.equal(target[reliable_full],q0[reliable_full])))
    return target.detach(), info, dict(allowed=allowed.detach(), reliable=reliable_full.detach())
