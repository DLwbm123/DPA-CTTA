"""Task-package math, integrated with isolated nested erasure and lifecycle."""
from dataclasses import dataclass
from typing import Optional
import hashlib,math
import torch

def local_seed(namespace: str, visit: int, base: int = 20260907) -> int:
    if visit < 1:
        raise ValueError("visit must be positive")
    payload = f"R1:{namespace}:{base}:{visit}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)

def bernoulli_entropy_bits(p: torch.Tensor) -> torch.Tensor:
    if not p.is_floating_point() or not torch.isfinite(p).all():
        raise ValueError("finite floating probabilities required")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("probabilities outside [0,1]")
    q = p.clamp(1e-6, 1 - 1e-6)
    return -(q * q.log() + (1 - q) * torch.log1p(-q)) / math.log(2)

@torch.no_grad()
def dense_sensitivity(p: torch.Tensor, valid: torch.Tensor, min_pixels: int = 16):
    """p: [3,C,H,W]; valid: [H,W], same unerased pixels for all views.

    Returns a scalar tensor and a list of valid (channel, region, count).
    No raw tensor is logged by production code.
    """
    if p.ndim != 4 or p.shape[0] != 3:
        raise ValueError("three aligned [C,H,W] probability views required")
    if valid.dtype != torch.bool or valid.shape != p.shape[-2:]:
        raise ValueError("valid must be a spatial bool tensor")
    h = bernoulli_entropy_bits(p.detach())
    scores, groups = [], []
    for c in range(p.shape[1]):
        for b in (0, 1):
            region = valid & ((p[0, c] >= 0.5) == bool(b))
            n = int(region.sum())
            if n < min_pixels:
                continue
            e = h[:, c, region].mean(dim=1)
            scores.append((abs(e[1] - e[0]) + abs(e[2] - e[1])) / 2)
            groups.append((c, b, n))
    if not scores:
        return None, groups
    return torch.stack(scores).mean(), groups

@dataclass
class TrendController:
    zeta: float = 0.9
    horizon: int = 50
    deterioration: float = 6.0
    floor: float = 1e-6
    age: int = 0
    ema: Optional[float] = None
    best: Optional[float] = None
    resets: int = 0

    def observe(self, sensitivity: float) -> dict:
        if not math.isfinite(sensitivity) or not 0 <= sensitivity <= 1:
            raise ValueError("finite sensitivity in [0,1] required")
        if self.ema is None:
            self.ema = sensitivity
            self.best = max(sensitivity, self.floor)
            self.age = 1
        else:
            self.ema = self.zeta * self.ema + (1 - self.zeta) * sensitivity
            self.best = min(self.best, max(self.ema, self.floor))
            self.age += 1
        trigger = self.age >= self.horizon and self.ema > (1 + self.deterioration) * self.best
        result = dict(trigger=trigger, age=self.age, ema=self.ema, best=self.best)
        if trigger:
            self.resets += 1
            self.age, self.ema, self.best = 0, None, None
        result["resets"] = self.resets
        return result

def periodic_due(visit):
    if type(visit) is not int or visit<1:raise ValueError('positive global visit')
    return visit>1 and (visit-1)%256==0


def nested_erasure(x,visit):
    if x.ndim!=4 or x.shape[0]!=1 or x.shape[-2]!=x.shape[-1]:raise ValueError('square single image')
    H=x.shape[-1];large=math.floor(math.sqrt(.20)*H);small=math.floor(math.sqrt(.10)*H)
    g=torch.Generator(device='cpu').manual_seed(local_seed('erasure',visit))
    y,z=[int(torch.randint(H-large+1,(),generator=g)) for _ in range(2)]
    dy,dz=[int(torch.randint(large-small+1,(),generator=g)) for _ in range(2)]
    m20=torch.zeros(H,H,dtype=torch.bool,device=x.device);m20[y:y+large,z:z+large]=True
    m10=torch.zeros_like(m20);m10[y+dy:y+dy+small,z+dz:z+dz+small]=True
    return (x,x.masked_fill(m10,0),x.masked_fill(m20,0)),~m20,dict(small_pixels=small**2,large_pixels=large**2,valid_pixels=H*H-large**2)
