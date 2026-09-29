"""Independent R10 mathematical reference, not a CTTA runner.

No private input, checkpoint, network, GPU, queue or authorization is opened here.
Production integration must separately test the real B carrier and journal semantics.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional
import torch

@dataclass(frozen=True)
class Memory:
    m: torch.Tensor
    q: torch.Tensor
    h: torch.Tensor

    @classmethod
    def zero(cls, *, dtype: torch.dtype = torch.float64) -> 'Memory':
        return cls(torch.zeros(64, dtype=dtype), torch.zeros(32, dtype=dtype),
                   torch.zeros((), dtype=dtype))

    def check(self) -> None:
        if self.m.shape != (64,) or self.q.shape != (32,) or self.h.shape != ():
            raise ValueError('memory shapes must be 64, 32 and scalar')
        if not all(torch.isfinite(x).all() for x in (self.m, self.q, self.h)):
            raise FloatingPointError('nonfinite memory')
        if not 0 <= self.h.detach().item() <= 1:
            raise ValueError('memory mass outside [0,1]')


def policy_observation(d: torch.Tensor, proposed: torch.Tensor,
                       memory: Memory, scale: torch.Tensor) -> torch.Tensor:
    memory.check()
    if d.shape != (32,) or proposed.shape != (64,) or scale.shape != (64,):
        raise ValueError('observation input shape')
    if not torch.isfinite(scale).all() or (scale < 1e-3).any():
        raise ValueError('source-fit scale floor')
    blocks=(d, memory.m / scale, proposed / scale, memory.h*d-memory.q)
    result=torch.cat([x.clamp(-10,10) for x in blocks]+[memory.h.reshape(1)])
    if result.shape != (193,) or not torch.isfinite(result).all():
        raise ValueError('actor observation')
    return result


def use_and_write(proposed: torch.Tensor, d: torch.Tensor, memory: Memory,
                  scale: torch.Tensor, raw_action: torch.Tensor, *,
                  forced_write: Optional[float] = None,
                  identity_use: bool = False) -> tuple[torch.Tensor, Memory]:
    """Return current use code first and committed memory second.

    identity_use=True is an equivalence-test override, never a learned extra action.
    """
    memory.check()
    if proposed.shape != (64,) or d.shape != (32,) or scale.shape != (64,) or raw_action.shape != (10,):
        raise ValueError('action dimensions')
    if (scale < 1e-3).any() or not all(torch.isfinite(x).all() for x in (proposed,d,scale,raw_action)):
        raise ValueError('finite action inputs and verified scale required')
    if identity_use:
        use=proposed
    else:
        residual=torch.cat((.5*scale[:8]*raw_action[1:9].tanh(), torch.zeros_like(scale[8:])))
        use=raw_action[0].sigmoid()*proposed+residual
    if forced_write is None:
        write=raw_action[9].sigmoid()
    else:
        if forced_write not in (0.,.5,1.):
            raise ValueError('unregistered write diagnostic')
        write=raw_action.new_tensor(forced_write)
    nxt=Memory((1-write)*memory.m+write*use,
               (1-write)*memory.q+write*d,
               (1-write)*memory.h+write)
    nxt.check()
    return use,nxt


def normal_log_prob(raw: torch.Tensor, mean: torch.Tensor, std: float = .35,
                    active_dims: int = 10) -> torch.Tensor:
    if raw.shape != mean.shape or active_dims not in (9,10) or raw.shape[-1] < active_dims:
        raise ValueError('raw action log-prob dimensions')
    if not math.isfinite(std) or std <= 0:
        raise ValueError('positive std required')
    z=(raw[..., :active_dims].double()-mean[..., :active_dims].double())/std
    return (-.5*z.square()-math.log(std)-.5*math.log(2*math.pi)).sum(-1)


def reference_kl(mean: torch.Tensor, reference_mean: torch.Tensor,
                 std: float = .35, active_dims: int = 10) -> torch.Tensor:
    """Analytic forward KL for equal-variance raw Gaussians, per-dimension mean."""
    if mean.shape != reference_mean.shape or active_dims not in (9,10):
        raise ValueError('KL dimensions')
    return ((mean[..., :active_dims].double()-reference_mean[..., :active_dims].detach().double()).square()
            /(2*std**2)).mean(-1)


def retention(anchor_dice: torch.Tensor, candidate_dice: torch.Tensor,
              *, existing_history: bool = True) -> torch.Tensor:
    """Last two axes are probes and OD/OC; supports leading group axes."""
    if candidate_dice.shape[-2:] != (2,2) or anchor_dice.shape[-2:] != (2,2):
        raise ValueError('two probes and two independent channels required')
    if not existing_history:
        return torch.ones_like(candidate_dice[..., 0,0])
    deficit=(anchor_dice.detach()-candidate_dice).clamp_min(0).mean((-2,-1))
    return torch.exp(-20*deficit)


def group_advantage(rewards: torch.Tensor, *, ema: Optional[float] = None
                    ) -> tuple[torch.Tensor, Optional[float]]:
    """One collection update. Pass ema=.01 initially ONLY for GR_RET_EMA."""
    if rewards.ndim != 1 or rewards.numel()!=4 or not torch.isfinite(rewards).all():
        raise ValueError('four finite group rewards required')
    r=rewards.detach().double()
    spread=float(r.std(unbiased=False))
    new_ema=None if ema is None else .99*float(ema)+.01*spread
    denominator=max(spread if new_ema is None else new_ema, .005)+1e-8
    return ((r-r.mean())/denominator).clamp(-5,5).detach(),new_ema


def clipped_surrogate(new_log_prob: torch.Tensor, old_log_prob: torch.Tensor,
                      advantage: torch.Tensor) -> torch.Tensor:
    """Old log-prob and reward feedback are detached; no silent ratio clipping."""
    ratio=(new_log_prob.double()-old_log_prob.detach().double()).exp()
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('nonfinite importance ratio')
    a=advantage.detach().double()
    return -torch.minimum(ratio*a,ratio.clamp(.8,1.2)*a).mean()
