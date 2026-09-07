"""Small tensor result records used by the public API."""

from dataclasses import dataclass
from torch import Tensor


@dataclass(frozen=True)
class AtlasResult:
    weights: Tensor
    precision: Tensor
    rhs: Tensor


@dataclass(frozen=True)
class ProximalResult:
    state: Tensor
    weights: Tensor
    precision: Tensor
    rhs: Tensor
    contraction_bound: float


@dataclass(frozen=True)
class OnlineResult:
    logits: Tensor
    state: Tensor
    weights: Tensor
    precision: Tensor
    rhs: Tensor
    contraction_bound: float


@dataclass(frozen=True)
class InjectionPoint:
    path: str
    channels: int
    shape: tuple[int, ...]
    output_type: str
    jvp_norm: float
