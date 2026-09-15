"""R4D mathematical reference; not an integrated CTTA host.

Frozen convolution weights are never overwritten. Proposed kernel-level properties
are not guarantees of boundary correctness or network-level forgetting resistance.
"""
from __future__ import annotations
import math
from typing import Literal
import torch
from torch import Tensor, nn

Mode = Literal['KDG', 'K_ALL', 'K_MAG', 'K_FREE']
MODES = ('KDG', 'K_ALL', 'K_MAG', 'K_FREE')


def split_spatial_dc(weight: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Orthogonal spatial-DC projection for OIHW kernels, computed in float64."""
    if weight.ndim != 4 or min(weight.shape) < 1:
        raise ValueError('expected nonempty OIHW weights')
    if not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError('finite floating point weights required')
    w = weight.detach().to(torch.float64)
    h, k = w.shape[-2:]
    b = torch.ones((h, k), dtype=w.dtype, device=w.device) / math.sqrt(h*k)
    # Factor the constant basis out of reduction: exact zero sums remain zero
    # without inventing a source-norm threshold or cutting coordinate gradients.
    a0 = w.sum((-2, -1)) / math.sqrt(h*k)
    detail = w - a0[..., None, None]*b
    return a0, detail, b


def cayley_from_upper(u: Tensor, channels: int) -> Tensor:
    """Q=(I+S/2)^(-1)(I-S/2); upper triangular entries of S are u.

    Zero initialization is identity but must retain a nonzero derivative. Do not
    replace it with a Python identity fast-path when u happens to be zero.
    """
    if channels < 1 or u.ndim != 1 or u.numel() != channels*(channels-1)//2:
        raise ValueError('incorrect skew coordinates')
    x = u.to(torch.float64)
    idx = torch.triu_indices(channels, channels, offset=1, device=x.device)
    upper = torch.zeros((channels, channels), dtype=x.dtype, device=x.device)
    upper = upper.index_put((idx[0], idx[1]), x)
    skew = upper - upper.T
    eye = torch.eye(channels, dtype=x.dtype, device=x.device)
    return torch.linalg.solve(eye + .5*skew, eye - .5*skew)


class KernelGeometry(nn.Module):
    """Effective-weight generator for one frozen kernel, not a network wrapper.

    Adaptable tensors default to the frozen weight dtype (normally float32).
    Small matrix algebra uses float64, and the effective weight is cast back.
    Constructor consumes no RNG; all added coordinates initialize at zero.
    """
    def __init__(self, weight: Tensor, mode: Mode):
        super().__init__()
        if mode not in MODES:
            raise ValueError('unknown mode')
        a0, _, b = split_spatial_dc(weight)
        self.mode = mode
        self.out_channels, self.in_channels = weight.shape[:2]
        self.register_buffer('base_weight', weight.detach().clone())
        self.register_buffer('a0', a0.clone())
        self.register_buffer('dc_basis', b.clone())
        opts = dict(dtype=weight.dtype, device=weight.device)
        if mode in ('KDG', 'K_ALL'):
            self.u = nn.Parameter(torch.zeros(self.in_channels*(self.in_channels-1)//2, **opts))
        else:
            self.register_parameter('u', None)
        if mode in ('KDG', 'K_ALL', 'K_MAG'):
            self.log_gain = nn.Parameter(torch.zeros(self.out_channels, **opts))
        else:
            self.register_parameter('log_gain', None)
        if mode == 'K_FREE':
            self.free_delta = nn.Parameter(torch.zeros((self.out_channels, self.in_channels), **opts))
        else:
            self.register_parameter('free_delta', None)

    def components(self) -> tuple[Tensor, Tensor, Tensor]:
        eye = torch.eye(self.in_channels, dtype=torch.float64, device=self.base_weight.device)
        q = eye if self.u is None else cayley_from_upper(self.u, self.in_channels)
        gain = (torch.ones(self.out_channels, dtype=torch.float64, device=self.base_weight.device)
                if self.log_gain is None else self.log_gain.double().exp())
        if self.mode == 'K_FREE':
            # Zero source DC rows remain zero for all methods; no source-derived
            # threshold/epsilon or invented source prototype is introduced.
            row_scale = self.a0.norm(dim=1, keepdim=True) / math.sqrt(self.in_channels)
            delta_a = row_scale * self.free_delta.double()
        else:
            # Residual form avoids reconstructing the full original kernel.
            delta_a = (gain[:, None]-1)*self.a0 + gain[:, None]*(self.a0 @ (q-eye))
        return q, gain, delta_a

    def forward(self) -> Tensor:
        q, gain, delta_a = self.components()
        w = self.base_weight.double()
        if self.mode == 'K_ALL':
            eye = torch.eye(self.in_channels, dtype=w.dtype, device=w.device)
            delta = ((gain[:, None, None, None]-1)*w +
                     gain[:, None, None, None]*torch.einsum('oihw,ij->ojhw', w, q-eye))
        else:
            delta = delta_a[..., None, None]*self.dc_basis
        effective = (w+delta).to(self.base_weight.dtype)
        if not torch.isfinite(effective).all():
            raise FloatingPointError('nonfinite effective kernel; no silent clipping')
        return effective

    @torch.no_grad()
    def audit(self) -> dict[str, float | int | str]:
        w = self().double()
        a = (w*self.dc_basis).sum((-2, -1))
        old = self.base_weight.double()
        h = w-a[..., None, None]*self.dc_basis
        h0 = old-self.a0[..., None, None]*self.dc_basis
        q, gain, _ = self.components()
        eye = torch.eye(self.in_channels, dtype=q.dtype, device=q.device)
        rows = self.a0.norm(dim=1) > 0
        if rows.sum() >= 2:
            x = torch.nn.functional.normalize(a[rows], dim=1)
            y = torch.nn.functional.normalize(self.a0[rows], dim=1)
            gram_error = (x@x.T-y@y.T).abs().max().item()
        else:
            gram_error = 0.0
        return dict(mode=self.mode, parameters=sum(p.numel() for p in self.parameters()),
                    detail_change_l2=(h-h0).norm().item(),
                    dc_change_l2=(a-self.a0).norm().item(),
                    flattened_cosine_gram_max_error=(torch.nn.functional.normalize(w.flatten(1),dim=1)@torch.nn.functional.normalize(w.flatten(1),dim=1).T-torch.nn.functional.normalize(old.flatten(1),dim=1)@torch.nn.functional.normalize(old.flatten(1),dim=1).T).abs().max().item(),
                    effective_change_l2=(w-old).norm().item(),
                    dc_cosine_gram_max_error=gram_error,
                    nonzero_source_dc_rows=int(rows.sum()),
                    orthogonality_error=(q.T@q-eye).norm().item(),
                    gain_min=gain.min().item(), gain_max=gain.max().item())
