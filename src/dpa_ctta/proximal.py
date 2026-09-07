"""Closed-form diagonal proximal update."""

import math
from numbers import Real

import torch

from .types import ProximalResult


def closed_form_update(previous_state, atlas_result, temporal_lambda, precision_floor):
    for value in (temporal_lambda, precision_floor):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value <= 0:
            raise ValueError("temporal_lambda and precision_floor must be finite and positive")
    tensors = (previous_state, atlas_result.precision, atlas_result.rhs, atlas_result.weights)
    if not all(isinstance(t, torch.Tensor) and t.is_floating_point() for t in tensors):
        raise ValueError("state, precision, rhs and weights must be floating tensors")
    if previous_state.ndim != 2 or 0 in previous_state.shape:
        raise ValueError("previous state must be nonempty [batch, latent_dim]")
    if any(t.shape != previous_state.shape for t in tensors[1:3]):
        raise ValueError("previous state, precision and rhs must have exactly the same shape")
    if any(t.dtype != previous_state.dtype or t.device != previous_state.device for t in tensors[1:]):
        raise ValueError("atlas tensors must match state dtype and device")
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError("atlas tensors and previous state must be finite")
    weights = atlas_result.weights
    if (weights.ndim != 2 or weights.shape[0] != previous_state.shape[0] or weights.shape[1] == 0
            or (weights < 0).any() or not torch.allclose(weights.sum(1), torch.ones_like(weights[:, 0]), atol=1e-6)):
        raise ValueError("weights must be nonnegative normalized [batch, anchors]")
    if not torch.isfinite(atlas_result.precision).all() or (atlas_result.precision <= 0).any():
        raise ValueError("precision must be finite and positive")
    if (atlas_result.precision < precision_floor).any():
        raise ValueError("precision is below the claimed floor")
    state = (atlas_result.rhs + temporal_lambda * previous_state) / (
        atlas_result.precision + temporal_lambda
    )
    if not torch.isfinite(state).all():
        raise FloatingPointError("nonfinite proximal result")
    return ProximalResult(
        state=state,
        weights=atlas_result.weights,
        precision=atlas_result.precision,
        rhs=atlas_result.rhs,
        contraction_bound=temporal_lambda / (temporal_lambda + precision_floor),
    )
