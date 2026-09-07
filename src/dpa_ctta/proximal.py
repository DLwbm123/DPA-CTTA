"""Closed-form diagonal proximal update."""

import torch

from .types import ProximalResult


def closed_form_update(previous_state, atlas_result, temporal_lambda, precision_floor):
    if temporal_lambda <= 0 or precision_floor <= 0:
        raise ValueError("temporal_lambda and precision_floor must be positive")
    if previous_state.shape != atlas_result.precision.shape:
        raise ValueError("previous state and atlas tensors must have the same shape")
    if not torch.isfinite(atlas_result.precision).all() or (atlas_result.precision <= 0).any():
        raise ValueError("precision must be finite and positive")
    if (atlas_result.precision < precision_floor).any():
        raise ValueError("precision is below the claimed floor")
    state = (atlas_result.rhs + temporal_lambda * previous_state) / (
        atlas_result.precision + temporal_lambda
    )
    return ProximalResult(
        state=state,
        weights=atlas_result.weights,
        precision=atlas_result.precision,
        rhs=atlas_result.rhs,
        contraction_bound=temporal_lambda / (temporal_lambda + precision_floor),
    )
