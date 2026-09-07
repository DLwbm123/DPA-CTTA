"""Prepared source/fixture proxy data; never a target-evaluator payload."""

from dataclasses import dataclass
from enum import Enum

import torch

from .medical_losses import medical_loss


class ProxyProvenance(Enum):
    SOURCE = "known_source_annotation"
    FIXTURE = "procedural_geometry_fixture"


@dataclass(frozen=True)
class FixedProxy:
    pixel_rgb: torch.Tensor
    mask: torch.Tensor
    signed_distance: torch.Tensor | None
    provenance: ProxyProvenance

    def __post_init__(self):
        if type(self.provenance) is not ProxyProvenance:
            raise ValueError("proxy provenance must be explicitly source or procedural fixture")
        # Own fixed tensors; a caller's later in-place edits cannot change this prepared proxy.
        for field in ("pixel_rgb", "mask", "signed_distance"):
            tensor = getattr(self, field)
            if tensor is None and field == "signed_distance":
                continue
            if not isinstance(tensor, torch.Tensor) or tensor.requires_grad:
                raise ValueError("proxy tensors must be fixed, without an autograd graph")
            object.__setattr__(self, field, tensor.detach().clone())


def proxy_supervision(model, prompt, model_input, proxy, beta_boundary):
    output = model(prompt(model_input)[0])
    logits = output[0] if isinstance(output, tuple) else output
    return medical_loss(logits, proxy.mask, proxy.signed_distance, beta_boundary)
