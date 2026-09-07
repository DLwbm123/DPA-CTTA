"""Per-image Bernoulli region/boundary supervision for fixed known masks."""

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class MedicalLoss:
    total: torch.Tensor
    region: torch.Tensor
    boundary: torch.Tensor
    defined_boundary_pairs: int


def medical_loss(logits, mask, signed_distance=None, beta_boundary=0.1):
    if not math.isfinite(beta_boundary) or beta_boundary < 0:
        raise ValueError("beta_boundary must be finite and nonnegative")
    if (not isinstance(logits, torch.Tensor) or logits.ndim != 4 or 0 in logits.shape
            or not logits.is_floating_point() or not torch.isfinite(logits).all()):
        raise ValueError("finite floating logits [image, independent class, height, width] required")
    if (not isinstance(mask, torch.Tensor) or mask.shape != logits.shape
            or mask.device != logits.device or mask.dtype != logits.dtype or mask.requires_grad
            or not ((mask == 0) | (mask == 1)).all()):
        raise ValueError("fixed binary known mask must exactly match logits shape/dtype/device")
    positive = mask.sum((-2, -1))
    negative = mask.shape[-2] * mask.shape[-1] - positive
    defined = (positive > 0) & (negative > 0)
    bce = F.binary_cross_entropy_with_logits(logits, mask, reduction="none")
    pos_loss = (bce * mask).sum((-2, -1)) / positive.clamp_min(1)
    neg_loss = (bce * (1 - mask)).sum((-2, -1)) / negative.clamp_min(1)
    balanced = torch.where(defined, 0.5 * (pos_loss + neg_loss), pos_loss + neg_loss)
    probability = logits.sigmoid()
    dice = (2 * (probability * mask).sum((-2, -1)) + 1e-6) / (
        probability.sum((-2, -1)) + positive + 1e-6)
    region = (balanced + 1 - dice).mean()
    boundary = logits.sum() * 0
    if signed_distance is None:
        if beta_boundary > 0 and defined.any():
            raise ValueError("normalized signed distance required for defined boundary pairs")
    else:
        d = signed_distance
        if (not isinstance(d, torch.Tensor) or d.shape != mask.shape or d.dtype != mask.dtype
                or d.device != mask.device or d.requires_grad or not torch.isfinite(d).all()
                or (d.abs() > 1).any()):
            raise ValueError("fixed finite signed distance, normalized by image diagonal, required")
        selected = defined[:, :, None, None].expand_as(mask)
        if ((d[selected & (mask == 1)] >= 0).any()
                or (d[selected & (mask == 0)] <= 0).any()):
            raise ValueError("distance sign mismatch: strictly negative inside, positive outside")
        if defined.any():
            boundary = (d * (probability - mask)).mean((-2, -1))[defined].mean()
    return MedicalLoss(region + beta_boundary * boundary, region, boundary, int(defined.sum()))
