"""Trainable synthetic potential anchors."""

import torch
from torch import nn
import torch.nn.functional as F


class DistilledPotentialAnchor(nn.Module):
    def __init__(
        self,
        image_logits,
        mask_logits,
        state_center,
        local_map,
        precision_raw,
        precision_floor=1e-3,
    ):
        super().__init__()
        tensors = (image_logits, mask_logits, state_center, local_map, precision_raw)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("anchor fields must be Tensors")
        if image_logits.ndim != 4 or image_logits.shape[0] != 1 or image_logits.shape[1] != 3:
            raise ValueError("image_logits must have shape [1, 3, height, width]")
        if mask_logits.ndim != 4 or mask_logits.shape[0] != 1:
            raise ValueError("mask_logits must have shape [1, classes, height, width]")
        if image_logits.shape[-2:] != mask_logits.shape[-2:]:
            raise ValueError("synthetic image and mask spatial shapes must match")
        if state_center.ndim != 1 or precision_raw.shape != state_center.shape:
            raise ValueError("state_center and precision_raw must have shape [latent_dim]")
        if local_map.ndim != 2 or local_map.shape[0] != state_center.numel():
            raise ValueError("local_map must have shape [latent_dim, descriptor_dim]")
        if precision_floor <= 0:
            raise ValueError("precision_floor must be positive")
        self.image_logits = nn.Parameter(image_logits)
        self.mask_logits = nn.Parameter(mask_logits)
        self.state_center = nn.Parameter(state_center)
        self.local_map = nn.Parameter(local_map)
        self.precision_raw = nn.Parameter(precision_raw)
        self.precision_floor = float(precision_floor)

    @property
    def image(self):
        return self.image_logits.sigmoid()

    @property
    def soft_mask(self):
        return self.mask_logits.sigmoid()

    @property
    def precision(self):
        numerical_min = torch.finfo(self.precision_raw.dtype).eps
        return F.softplus(self.precision_raw).clamp_min(numerical_min) + self.precision_floor

    def descriptor(self, frozen_descriptor, frozen_feature):
        return frozen_descriptor(self.image, self.soft_mask, frozen_feature)
