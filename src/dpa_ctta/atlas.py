"""Soft association and local-chart aggregation."""

import torch
from torch import nn

from .types import AtlasResult


class DistilledPotentialAtlas(nn.Module):
    def __init__(self, anchors, descriptor, scaler=None, temperature=1.0):
        super().__init__()
        if not anchors:
            raise ValueError("atlas needs at least one anchor")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.anchors = nn.ModuleList(anchors)
        self.descriptor = descriptor
        self.scaler = scaler if scaler is not None else nn.Identity()
        self.temperature = float(temperature)

    def anchor_descriptors(self, frozen_feature_fn):
        values = []
        for anchor in self.anchors:
            feature = frozen_feature_fn(anchor.image)
            descriptor = anchor.descriptor(self.descriptor, feature)
            if descriptor.shape[0] != 1:
                raise ValueError("each anchor must describe one synthetic image")
            values.append(descriptor[0])
        return torch.stack(values)

    def forward(self, query_descriptor, frozen_feature_fn):
        if query_descriptor.ndim != 2:
            raise ValueError("query_descriptor must have shape [batch, descriptor_dim]")
        anchor_descriptors = self.anchor_descriptors(frozen_feature_fn)
        scaled_query = self.scaler(query_descriptor)
        scaled_anchors = self.scaler(anchor_descriptors)
        raw_delta = query_descriptor[:, None, :] - anchor_descriptors[None, :, :]
        scaled_delta = scaled_query[:, None, :] - scaled_anchors[None, :, :]
        weights = torch.softmax(-scaled_delta.square().sum(2) / self.temperature, dim=1)

        centers = torch.stack([anchor.state_center for anchor in self.anchors])
        maps = torch.stack([anchor.local_map for anchor in self.anchors])
        if maps.shape[2] != query_descriptor.shape[1]:
            raise ValueError("anchor local_map descriptor dimension mismatch")
        local_centers = centers[None, :, :] + torch.einsum("kdq,bkq->bkd", maps, raw_delta)
        anchor_precision = torch.stack([anchor.precision for anchor in self.anchors])
        weighted_precision = weights[:, :, None] * anchor_precision[None, :, :]
        precision = weighted_precision.sum(1)
        rhs = (weighted_precision * local_centers).sum(1)
        return AtlasResult(weights=weights, precision=precision, rhs=rhs)
