"""Label-free, optimizer-free online state update."""

import torch
from torch import nn

from .hooks import capture_output
from .proximal import closed_form_update
from .types import OnlineResult


def _logits(output):
    return output[0] if isinstance(output, tuple) else output


class DPAOnlineAdapter(nn.Module):
    def __init__(
        self,
        injected_model,
        atlas,
        descriptor_feature_path,
        temporal_lambda=1.0,
        precision_floor=1e-3,
    ):
        super().__init__()
        if temporal_lambda <= 0 or precision_floor <= 0:
            raise ValueError("temporal_lambda and precision_floor must be positive")
        modules = dict(injected_model.source_model.named_modules())
        if descriptor_feature_path not in modules:
            raise ValueError(f"descriptor feature module not found: {descriptor_feature_path}")
        self.injected_model = injected_model
        self.atlas = atlas
        self.descriptor_feature_path = descriptor_feature_path
        self.temporal_lambda = float(temporal_lambda)
        self.precision_floor = float(precision_floor)
        self.register_buffer("z_prev", torch.zeros(1, injected_model.latent_dim))
        self.eval()

    def _zero_pass(self, image):
        zero = self.z_prev.new_zeros(image.shape[0], self.z_prev.shape[1])
        with capture_output(self.injected_model.source_model, self.descriptor_feature_path) as captured:
            output = self.injected_model(image, zero)
        if len(captured) != 1 or not isinstance(captured[0], torch.Tensor):
            raise RuntimeError("descriptor feature path must execute once with a Tensor output")
        return _logits(output), captured[0]

    def _frozen_feature(self, image):
        return self._zero_pass(image)[1]

    @torch.no_grad()
    def step(self, image):
        if image.ndim != 4 or image.shape[0] != 1:
            raise ValueError("online step expects one [1, 3, height, width] image")
        frozen_logits, frozen_feature = self._zero_pass(image)
        query = self.atlas.descriptor(image, frozen_logits.sigmoid(), frozen_feature)
        atlas_result = self.atlas(query, self._frozen_feature)
        update = closed_form_update(
            self.z_prev, atlas_result, self.temporal_lambda, self.precision_floor
        )
        self.z_prev.copy_(update.state)
        logits = _logits(self.injected_model(image, self.z_prev))
        return OnlineResult(
            logits=logits,
            state=self.z_prev.clone(),
            weights=update.weights,
            precision=update.precision,
            rhs=update.rhs,
            contraction_bound=update.contraction_bound,
        )

    @torch.no_grad()
    def reset(self):
        self.z_prev.zero_()
