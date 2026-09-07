"""State-independent anatomy/style descriptors for images and frozen outputs."""

import torch
from torch import nn
import torch.nn.functional as F


DESCRIPTOR_SCHEMA_VERSION = 2


class _DescriptorSchema(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("descriptor_schema_version", torch.tensor(DESCRIPTOR_SCHEMA_VERSION))

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        version = state_dict.get(prefix + "descriptor_schema_version")
        if not isinstance(version, torch.Tensor) or version.ndim != 0 or version.item() != DESCRIPTOR_SCHEMA_VERSION:
            raise RuntimeError("descriptor schema mismatch; rebuild old descriptors/scalers and caches")
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class DescriptorScaler(_DescriptorSchema):
    """Source-statistics interface; defaults to an unfitted identity transform."""

    def __init__(self, dim):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.register_buffer("center", torch.zeros(dim))
        self.register_buffer("scale", torch.ones(dim))
        self.register_buffer("fitted", torch.tensor(False))

    def set_statistics(self, center, scale):
        if center.shape != self.center.shape or scale.shape != self.scale.shape:
            raise ValueError("scaler statistics shape mismatch")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all() or (scale <= 0).any():
            raise ValueError("scaler statistics must be finite with positive scale")
        self.center.copy_(center)
        self.scale.copy_(scale)
        self.fitted.fill_(True)
        return self

    def forward(self, descriptor):
        if descriptor.shape[-1] != self.center.numel():
            raise ValueError("descriptor dimension mismatch")
        return (descriptor - self.center) / self.scale


class FrozenDescriptor(_DescriptorSchema):
    def __init__(self, task, feature_channels, projection_dim=8, eps=1e-6):
        super().__init__()
        if task not in {"fundus", "polyp"}:
            raise ValueError("unsupported task")
        if (type(feature_channels) is not int or type(projection_dim) is not int
                or feature_channels <= 0 or not 0 < projection_dim <= 2 * feature_channels):
            raise ValueError("integer projection_dim must be in [1, 2 * feature_channels]")
        self.task = task
        self.num_classes = 2 if task == "fundus" else 1
        self.feature_channels = feature_channels
        self.projection_dim = projection_dim
        self.eps = eps
        generator = torch.Generator(device="cpu").manual_seed(20260907)
        matrix = torch.randn(feature_channels * 2, projection_dim, generator=generator, dtype=torch.float64)
        projection = torch.linalg.qr(matrix, mode="reduced").Q.T.float().contiguous()
        if (not torch.isfinite(projection).all()
                or torch.linalg.matrix_rank(projection) != projection_dim
                or not torch.allclose(projection @ projection.T, torch.eye(projection_dim), atol=1e-6)):
            raise RuntimeError("invalid orthogonal descriptor projection")
        self.register_buffer("feature_projection", projection)

    @property
    def output_dim(self):
        anatomy = 7 * self.num_classes + (2 if self.task == "fundus" else 1)
        return 9 + self.projection_dim + anatomy

    def _style(self, image, feature):
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("image must have shape [batch, 3, height, width]")
        if feature.ndim != 4 or feature.shape[1] != self.feature_channels:
            raise ValueError("frozen feature shape mismatch")
        r, g, b = image.unbind(1)
        opponent = torch.stack(((r + g + b) / 3, r - g, b - (r + g) / 2), dim=1)
        opponent_stats = torch.cat(
            (opponent.mean((-2, -1)), opponent.std((-2, -1), unbiased=False)), dim=1
        )

        spectrum = torch.fft.rfft2(image.float(), norm="ortho").abs().square().mean(1)
        fy = torch.fft.fftfreq(image.shape[-2], device=image.device).abs()[:, None]
        fx = torch.fft.rfftfreq(image.shape[-1], device=image.device).abs()[None, :]
        radius = torch.sqrt(fx.square() + fy.square())
        bands = (radius <= 0.15, (radius > 0.15) & (radius <= 0.35), radius > 0.35)
        fourier = torch.stack([spectrum[:, band].mean(1) for band in bands], dim=1)
        fourier = torch.log1p(fourier)

        feature_stats = torch.cat(
            (feature.mean((-2, -1)), feature.std((-2, -1), unbiased=False)), dim=1
        )
        projected = F.linear(feature_stats, self.feature_projection.to(feature_stats.dtype))
        return torch.cat((opponent_stats, fourier.to(image.dtype), projected), dim=1)

    def _anatomy(self, soft_mask):
        if soft_mask.ndim != 4 or soft_mask.shape[1] != self.num_classes:
            raise ValueError("soft mask class count mismatch")
        if not torch.isfinite(soft_mask).all():
            raise ValueError("soft mask must be finite")
        batch, _classes, height, width = soft_mask.shape
        yy = torch.linspace(-1, 1, height, device=soft_mask.device, dtype=soft_mask.dtype)[None, None, :, None]
        xx = torch.linspace(-1, 1, width, device=soft_mask.device, dtype=soft_mask.dtype)[None, None, None, :]
        mass = soft_mask.sum((-2, -1)).clamp_min(self.eps)
        area = soft_mask.mean((-2, -1))
        cx = (soft_mask * xx).sum((-2, -1)) / mass
        cy = (soft_mask * yy).sum((-2, -1)) / mass
        mx2 = (soft_mask * (xx - cx[:, :, None, None]).square()).sum((-2, -1)) / mass
        my2 = (soft_mask * (yy - cy[:, :, None, None]).square()).sum((-2, -1)) / mass
        mxy = (
            soft_mask
            * (xx - cx[:, :, None, None])
            * (yy - cy[:, :, None, None])
        ).sum((-2, -1)) / mass
        dx = (soft_mask[..., 1:] - soft_mask[..., :-1]).abs().mean((-2, -1))
        dy = (soft_mask[..., 1:, :] - soft_mask[..., :-1, :]).abs().mean((-2, -1))
        boundary = dx + dy
        per_class = torch.stack((area, cx, cy, mx2, my2, mxy, boundary), dim=2).flatten(1)
        if self.task == "fundus":
            disc, cup = soft_mask[:, 0], soft_mask[:, 1]
            ratio = cup.mean((-2, -1)) / disc.mean((-2, -1)).clamp_min(self.eps)
            nesting = F.relu(cup - disc).mean((-2, -1))
            extra = torch.stack((ratio, nesting), dim=1)
        else:
            compactness = boundary[:, 0].square() / area[:, 0].clamp_min(self.eps)
            extra = compactness[:, None]
        return torch.cat((per_class, extra), dim=1).reshape(batch, -1)

    def forward(self, image, soft_mask, frozen_feature):
        result = torch.cat((self._style(image, frozen_feature), self._anatomy(soft_mask)), dim=1)
        if result.shape[1] != self.output_dim:
            raise RuntimeError("descriptor dimension invariant failed")
        return result
