"""Multi-scale latent FiLM with an exact zero-state bypass."""

import torch
from torch import nn
import torch.nn.functional as F


class LatentFiLM(nn.Module):
    def __init__(self, channels, latent_dim=16):
        super().__init__()
        if channels <= 0 or latent_dim <= 0:
            raise ValueError("channels and latent_dim must be positive")
        self.channels = channels
        self.latent_dim = latent_dim
        self.gamma_basis = nn.Parameter(torch.empty(channels, latent_dim))
        self.beta_basis = nn.Parameter(torch.empty(channels, latent_dim))
        nn.init.normal_(self.gamma_basis, std=1e-3)
        nn.init.normal_(self.beta_basis, std=1e-3)

    def forward(self, feature, state):
        if feature.ndim != 4 or feature.shape[1] != self.channels:
            raise ValueError("feature shape does not match FiLM channels")
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[1] != self.latent_dim:
            raise ValueError("state must have shape [latent_dim] or [batch, latent_dim]")
        if state.shape[0] not in {1, feature.shape[0]}:
            raise ValueError("state batch must be one or match feature batch")
        if not torch.any(state) and not state.requires_grad:
            return feature
        gamma = F.linear(state, self.gamma_basis).view(state.shape[0], self.channels, 1, 1)
        beta = F.linear(state, self.beta_basis).view(state.shape[0], self.channels, 1, 1)
        return feature * torch.exp(gamma) + beta


class MultiScaleLatentFiLM(nn.Module):
    """Inject one latent state into exactly three named source-model outputs."""

    def __init__(self, source_model, injection_channels, latent_dim=16):
        super().__init__()
        if len(injection_channels) != 3:
            raise ValueError("exactly three injection points are required")
        modules = dict(source_model.named_modules())
        missing = set(injection_channels) - set(modules)
        if missing:
            raise ValueError(f"missing injection modules: {sorted(missing)}")
        self.source_model = source_model.requires_grad_(False).eval()
        self.latent_dim = latent_dim
        self.injection_paths = tuple(injection_channels)
        self.adapters = nn.ModuleList(
            [LatentFiLM(injection_channels[path], latent_dim) for path in self.injection_paths]
        )
        self._active_state = None
        self._closed = False
        self._handles = [
            modules[path].register_forward_hook(self._make_hook(adapter))
            for path, adapter in zip(self.injection_paths, self.adapters)
        ]

    def _make_hook(self, adapter):
        def hook(_module, _inputs, output):
            return output if self._active_state is None else adapter(output, self._active_state)
        return hook

    def train(self, mode=True):
        super().train(mode)
        self.source_model.eval()
        return self

    def forward(self, image, state=None):
        if self._closed:
            raise RuntimeError("MultiScaleLatentFiLM is closed")
        if state is not None and state.shape[-1] != self.latent_dim:
            raise ValueError("state latent dimension mismatch")
        self.source_model.eval()
        self._active_state = state
        try:
            return self.source_model(image)
        finally:
            self._active_state = None

    def close(self):
        if not self._closed:
            for handle in self._handles:
                handle.remove()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False
