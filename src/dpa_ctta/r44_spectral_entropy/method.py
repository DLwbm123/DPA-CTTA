"""Causal, per-binary-channel 2D adaptation of spectral entropy regularization."""
import copy
from collections import deque
import torch
from ..r37_cw_orientation.method import Host as RetainedHost, candidates as retained_candidates


def candidates():
    c, w, _, lso = retained_candidates()
    return [c, w, lso, dict(lso, id='CW_LSO_SE', spectral_strength=.01,
                          spectral_bands=5, spectral_window=50)]


def band_powers(logits, bands=5):
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != 2:
        raise ValueError('one image and independent OD/OC binary channels required')
    h, w = logits.shape[-2:]
    fy = torch.fft.fftfreq(h, device=logits.device) * h
    fx = torch.fft.fftfreq(w, device=logits.device) * w
    radius = (fy[:, None].square() + fx[None, :].square()).sqrt()
    edge = min(h, w) / (2 * bands)
    # Centered binary logits +/-z/2 give max=|z|/2; fixed scale cancels in band ratios.
    energy = torch.fft.fft2(logits.float().abs(), norm='ortho').abs().square()
    return torch.stack([energy[..., (radius > k*edge) & (radius <= (k+1)*edge)].sum(-1)
                        for k in range(bands)], -1).squeeze(0)


def entropy(powers):
    p = powers / (powers.sum(-1, keepdim=True) + 1e-6)
    return -(p * (p + 1e-6).log()).sum(-1)


class Host(RetainedHost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spectral_history = deque(maxlen=int(self.config.get('spectral_window', 50)))
        self.pending_spectral = None

    def _criterion(self, logits, pseudo):
        base = super()._criterion(logits, pseudo)
        strength = float(self.config.get('spectral_strength', 0))
        if not strength:
            return base
        if self.pending_spectral is not None:
            raise ValueError('exactly one consistency criterion per arrival required')
        powers = band_powers(logits, int(self.config['spectral_bands']))
        current = entropy(powers)
        target = entropy(torch.stack(list(self.spectral_history)).mean(0).to(powers)) if self.spectral_history else current.detach()
        gap = (target-current).relu()
        loss = gap.square().mean()
        self.pending_spectral = powers.detach().cpu().clone()
        self.diag.update(spectral_loss=float(loss.detach()), spectral_entropy=current.detach().tolist(),
                         spectral_target_entropy=target.tolist(), spectral_positive_gap=gap.detach().tolist(),
                         spectral_active=(gap>0).float().detach().tolist(), spectral_band_power=powers.detach().tolist(),
                         spectral_reference_count=len(self.spectral_history), spectral_coefficient=strength)
        return base + strength*loss

    def step(self, x):
        z, detail = super().step(x)
        if self.pending_spectral is not None:
            self.spectral_history.append(self.pending_spectral)
            self.pending_spectral = None
        return z, detail

    def snapshot(self):
        saved = super().snapshot()
        if self.config.get('spectral_strength', 0):
            saved['spectral_history'] = [p.clone() for p in self.spectral_history]
        return saved

    def restore(self, saved):
        super().restore(saved)
        self.spectral_history.clear()
        self.spectral_history.extend(p.clone() for p in saved.get('spectral_history', []))
        self.pending_spectral = None
