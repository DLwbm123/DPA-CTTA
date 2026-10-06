"""One frozen pilot: delayed prototypes, regional soft correction, BN-only C."""
import numpy as np
import torch
from scipy.ndimage import label
from torch.nn import functional as F
from ..r20_model_only_search.method import Host as Base, weights_for


def correction(q, history, ready):
    """No labels; historical class support and connected disagreement regions."""
    if history is None or not ready:
        return q.detach(), torch.zeros_like(q[:, :1]), 0
    # Three mutually exclusive candidate classes preserve OD/OC nesting.
    pi = torch.cat((1-history[:, :1], history[:, :1]-history[:, 1:], history[:, 1:]), 1)
    confidence = pi.max(1).values[0].cpu().numpy()
    disagreement = ((q >= .5) != (history >= .5)).any(1)[0].cpu().numpy()
    regions, count = label(disagreement)
    sizes = np.bincount(regions.ravel(), minlength=count+1)
    sums = np.bincount(regions.ravel(), weights=confidence.ravel(), minlength=count+1)
    means = sums / np.maximum(sizes, 1)
    accepted = (sizes >= 32) & (means >= .7); accepted[0] = False
    strength = np.where(accepted, np.clip(means-.5, 0, .5), 0)
    alpha = torch.as_tensor(strength[regions], device=q.device, dtype=q.dtype)[None, None]
    return (q + alpha*(history-q)).detach(), alpha, int(accepted.sum())


def regional_loss(z, target, weights, alpha):
    pixel = F.binary_cross_entropy_with_logits(z, target, reduction='none') * weights
    base = pixel.mean()
    regions, count = label((alpha[0, 0] > 0).cpu().numpy())
    if not count:
        return base
    pieces = []; strengths = []
    for k in range(1, count+1):
        mask = torch.as_tensor(regions == k, device=z.device)[None, None].expand_as(z)
        a = alpha[0, 0][torch.as_tensor(regions == k, device=z.device)].mean()
        # Normalize within each region, retaining its reliability outside the ratio.
        pieces.append(a * pixel[mask].sum() / weights[mask].sum().clamp_min(1e-8))
        strengths.append(a)
    strength = torch.stack(strengths).mean()
    return (1-.25*strength)*base + .25*torch.stack(pieces).mean()


class Host(Base):
    def _criterion(self, z, q_native):
        self.last_before = q_native.detach().cpu()
        # Reuse the existing C/W criterion without the older near-boundary P edit.
        p = self.modules.pop('P', None)
        try:
            base = super()._criterion(z, q_native)
        finally:
            if p is not None:
                self.modules['P'] = p
        ready = p is not None and all(len(bank) >= 2 for bank in self.proto.values())
        target, alpha, regions = correction(q_native.detach(), None if self.proto_target is None else self.proto_target.to(z), ready)
        self.last_after = target.cpu()
        self.last_low_variance = ((torch.stack(self.weak).sigmoid().var(0, unbiased=False) <= .001)).cpu()
        self.diag.update(correction_regions=regions, correction_coverage=float((alpha > 0).float().mean()), correction_strength=float(alpha.mean()), correction_target_change=float((target-q_native).abs().mean()), history_ready=ready)
        if not regions:
            return base
        w = torch.ones_like(z)
        if 'W' in self.modules:
            w = weights_for(torch.stack(self.weak).sigmoid(), .1, 2).to(z)
        loss = regional_loss(z, target, w, alpha)
        self.diag['pseudo_loss'] = float(loss.detach())
        return loss
