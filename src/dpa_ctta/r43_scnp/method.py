"""Own unlabeled SCNP adaptation on the retained CW_LSO objective."""
import copy
import torch
from torch.nn import functional as F
from ..r37_cw_orientation.method import Host as RetainedHost, candidates as retained_candidates
from ..r20_model_only_search.method import weights_for


def candidates():
    c, w, _, lso = copy.deepcopy(retained_candidates())
    return [c, w, lso, dict(lso, id='CW_LSO_SCNP', scnp_strength=.1, scnp_kernel=3)]


def scnp_logits(logits, target, kernel=3):
    """Route each class's loss through its worst same-class neighbor, including self."""
    if kernel != 3 or logits.shape != target.shape or logits.ndim != 4:
        raise ValueError('registered 3x3 SCNP with matching NCHW tensors required')
    foreground = target.detach() >= .5
    low = logits.masked_fill(~foreground, torch.inf)
    high = logits.masked_fill(foreground, -torch.inf)
    low = -F.max_pool2d(F.pad(-low, (1,)*4, mode='replicate'), kernel, 1)
    high = F.max_pool2d(F.pad(high, (1,)*4, mode='replicate'), kernel, 1)
    # Self membership ensures the selected pool is nonempty, even for isolated pixels.
    return torch.where(foreground, low, high)


class Host(RetainedHost):
    def _criterion(self, logits, pseudo):
        base = super()._criterion(logits, pseudo)
        strength = float(self.config.get('scnp_strength', 0))
        if not strength:
            return base
        q = pseudo.detach()
        transformed = scnp_logits(logits, q, self.config['scnp_kernel'])
        p = self.modules['W']
        reliability = weights_for(torch.stack(self.weak).sigmoid(),
                                  p['variance_temperature'], p['boundary_boost']).to(logits)
        extra = (F.binary_cross_entropy_with_logits(transformed, q, reduction='none') * reliability).mean()
        self.diag.update(SCNP_loss=float(extra.detach()), SCNP_coefficient=strength,
                         SCNP_changed_fraction=(transformed.detach() != logits.detach()).float().mean((-2,-1)).flatten().tolist(),
                         SCNP_logit_displacement=(transformed.detach()-logits.detach()).abs().mean((-2,-1)).flatten().tolist(),
                         SCNP_foreground_fraction=(q >= .5).float().mean((-2,-1)).flatten().tolist())
        return base + strength * extra
