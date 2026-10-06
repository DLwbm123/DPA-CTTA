"""Channelwise disagreement masking or detached interval projection."""
import torch
from torch.nn import functional as F
from ..r20_model_only_search.method import Host as Base, weights_for
from ..r7_shared.context import tensor_digest
from ..r8_ba.rng import capture, restore


def conflict_loss(z, q, source, w, mode):
    if mode not in ('MASK', 'INTERVAL'):
        raise ValueError('unknown conflict mode')
    q = q.detach(); source = source.detach()
    conflict = (q >= .5) != (source >= .5)
    p = z.detach().sigmoid()
    projected = torch.maximum(torch.minimum(p, torch.maximum(q, source)), torch.minimum(q, source))
    target = torch.where(conflict, projected, q) if mode == 'INTERVAL' else q
    pixel = F.binary_cross_entropy_with_logits(z, target, reduction='none') * w
    if mode == 'MASK':
        pixel = pixel * ~conflict
    # Fixed full-image denominator: removing pixels must not amplify the rest.
    return pixel.mean(), target, conflict, conflict & ((p < torch.minimum(q, source)) | (p > torch.maximum(q, source)))


class Host(Base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = self.config['id']
        if self.mode not in ('CW', 'MASK', 'INTERVAL'):
            raise ValueError('unknown arm')
        if self.mode != 'CW':
            saved = capture()
            try:
                self.f0 = self._new_model(self.initial, adaptive=False)
                self.f0_digest = tensor_digest(list(self.f0.state_dict().items()))
            finally:
                restore(saved)

    def _prepare_prototypes(self, x):
        self.current_proto = {}
        if self.f0 is not None:
            self.source = self._readonly(self.f0, x).sigmoid()

    def _criterion(self, z, q_native):
        base = super()._criterion(z, q_native)
        self.diag.update(conflict_coverage=0., outside_coverage=0., target_change=0.)
        if self.mode == 'CW':
            return base
        w = weights_for(torch.stack(self.weak).sigmoid(), .1, 2).to(z)
        loss, target, conflict, outside = conflict_loss(z, q_native, self.source.to(z), w, self.mode)
        self.diag.update(conflict_coverage=float(conflict.float().mean()), outside_coverage=float(outside.float().mean()), target_change=float((target-q_native).abs().mean()), pseudo_loss=float(loss.detach()))
        return loss
