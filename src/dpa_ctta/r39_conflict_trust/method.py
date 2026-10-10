"""Apply the retained trust cap only to gradients opposing preceding BN gradients."""
import copy
import torch
from ..r38_bn_trust.method import Host as TrustHost, candidates as previous_candidates


def candidates():
    c, w, lso, capped = copy.deepcopy(previous_candidates())
    return [c, w, lso, dict(capped, id='CW_LSO_GC', bn_conflict_ema=.9)]


class Host(TrustHost):
    def __init__(self, *args, **kwargs):
        self.gradient_memory = None
        self.gradient_conflict = False
        super().__init__(*args, **kwargs)
        beta = self.config.get('bn_conflict_ema')
        if beta is not None and not 0 < beta < 1:
            raise ValueError('gradient EMA decay must be between zero and one')

    def _lr(self, opt, args, kwargs):
        super()._lr(opt, args, kwargs)
        if not self.config.get('bn_trust_radius', 0):
            return
        gradient = torch.cat([p.grad.detach().flatten() for p in self.native.params])
        if not torch.isfinite(gradient).all():
            raise ValueError('nonfinite BN gradient memory input')
        cosine = 0.
        if self.gradient_memory is not None:
            denominator = float(gradient.norm()*self.gradient_memory.norm())
            if denominator > 1e-12:
                cosine = float(torch.dot(gradient, self.gradient_memory))/denominator
        self.gradient_conflict = cosine < 0
        self.diag.update(BN_gradient_cosine=cosine,
                         BN_gradient_conflict=int(self.gradient_conflict),
                         BN_gradient_memory_ready=int(self.gradient_memory is not None))
        beta = self.config['bn_conflict_ema']
        if self.gradient_memory is None:
            self.gradient_memory = gradient.clone()
        else:
            self.gradient_memory.mul_(beta).add_(gradient, alpha=1-beta)

    def _trust_factor(self, norm, radius):
        return super()._trust_factor(norm, radius) if self.gradient_conflict else 1.

    def snapshot(self):
        saved = super().snapshot()
        if self.config.get('bn_trust_radius', 0):
            saved['gradient_memory'] = (None if self.gradient_memory is None
                                        else self.gradient_memory.detach().cpu().clone())
        return saved

    def restore(self, saved):
        super().restore(saved)
        if self.config.get('bn_trust_radius', 0):
            memory = saved['gradient_memory']
            self.gradient_memory = None if memory is None else memory.to(self.device).clone()
        self.gradient_conflict = False
        self.trust_before = None
