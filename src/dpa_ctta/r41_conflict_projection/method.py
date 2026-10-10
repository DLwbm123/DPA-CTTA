"""Keep selective trust and remove only a negative historical gradient component."""
import copy
import torch
from ..r39_conflict_trust.method import Host as ConflictHost, candidates as previous_candidates


def candidates():
    c, w, _, gc = copy.deepcopy(previous_candidates())
    return [c, w, gc, dict(gc, id='CW_LSO_GP', bn_conflict_projection=True)]


def project_conflict(gradient, reference):
    denominator = float(reference.square().sum())
    coefficient = min(0., float(torch.dot(gradient, reference))/denominator) if denominator > 1e-12 else 0.
    return gradient-coefficient*reference, coefficient


class Host(ConflictHost):
    def _lr(self, opt, args, kwargs):
        enabled = self.config.get('bn_conflict_projection', False)
        reference = self.gradient_memory.clone() if enabled and self.gradient_memory is not None else None
        super()._lr(opt, args, kwargs)  # EMA and cap trigger continue to use the raw gradient.
        if not enabled:
            return
        gradient = torch.cat([p.grad.detach().flatten() for p in self.native.params])
        projected, coefficient = project_conflict(gradient, reference) if reference is not None else (gradient, 0.)
        if not torch.isfinite(projected).all():
            raise ValueError('nonfinite projected BN gradient')
        if coefficient < 0:
            with torch.no_grad():
                offset = 0
                for p in self.native.params:
                    p.grad.copy_(projected[offset:offset+p.numel()].reshape_as(p))
                    offset += p.numel()
        self.diag.update(BN_projection_active=int(coefficient < 0),
                         BN_projection_removed_norm=float((gradient-projected).norm()),
                         BN_projected_gradient_norm=float(projected.norm()),
                         BN_projected_reference_dot=0. if reference is None else float(torch.dot(projected,reference)))
