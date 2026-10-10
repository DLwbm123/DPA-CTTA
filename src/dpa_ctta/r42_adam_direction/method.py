import copy
import torch
from ..r41_conflict_projection.method import Host as GradientHost, candidates as previous_candidates, project_conflict


def candidates():
    c, w, _, gp = copy.deepcopy(previous_candidates())
    return [c, w, gp, dict(gp, id='CW_LSO_GDP', bn_adam_direction_projection=True)]


class Host(GradientHost):
    def _lr(self, opt, args, kwargs):
        self.step_reference = self.gradient_memory.clone() if self.gradient_memory is not None else None
        super()._lr(opt, args, kwargs)

    @torch.no_grad()
    def _trust_after_adam(self, opt, args, kwargs):
        before = self.trust_before
        if before is None:
            raise ValueError('missing pre-Adam BN state')
        descent = torch.cat([(a-p).flatten() for a,p in zip(before,self.native.params)])
        reference = self.step_reference
        projected, coefficient = project_conflict(descent, reference) if reference is not None else (descent, 0.)
        if not torch.isfinite(projected).all():
            raise ValueError('nonfinite Adam descent projection')
        active = self.config.get('bn_adam_direction_projection', False) and coefficient < 0
        if active:
            offset = 0
            for p,a in zip(self.native.params,before):
                p.copy_(a-projected[offset:offset+p.numel()].reshape_as(p)); offset += p.numel()
        super()._trust_after_adam(opt, args, kwargs)
        actual = torch.cat([(a-p).flatten() for a,p in zip(before,self.native.params)])
        self.diag.update(BN_adam_direction_active=int(active),
                         BN_adam_raw_descent_norm=float(descent.norm()),
                         BN_adam_removed_norm=float((descent-projected).norm()) if active else 0.,
                         BN_adam_raw_reference_dot=0. if reference is None else float(descent@reference),
                         BN_adam_actual_reference_dot=0. if reference is None else float(actual@reference),
                         BN_adam_actual_descent_norm=float(actual.norm()))
        self.step_reference = None

    def restore(self, saved):
        super().restore(saved)
        self.step_reference = None
