import copy
import math
import torch
from ..r36_incremental_modules.method import Host as RetainedHost
from ..r37_cw_orientation.method import candidates as previous_candidates


def candidates():
    c, w, _, lso = copy.deepcopy(previous_candidates())
    return [c, w, lso, dict(lso, id='CW_LSO_TR', bn_trust_radius=.0015)]


class Host(RetainedHost):
    def __init__(self, *args, **kwargs):
        self.trust_before = None
        super().__init__(*args, **kwargs)
        if self.config.get('bn_trust_radius', 0):
            if (self.config['host'] != 'C' or self.config['bn_trust_radius'] <= 0
                    or not math.isfinite(self.config['bn_trust_radius'])):
                raise ValueError('positive C-host BN displacement radius required')
            self.handles.append(self.native.base.register_step_post_hook(self._trust_after_adam))

    def _lr(self, opt, args, kwargs):
        super()._lr(opt, args, kwargs)
        if self.config.get('bn_trust_radius', 0):
            self.trust_before = [p.detach().clone() for p in self.native.params]

    @torch.no_grad()
    def _trust_after_adam(self, opt, args, kwargs):
        if self.trust_before is None:
            raise ValueError('missing pre-Adam BN state')
        deltas = [p-before for p,before in zip(self.native.params,self.trust_before)]
        norm = torch.stack([d.square().sum() for d in deltas]).sum().sqrt()
        if not torch.isfinite(norm):
            raise ValueError('nonfinite BN displacement')
        radius = float(self.config['bn_trust_radius'])
        factor = self._trust_factor(float(norm), radius)
        if factor < 1:
            for p,before,delta in zip(self.native.params,self.trust_before,deltas):
                p.copy_(before+factor*delta)
        self.diag.update(BN_trust_raw_norm=float(norm), BN_trust_scale=factor,
                         BN_trust_radius=radius)
        self.trust_before = None

    def _trust_factor(self, norm, radius):
        return min(1., radius/max(norm, 1e-12))
