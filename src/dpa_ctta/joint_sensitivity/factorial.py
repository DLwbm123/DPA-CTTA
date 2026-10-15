"""CPU-qualified C-background LSO x shared-sensitivity wiring; no run entry point."""
import copy

import torch

from .method import Host as JointHost
from .profile import check_joint_update
from ..r20_model_only_search.method import Host as ConsistencyHost, weights_for
from ..r36_incremental_modules.method import local_structure_loss
from ..r37_cw_orientation.method import Host as BaseHost, candidates as base_candidates


def candidates():
    c, _, _, strong = base_candidates()
    lso = dict(c, id='C_LSO', module='orientation', module_strength=.1)
    joint = dict(joint_basis='shared', joint_strength=1.)
    return [c, lso, dict(c, id='C_JS_SHARED', **joint),
            dict(lso, id='C_LSO_JS_SHARED', **joint), strong]


class Host(JointHost):
    conditions = staticmethod(candidates)

    def __init__(self, state, config, seed, identity, device='cuda:0', model=None):
        expected = {c['id']: c for c in self.conditions()}.get(config.get('id'))
        if expected is None:
            raise ValueError('one of the five fixed factorial conditions required')
        expected = copy.deepcopy(expected)
        # The zero-J limit is a parity check, not another scientific candidate.
        if 'joint_strength' in expected and config.get('joint_strength') == 0:
            expected['joint_strength'] = 0.
        if config != expected or isinstance(config.get('joint_strength'), bool):
            raise ValueError('unchanged factorial parameters required')
        # Keep the historical JointHost CW_LSO-only admission guard intact.
        BaseHost.__init__(self, state, config, seed, identity, device=device, model=model)
        self.strength = float(self.config.get('joint_strength', 0.))
        self.basis = self.config.get('joint_basis', 'acquisition')
        self.descent = False

    def _criterion(self, logits, pseudo):
        if 'W' in self.modules or self.config.get('module') != 'orientation':
            return BaseHost._criterion(self, logits, pseudo)
        base = ConsistencyHost._criterion(self, logits, pseudo)
        params = base_candidates()[3]['params']
        reliability = weights_for(torch.stack(self.weak).sigmoid(),
                                  params['variance_temperature'], params['boundary_boost']).to(logits)
        orientation, magnitude = local_structure_loss(logits, pseudo, reliability)
        strength = self.config['module_strength']
        self.diag.update(LSO=float(orientation.detach()), LSM=float(magnitude.detach()),
                         module_loss=float(orientation.detach()), module_coefficient=strength)
        return base + strength * orientation

    def step(self, pixels):
        logits, detail = super().step(pixels)
        if self.strength:
            check_joint_update(self.config, detail['diagnostics']['joint'], detail['visit'])
        return logits, detail
