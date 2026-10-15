"""Four frozen arms; read-only derivative observation preserves every base update."""
from ..joint_sensitivity.factorial import Host as FactorialHost, candidates as factorial_candidates
from ..joint_sensitivity.acquisition_weighted import candidates as weighted_candidates
from ..joint_sensitivity.method import flatten
from ..joint_sensitivity.weighted_profile import derivatives


def candidates():
    old = factorial_candidates()
    return [old[0], old[2], weighted_candidates()[-1], old[-1]]


class Host(FactorialHost):
    conditions = staticmethod(candidates)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acquisition_weighted = self.config.get('joint_filter') == 'acquisition_weighted'
        self.observation = {}
        def before(*args):
            self.observation.update(before=flatten(self.native.params), gradient=flatten([p.grad for p in self.native.params]))
        def after(*args):
            self.observation['proposal'] = flatten(self.native.params)-self.observation['before']
        self.handles += [self.native.base.register_step_pre_hook(before), self.native.base.register_step_post_hook(after)]

    def step(self, pixels):
        logits, detail = super().step(pixels)
        actual = flatten(self.native.params)-self.observation['before']
        measured = derivatives(self.observation['gradient'], self.observation['proposal'], actual)
        if not self.strength and measured['dot_change'] != 0:
            raise ValueError('native baseline update changed')
        if self.acquisition_weighted and self.strength:
            joint = detail['diagnostics']['joint']
            if measured['actual_dot'] != joint['loss_actual_dot'] or measured['proposal_dot'] != joint['loss_proposal_dot']:
                raise ValueError('independent derivative mismatch')
        detail['diagnostics']['update_direction'] = measured
        self.observation.clear()
        return logits, detail
