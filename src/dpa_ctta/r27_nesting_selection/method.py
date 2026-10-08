"""Only the selector reward changes; all legacy reward components remain logged."""
from ..r26_residual_action.method import Host as ResidualHost
from ..r25_parameter_policy.method import reward

ARMS = ('ANCHOR', 'RANDOM4', 'GREEDY')


def nesting_reward(*args, **kwargs):
    values = reward(*args, **kwargs)
    values['legacy_retain'] = values['retain']
    values['retain'] = -values['nesting']
    return values


class Host(ResidualHost):
    score_reward = staticmethod(nesting_reward)

    def __init__(self, state, config, seed, identity, device='cuda:0', model=None):
        if config['id'] not in ARMS:
            raise ValueError('unregistered nesting-selection condition')
        super().__init__(state, config, seed, identity, device, model)
