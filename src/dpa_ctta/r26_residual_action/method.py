"""Move four-dimensional actions past the saturated context gate, with fixed bounds."""
import torch
from ..r25_parameter_policy.method import Host as Previous

ARMS = ('C', 'ANCHOR', 'REPEAT5', 'RANDOM4', 'GREEDY', 'PG_RETAIN')


def residual_action(h, output, action):
    if action.shape != (4,) or not torch.isfinite(action).all():
        raise ValueError('four finite residual actions required')
    # Fixed interleaved channel groups; no labels or semantic channel selection.
    factors = (.75 * action.to(output).tanh()).exp()
    scale = factors[torch.arange(output.shape[1], device=output.device) % 4][None, :, None, None]
    # This expression preserves the original output exactly at zero action.
    return output + (scale - 1) * (output - h)


class Host(Previous):
    def __init__(self, state, config, seed, identity, device='cuda:0', model=None):
        if config['id'] not in ARMS:
            raise ValueError('unregistered residual-action condition')
        super().__init__(state, config, seed, identity, device, model)
        if self.policy is not None:
            self.handles.pop().remove()  # R25's final handle is the gate-logit action.
            self.handles.append(self.adapter.register_forward_hook(self._apply_action))

    def _apply_action(self, module, args, output):
        result = residual_action(args[0], output, self.action)
        module.last['applied_relative_rms'] = float(((result-args[0]).square().mean() /
            (args[0].square().mean()+1e-12)).sqrt().detach())
        return result

    def step(self, x):
        logits, trace = super().step(x)
        trace['diagnostics']['residual_action_enabled'] = self.policy is not None
        trace['diagnostics']['residual_action_factors'] = (.75*self.action.tanh()).exp().tolist()
        return logits, trace
