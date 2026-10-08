"""Same-state probes which leave exactly one ordinary ANCHOR update committed."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from ..r27_nesting_selection.method import Host as Previous
from ..r24_c_context.method import Host as Anchor


def actions_for(seed, order, visit):
    # Separate per-arrival streams keep selection from perturbing candidate noise.
    g = torch.Generator().manual_seed(seed + 710003 + order*100000 + visit*1009)
    return .3*torch.randn((4, 4), generator=g)


def nesting(z):
    p = z.sigmoid()
    coarse = F.interpolate(p, (64, 64), mode='area')
    violation = F.relu(p[:, 1]-p[:, 0])
    return dict(reward64=-float(F.relu(coarse[:, 1]-coarse[:, 0]).mean()),
                reward512=-float(violation.mean()),
                violation_fraction512=float((violation>0).float().mean()))


class Host(Previous):
    def snapshot(self):
        s = super().snapshot()
        s['current_context'] = copy.deepcopy((self.anchor, self.adapter.context))
        return s

    def restore(self, s):
        super().restore(s)
        self.anchor, self.adapter.context = copy.deepcopy(s['current_context'])

    def step(self, x):
        self.action.zero_()
        return Anchor.step(self, x)

    def probe(self, x, actions):
        if actions.shape != (4, 4) or not torch.isfinite(actions).all():
            raise ValueError('four finite four-dimensional actions required')
        before = self.snapshot()
        z, _ = self.step(x)
        committed = self.snapshot()
        logits = [z]
        try:
            for action in actions:
                self.restore(before)
                self.action = action.clone()
                logits.append(Anchor.step(self, x)[0])
        finally:
            self.restore(committed)
        metrics = [nesting(z) for z in logits]
        rewards = np.array([m['reward64'] for m in metrics[1:]])
        return np.stack([(z.numpy()[0]>=0) for z in logits]), dict(
            candidate_metrics=metrics, selected=int(rewards.argmax()),
            tied_max_count=int((rewards==rewards.max()).sum()),
            reward_range=float(np.ptp(rewards)))
