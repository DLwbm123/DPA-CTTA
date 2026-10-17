"""Finite entropy-seeking gamma selection; a component study, not full TeSLA."""
import copy
from contextlib import contextmanager
import numpy as np
import torch
from ..r37_cw_orientation.method import Host as RetainedHost, candidates
from ..b1_host import official
from ..hosts.vptta import model_input_from_pixels
from ..r19_model_only.method import equal

ARMS = ('NATIVE', 'RANDOM', 'SELECT')
GAMMAS = (.7, 1., 1/.7)


def configuration(arm):
    if arm not in ARMS: raise ValueError('unregistered arm')
    c = copy.deepcopy(candidates()[3])
    c.update(id='CW_LSO_'+arm, view_arm=arm)
    return c


def entropy(p):
    p = p.clamp(1e-6, 1-1e-6)
    return -(p*p.log()+(1-p)*(1-p).log()).mean()


def scores(probabilities, penalties):
    avg = torch.stack(probabilities).mean(0).clamp(1e-6, 1-1e-6)
    out = []
    for p, penalty in zip(probabilities, penalties):
        p = p.clamp(1e-6, 1-1e-6)
        kl = (avg*(avg/p).log()+(1-avg)*((1-avg)/(1-p)).log()).mean()
        out.append(entropy(p)-penalty+kl)
    return torch.stack(out)


@contextmanager
def replacement_view(chosen):
    module = official(); original = module.augmentation_strong_style
    def replacement(data):
        original(data)  # Preserve the native random stream, even though its view is discarded.
        return chosen.copy()
    module.augmentation_strong_style = replacement
    try: yield
    finally: module.augmentation_strong_style = original


class Host(RetainedHost):
    @torch.no_grad()
    def choose(self, x):
        saved = self.snapshot(); hooks = []; stats = []
        def record(module, inputs):
            stats.append(inputs[0].mean((-2,-1)).detach().clone())
        try:
            for module in self.native.model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    hooks.append(module.register_forward_pre_hook(record))
            clean = model_input_from_pixels(x, 'fundus')
            self._readonly(self.native.model, clean)
            reference = stats[:]; stats.clear()
            views = [clean.pow(g) for g in GAMMAS]
            probs = []; penalties = []
            for view in views:
                probs.append(self._readonly(self.native.model, view).sigmoid())
                if len(stats) != len(reference) or not stats:
                    raise ValueError('normalization hook coverage')
                penalties.append(torch.stack([(a-b).square().mean() for a,b in zip(stats,reference)]).mean())
                stats.clear()
            objective = scores(probs, penalties)
            if not torch.isfinite(objective).all(): raise ValueError('nonfinite selection objective')
            random_index = int(np.random.default_rng(self.seed+710003+self.visits).integers(3))
            index = int(objective.argmax()) if self.config['view_arm']=='SELECT' else random_index
            detail = dict(gamma=GAMMAS[index], selected_index=index, entropy=float(entropy(probs[index])),
                          objective=float(objective[index]), objective_gain=float(objective[index]-objective.mean()),
                          view_change=float((views[index]-clean).abs().mean()), preview_forwards=4)
            chosen = views[index].numpy().copy()
        finally:
            for h in hooks: h.remove()
            self.restore(saved)
        if not equal(self.snapshot(),saved): raise ValueError('selection mutated host state')
        return chosen, detail

    def step(self, x):
        if self.config['view_arm']=='NATIVE': return super().step(x)
        chosen, diagnostic = self.choose(x)
        with replacement_view(chosen): z, detail = super().step(x)
        detail['diagnostics'].update(diagnostic)
        return z, detail


def self_check():
    x = torch.linspace(0,1,33).reshape(1,1,1,33)
    for gamma in GAMMAS:
        y = x.pow(gamma)
        assert torch.isfinite(y).all() and y.min()==0 and y.max()==1
        assert torch.all(y.flatten()[1:] >= y.flatten()[:-1])
        assert torch.equal((y-y.min())/(y.max()-y.min()), y)
        if gamma!=1: assert not torch.allclose(y,x)
    p = [torch.full((1,2,4,4),a) for a in (.2,.5,.8)]
    got = scores(p,[torch.tensor(0.)]*3)
    assert torch.isfinite(got).all()
    assert torch.allclose(got[0],got[2])
    punished = scores(p,[torch.tensor(10.),torch.tensor(0.),torch.tensor(0.)])
    assert punished[0] < got[0]-9.9
    assert [int(np.random.default_rng(71+i).integers(3)) for i in range(10)] == [int(np.random.default_rng(71+i).integers(3)) for i in range(10)]
    print('ENTROPY_VIEW_SELF_CHECK_PASS')
