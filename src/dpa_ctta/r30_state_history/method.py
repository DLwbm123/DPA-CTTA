"""Intervene on parameter/Adam history without changing the C objective."""
import copy
import random
import numpy as np
import torch
from ..b1_host import configure
from ..r24_c_context.method import Host as ContextHost
from ..r29_update_strength.method import Host as StrengthHost
from ..r19_model_only.method import clone, equal
from ..r8_ba.rng import capture, restore
from ..hosts.vptta import model_input_from_pixels
from ..integrations.ctta_suite import build_reference_model

ARMS = ('S_SOURCE', 'S_BATCH', 'C_CONT', 'C_EPISODIC', 'C_OPT32', 'C_PARAM32', 'C_BOTH32', 'ANCHOR')
ACTIONS = ('FULL', 'ZERO', 'HALF')


def config(arm):
    return dict(id=arm, family=arm, host='C', params={}, components={})


def norm(values):
    values = list(values)
    return float(torch.stack([x.detach().float().square().sum() for x in values]).sum().sqrt()) if values else 0.


class StateHost(ContextHost):
    def __init__(self, state, arm, seed, identity, device='cuda:0', model=None):
        if arm not in ARMS[2:]:
            raise ValueError('unknown adapting arm')
        self.arm = arm
        super().__init__(state, config('ANCHOR' if arm == 'ANCHOR' else 'C'), seed, identity, device, model)
        self.source_affine = {k: p.detach().clone() for k, p in self.native.model.named_parameters() if k in self.native.names}
        self.resets = 0

    def reset_history(self, parameters, optimizer):
        n = self.native
        if parameters:
            with torch.no_grad():
                for k, p in n.model.named_parameters():
                    if k in self.source_affine:
                        p.copy_(self.source_affine[k])
        if optimizer:
            n.base.state.clear()
            n.steps = 0  # Native steps is Adam-local; self.visits and physical counters remain global.
        self.resets += 1

    def prepare_arrival(self):
        periodic = self.visits > 0 and self.visits % 32 == 0
        if self.arm == 'C_EPISODIC':
            self.reset_history(True, True)
        elif periodic and self.arm in ('C_OPT32', 'C_PARAM32', 'C_BOTH32'):
            self.reset_history(self.arm != 'C_OPT32', self.arm != 'C_PARAM32')

    def diagnostics(self):
        n = self.native
        drift = [p - self.source_affine[k] for k, p in n.model.named_parameters() if k in self.source_affine]
        return dict(parameter_drift=norm(drift), adam_m_norm=norm(s['exp_avg'] for s in n.base.state.values()),
                    adam_v_norm=norm(s['exp_avg_sq'] for s in n.base.state.values()),
                    adam_local_step=n.steps, global_arrivals=self.visits, resets=self.resets)


def readonly(h, x):
    before = h.snapshot()
    aux = copy.deepcopy((h.anchor, None if h.adapter is None else h.adapter.context,
                         h.context_differences, h.used_context_differences, h.diag))
    try:
        h._prepare_prototypes(x)
        return h._readonly(h.native.model, x).cpu()
    finally:
        h.restore(before)
        h.anchor, context, h.context_differences, h.used_context_differences, h.diag = aux
        if h.adapter is not None:
            h.adapter.context = context
        if not equal(before, h.snapshot()):
            raise ValueError('read-only diagnostic changed method state')


class FrozenHost:
    def __init__(self, state, arm, seed, device='cuda:0', model=None):
        if arm not in ARMS[:2]:
            raise ValueError('unknown frozen arm')
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        self.model = build_reference_model('fundus')[0] if model is None else model
        if state is not None:
            self.model.load_state_dict(state)
        self.device = torch.device(device)
        self.model.to(self.device).eval().requires_grad_(False)
        if arm == 'S_BATCH':
            configure(self.model)
            self.model.requires_grad_(False)
        self.visits = 0
        self.initial = {k: p.detach().clone() for k, p in self.model.state_dict().items()}

    def step(self, x):
        with torch.no_grad():
            z = self.model(model_input_from_pixels(x, 'fundus').to(self.device))[0].float().cpu()
        if not torch.isfinite(z).all():
            raise ValueError('nonfinite frozen prediction')
        self.visits += 1
        return z, dict(diagnostics={}, state_committed=False)

    def check_frozen(self, *_):
        if not equal(self.initial, self.model.state_dict()):
            raise ValueError('frozen source state changed')

    def close(self):
        pass


class DelayHost(StrengthHost):
    def first(self, x, action):
        if action not in ACTIONS:
            raise ValueError('unknown first intervention')
        if action == 'ZERO':
            self.action.zero_(); self._prepare_prototypes(x)
            z = self._readonly(self.native.model, x).cpu()
            self.visits += 1
            return z
        self.strength = .5 if action == 'HALF' else 1.
        try:
            return self.step(x)[0]
        finally:
            self.strength = 1.


def set_native_rng(h, value):
    # Future exogenous augmentation draws follow FULL without changing learned state.
    h.native.rng = clone(value)

