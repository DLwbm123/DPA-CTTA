"""R8 A/B parameterizations; the task graph and runner are separate."""
import math

import torch
from torch import nn

from ..r7_a_psf import gaussian_filter
from ..r7_b_rca import energy
from ..r7_shared.host import Method
from ..r7_shared.network import Segmenter
from ..r7_shared.numerics import (COUNTS, attention, coherence, gaussian_nll, mlp,
                                  normalize, stable, temperature, temperature_initial, variance)


def film(h, v, amplitude):
    if v.shape != (512,) or amplitude not in (0.1, 0.3):
        raise ValueError("R8 FiLM shape/amplitude")
    gamma, beta = v[:256].to(h), v[256:].to(h)
    return h + torch.expm1(amplitude * gamma.tanh())[None, :, None, None] * h + amplitude * beta.tanh()[None, :, None, None]


def correct(h, o, prior, kappa, steps=5):
    if steps not in (5, 20):
        raise ValueError("R8 ISTA steps")
    h, o, prior = (x.double() for x in (h, o, prior))
    kappa = torch.as_tensor(kappa, dtype=torch.float64)
    if h.shape != (64, len(prior)) or o.shape != (64,) or not torch.isfinite(kappa) or kappa <= 0:
        raise ValueError("R8 ISTA input")
    COUNTS["spectral_norms"] += 1
    eta = 1 / (torch.linalg.matrix_norm(h, 2).square() / kappa.square() + 0.1)
    delta = torch.zeros_like(prior)
    history = [energy(delta, h, o, prior, kappa)]
    for _ in range(steps):
        gradient = h.T @ (h @ (prior + delta) - o) / kappa.square() + 0.1 * delta
        candidate = delta - eta * gradient
        delta = candidate.sign() * (candidate.abs() - 0.01 * eta).clamp_min(0)
        COUNTS["ISTA_iterations"] += 1
        history.append(energy(delta, h, o, prior, kappa))
    return prior + delta, dict(delta=delta, eta=eta, energy=torch.stack(history))


class R8Segmenter(Segmenter):
    def __init__(self, model, amplitude, device="cpu"):
        if amplitude not in (0.1, 0.3):
            raise ValueError("R8 amplitude")
        self.amplitude = amplitude
        super().__init__(model, device)

    def _up1(self, module, inputs, h):
        if h.shape[1] != 256:
            raise ValueError("up1 channels")
        return film(h, self.v[:512], self.amplitude)

    def _up3(self, module, inputs, h):
        # Keep R7's transient token capture, then use the R8 amplitude.
        if h.shape[1] != 256:
            raise ValueError("up3 channels")
        if self.capture:
            from torch.nn import functional as F
            norm = (h - h.mean((-2, -1), keepdim=True)) / h.std((-2, -1), unbiased=False, keepdim=True).clamp_min(1e-6)
            e = F.adaptive_avg_pool2d(norm, (8, 8))[0].flatten(1).T @ self.projection
            self.cache["tokens"] = F.layer_norm(e, (64,), eps=1e-6)
        return film(h, self.v[512:], self.amplitude)


class R8A(Method):
    group = "A"

    def __init__(self, basis, amplitude, static=False):
        self.rank = basis.shape[1]
        if self.rank not in (16, 32) or amplitude not in (0.1, 0.3):
            raise ValueError("R8 A rank/amplitude")
        super().__init__(basis, static)
        self.amplitude = amplitude
        self.style = nn.Parameter(torch.randn(8, 32) * 0.1)
        self.content = nn.Parameter(torch.randn(32, 64) * 0.1)
        self.head = mlp(128, 64, 2 * self.rank)
        self.W = nn.Parameter(torch.eye(self.rank) * (0.9 / 0.95))
        u = (0.01 - 1e-4) / (1 - 1e-4)
        self.qraw = nn.Parameter(torch.full((self.rank,), math.log(u / (1 - u))))
        self.cal_raw = nn.Parameter(temperature_initial(), requires_grad=False)

    def initial(self):
        return dict(m=torch.zeros(self.rank, dtype=torch.float64),
                    P=torch.eye(self.rank, dtype=torch.float64), counter=0)

    def set_stage(self, stage):
        super().set_stage(stage)
        self.cal_raw.requires_grad_(stage == "cal")

    def observe(self, raw, tokens):
        d = self.observer(raw)
        ks = attention(d, self.style)
        dh = ks @ self.style
        kc = attention(tokens, self.content)
        c = kc @ self.content
        COUNTS["method_MLP"] += 1
        value = self.head(torch.cat((d, dh, c.mean(0))))
        r = variance(value[self.rank:]).double()
        if self.stage != "fit":
            r = r * temperature(self.cal_raw).double().square()
        return dict(d=d, dhat=dh, k=kc, c=c, E=tokens, o=value[:self.rank].double(), R=r)

    def update(self, raw, tokens, state, ablation=None):
        self.validate_state(state)
        old_counter = state["counter"]
        state = self.initial() if self.static or ablation == "RESET_HISTORY" else state
        a = self.observe(raw, tokens)
        r = a["R"]
        if ablation == "ISOTROPIC_R":
            r = r.mean().expand_as(r)
        elif ablation not in (None, "RESET_HISTORY"):
            raise ValueError("R8 A ablation")
        m, p = gaussian_filter(state["m"], state["P"], stable(self.W.double()),
                               torch.diag(variance(self.qraw.double(), 1.)), a["o"], torch.diag(r))
        return dict(m=m, P=p, counter=old_counter + 1), a

    def fit_loss(self, a, clean, zstar, state):
        content = ((a["c"] - clean["E"].detach()).square().mean() +
                   (a["k"] - clean["k"].detach()).square().mean()) / 2
        return (0.1 * (a["o"] - zstar).square().mean() + 0.01 * gaussian_nll(zstar, state["m"], state["P"]) +
                0.05 * (a["dhat"] - a["d"].detach()).square().mean() + 0.05 * content)

    def cal_loss(self, a, zstar, state):
        return gaussian_nll(zstar, state["m"], state["P"])


class R8B(Method):
    group = "B"

    def __init__(self, basis, amplitude, observer="global", aux_multiplier=1., static=False):
        self.rank = basis.shape[1]
        if self.rank not in (32, 64) or amplitude not in (0.1, 0.3) or observer not in ("global", "current_tokens") or aux_multiplier not in (1., 0.1):
            raise ValueError("R8 B config")
        super().__init__(basis, static)
        if not torch.allclose(self.basis.T @ self.basis, torch.eye(self.rank, dtype=torch.float64), atol=1e-8, rtol=1e-8):
            raise ValueError("B basis orthonormal")
        self.amplitude, self.observation, self.aux_multiplier = amplitude, observer, aux_multiplier
        self.W = nn.Parameter(0.9 * torch.eye(self.rank))
        self.G = nn.Parameter(torch.randn(self.rank, 32) * 0.01)
        self.bias = mlp(32 if observer == "global" else 160, 64, self.rank)
        self.head = mlp(32 if observer == "global" else 160, 64, 64)
        self.Hraw = nn.Parameter(torch.randn(64, self.rank))
        self.cal_raw = nn.Parameter(temperature_initial(), requires_grad=False)
        self.register_buffer("frozen_eta", 1 / (torch.linalg.matrix_norm(normalize(self.Hraw.detach().double(), 0), 2).square() + 0.1))

    def initial(self):
        return dict(z=torch.zeros(self.rank, dtype=torch.float64),
                    d=torch.zeros(32, dtype=torch.float64), counter=0)

    def set_stage(self, stage):
        super().set_stage(stage)
        self.cal_raw.requires_grad_(stage == "cal")
        if stage == "online":
            with torch.no_grad():
                self.frozen_eta.copy_(1 / (torch.linalg.matrix_norm(normalize(self.Hraw.double(), 0), 2).square() /
                                           temperature(self.cal_raw).double().square() + 0.1))

    def observe(self, raw, tokens):
        d = self.observer(raw)
        u = d if self.observation == "global" else torch.cat((d, tokens.mean(0), tokens.std(0, unbiased=False)))
        COUNTS["method_MLP"] += 2
        return dict(d=d.double(), o=self.head(u).double(), bias=self.bias(u).double(),
                    H=normalize(self.Hraw.double(), 0))

    def update(self, raw, tokens, state, ablation=None):
        self.validate_state(state)
        old_counter = state["counter"]
        state = self.initial() if self.static or ablation == "RESET_HISTORY" else state
        a = self.observe(raw, tokens)
        d = a["d"]
        diff = torch.zeros_like(d) if state["counter"] == 0 else d - state["d"]
        prior = stable(self.W.double()) @ state["z"] + a["bias"] + stable(self.G.double(), 1.) @ diff
        kappa = 1. if self.stage == "fit" else temperature(self.cal_raw).double()
        if ablation == "PRED_ONLY":
            z = prior
            a.update(delta=torch.zeros_like(z), eta=None, energy=None)
        elif ablation in (None, "RESET_HISTORY", "ISTA_20"):
            z, diag = correct(a["H"], a["o"], prior, kappa, 20 if ablation == "ISTA_20" else 5)
            a.update(diag)
            if self.stage == "online" and not torch.allclose(a["eta"], self.frozen_eta, rtol=1e-10, atol=1e-12):
                raise ValueError("frozen ISTA step mismatch")
        else:
            raise ValueError("R8 B ablation")
        a.update(prior=prior, difference=diff)
        return dict(z=z, d=d, counter=old_counter + 1), a

    def fit_loss(self, a, clean, zstar, state):
        return self.aux_multiplier * (0.1 * (state["z"] - zstar).square().mean() +
                                      0.1 * (a["o"] - a["H"] @ zstar).square().mean() +
                                      0.001 * coherence(a["H"].T))

    def cal_loss(self, a, zstar, state):
        return (state["z"] - zstar).square().mean()


class CurrentMLP(Method):
    """Matched B basis and observation with no history or proxy objective."""
    group = "B"

    def __init__(self, basis, amplitude, observer):
        self.rank = basis.shape[1]
        if self.rank not in (32, 64) or amplitude not in (0.1, 0.3) or observer not in ("global", "current_tokens"):
            raise ValueError("R8 MLP config")
        super().__init__(basis, static=True)
        self.amplitude, self.observation = amplitude, observer
        self.head = mlp(32 if observer == "global" else 160, 64, self.rank)

    def initial(self):
        return dict(z=torch.zeros(self.rank, dtype=torch.float64),
                    d=torch.zeros(32, dtype=torch.float64), counter=0)

    def update(self, raw, tokens, state, ablation=None):
        if ablation is not None:
            raise ValueError("MLP has no ablation")
        self.validate_state(state)
        d = self.observer(raw)
        u = d if self.observation == "global" else torch.cat((d, tokens.mean(0), tokens.std(0, unbiased=False)))
        COUNTS["method_MLP"] += 1
        z = self.head(u).double()
        return dict(z=z, d=d.double(), counter=state["counter"] + 1), dict(d=d)

    def fit_loss(self, a, clean, zstar, state):
        return state["z"].new_zeros(())

    def cal_loss(self, a, zstar, state):
        raise ValueError("CURRENT_MLP has no calibration")


def build(config, basis, static=False, seed=20260924):
    """Build only an explicitly listed R8 config; never infer a new candidate."""
    if config["route"] == "A":
        cls = R8A
        kwargs = {}
    elif config["route"] == "B":
        cls = R8B
        kwargs = dict(observer=config["observer"], aux_multiplier=config["aux_multiplier"])
    else:
        raise ValueError("R8 route")
    if basis.shape != (1024, config["rank"]):
        raise ValueError("R8 basis rank/config mismatch")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        return cls(basis, config["film_amplitude"], static=static, **kwargs)
