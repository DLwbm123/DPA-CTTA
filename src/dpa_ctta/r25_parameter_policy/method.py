"""Conditional parameter exploration, detached structural rewards and retention.

These are mechanism transfers, not reproductions of LLM 3PO/RaPO or TopoOT.
The learned object is a contextual Gaussian policy over four adapter gate logits.
"""
import base64
import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy import ndimage
from ..r24_c_context.method import Host as ContextHost
from ..r19_model_only.method import clone

ARMS = ('C', 'ANCHOR', 'REPEAT5', 'RANDOM4', 'GREEDY', 'PG_CONS',
        'PG_STRUCT', 'PG_RETAIN', 'PG_COV', 'RN_DPO')
POLICY_ARMS = ('PG_CONS', 'PG_STRUCT', 'PG_RETAIN', 'PG_COV')


def small(p):
    return F.interpolate(p.detach().float().cpu(), (64, 64), mode='area')


def balanced_distance(a, b, reference):
    foreground = reference >= .5
    error = (a - b).abs()
    terms = [error[m].mean() for m in (foreground, ~foreground) if m.any()]
    return float(torch.stack(terms).mean())


def components(mask, confidence, holes=False):
    labels, count = ndimage.label(~mask if holes else mask)
    ids = np.arange(1, count + 1)
    if holes:
        exterior = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
        ids = ids[~np.isin(ids, exterior)]
    if not len(ids):
        return np.empty((0, 4), dtype=np.float64)
    areas = ndimage.sum(np.ones_like(mask), labels, ids)
    keep = np.argsort(-areas, kind='stable')[:8]
    keep = keep[areas[keep] >= 2]
    ids, areas = ids[keep], areas[keep]
    if not len(ids):
        return np.empty((0, 4), dtype=np.float64)
    centers = np.asarray(ndimage.center_of_mass(np.ones_like(mask), labels, ids)) / np.asarray(mask.shape)
    certainty = ndimage.mean(1 - confidence if holes else confidence, labels, ids)
    return np.column_stack((centers, np.sqrt(areas / mask.size), certainty))


def transport(a, b):
    if not len(a) or not len(b):
        return float(bool(len(a) or len(b)))
    cost = np.minimum(np.sqrt(((a[:, None] - b[None]) ** 2).sum(-1)), 2.)
    kernel = np.exp(-cost / .1).clip(1e-12)
    mass_a, mass_b = np.full(len(a), 1 / len(a)), np.full(len(b), 1 / len(b))
    u, v = np.ones(len(a)), np.ones(len(b))
    for _ in range(32):
        u = mass_a / (kernel @ v + 1e-12)
        v = mass_b / (kernel.T @ u + 1e-12)
    return float((u[:, None] * kernel * v[None] * cost).sum() + abs(len(a)-len(b))/max(len(a), len(b)))


def structure(p, view):
    """Threshold-component OT chains, including holes, on an explicit 64px grid.

    This does not compute persistence diagrams; it is a TopoOT-inspired transfer.
    """
    p, view = p.numpy()[0], view.numpy()[0]
    cross, chains = [], []
    for channel in range(2):
        for holes in (False, True):
            left, right = [], []
            for threshold in (.2, .35, .5, .65, .8):
                left.append(components(p[channel] >= threshold, p[channel], holes))
                right.append(components(view[channel] >= threshold, view[channel], holes))
                cross.append(transport(left[-1], right[-1]))
            for sequence in (left, right):
                chains.extend(transport(a, b) for a, b in zip(sequence, sequence[1:]))
    return float(np.mean(cross) + .25 * np.mean(chains))


def reward(p, view, reference, past=None, past_reference=None):
    consistency = balanced_distance(p, view, reference)
    # A relative soft-area penalty reduces the trivial empty/filled-mask solution.
    # It is a conservative current-model reference, not a correctness guarantee.
    extent = float((torch.log((p.mean((-2, -1)) + 1e-3) /
                             (reference.mean((-2, -1)) + 1e-3))).abs().clamp(max=2).mean())
    topology = structure(p, view)
    nesting = float(F.relu(p[:, 1] - p[:, 0]).mean())
    retention = 0. if past is None else balanced_distance(past, past_reference, past_reference)
    cons = -consistency - .05 * extent
    structural = cons - .2 * topology - .1 * nesting
    return dict(cons=cons, structural=structural, retain=structural-.1*retention,
                consistency=consistency, extent=extent, topology=topology,
                nesting=nesting, retention=retention)


class Policy(nn.Module):
    def __init__(self, context_size, seed, correlated=False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.layer = nn.Linear(context_size + 5, 14 if correlated else 8)
            nn.init.zeros_(self.layer.weight); nn.init.zeros_(self.layer.bias)
        self.correlated = correlated

    def distribution(self, context, history):
        values = self.layer(torch.cat((context.detach().cpu().flatten(), history)))
        mean = .5 * values[:4].tanh()
        diagonal = .1 + .4 * values[4:8].sigmoid()
        scale = torch.diag(diagonal)
        if self.correlated:
            i, j = torch.tril_indices(4, 4, -1)
            scale = scale.index_put((i, j), .1 * values[8:].tanh())
        return torch.distributions.MultivariateNormal(mean, scale_tril=scale)


def reinforce(policy, optimizer, context, history, actions, rewards, scale):
    distribution = policy.distribution(context, history)
    values = torch.tensor(rewards, dtype=torch.float32)
    # Leave-one-out group baseline; the EMA denominator comes from earlier arrivals.
    advantage = (values-values.mean()) * len(values)/(len(values)-1) / max(.01, scale)
    prior = torch.distributions.MultivariateNormal(torch.zeros(4), .09*torch.eye(4))
    kl = torch.distributions.kl_divergence(distribution, prior)
    loss = -(distribution.log_prob(actions.detach()) * advantage.detach()).mean() + .01*kl
    optimizer.zero_grad(); loss.backward()
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in policy.parameters()):
        raise ValueError('invalid policy gradient')
    norm = float(nn.utils.clip_grad_norm_(policy.parameters(), 1.))
    optimizer.step()
    return .99*scale+.01*float(values.std(unbiased=False)), dict(
        policy_loss=float(loss.detach()), policy_gradient_norm=norm, policy_KL=float(kl.detach()),
        advantage_std=float(advantage.std(unbiased=False)), reward_std=float(values.std(unbiased=False)))


def region_preference(logits, reference, winner, loser):
    region = winner != loser
    if not region.any():
        return logits.sum()*0
    # Bernoulli log-likelihood differences cancel to (winner-loser)*logits.
    margin = ((winner.float()-loser.float()) * (logits-reference.detach()))[region].mean()
    return -F.logsigmoid(.1*margin)


class Host(ContextHost):
    def __init__(self, state, config, seed, identity, device='cuda:0', model=None):
        self.arm = config['id']
        if self.arm not in ARMS:
            raise ValueError('unregistered arm')
        base_config = dict(config, id='C' if self.arm == 'C' else 'ANCHOR')
        super().__init__(state, base_config, seed, identity, device, model)
        self.action = torch.zeros(4)
        self.policy = None
        self.memory = []
        self.history = torch.zeros(5)
        self.reward_scale = .05
        self.generator = torch.Generator().manual_seed(seed + 710003)
        self.preference = None
        if self.arm not in ('C', 'ANCHOR', 'REPEAT5'):
            self.policy = Policy(self.adapter.gate.in_features, seed+910009, self.arm == 'PG_COV')
            self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4)
            self.handles.append(self.adapter.gate.register_forward_hook(
                lambda m, a, out: out + .75*self.action.to(out).tanh()[None]))

    def _criterion(self, logits, target):
        loss = super()._criterion(logits, target)
        if self.preference is not None:
            reference, winner, loser = self.preference
            extra = region_preference(logits, reference.to(logits), winner.to(logits.device), loser.to(logits.device))
            self.diag['preference_loss'] = float(extra.detach())
            self.diag['preference_fraction'] = float((winner != loser).float().mean())
            loss = loss + .1*extra
        return loss

    def _probe(self, x, reset_context=False):
        anchor, context = self.anchor, self.adapter.context
        if reset_context:
            self.anchor = None
        try:
            result = self._readonly(self.native.model, x).sigmoid().cpu()
            descriptor = self.anchor.detach().cpu().clone()
            return result, descriptor
        finally:
            self.anchor, self.adapter.context = anchor, context

    def step(self, x):
        if self.arm in ('C', 'ANCHOR'):
            return super().step(x)
        if self.arm == 'REPEAT5':
            visit = self.visits
            for _ in range(5):
                z, trace = super().step(x)
            self.visits = visit+1
            trace['visit'] = self.visits
            trace['diagnostics']['committed_updates'] = 5
            return z, trace
        self.preference = None
        self.action.zero_()
        reference, context = self._probe(x, reset_context=True)
        reference_small = small(reference)
        start = super().snapshot()
        distribution = self.policy.distribution(context, self.history)
        with torch.no_grad():
            actions = distribution.loc + torch.randn((4, 4), generator=self.generator) @ distribution.scale_tril.T
        results, statistics = [], []
        previous = self.memory[self.visits % len(self.memory)] if self.memory else None
        for action in actions:
            super().restore(start)
            self.action = action.clone()
            logits, trace = super().step(x)
            prediction = logits.sigmoid()
            # Independent deterministic photometric probe, never used for the C gradient.
            transformed, _ = self._probe(x.clamp(0, 1).pow(1.2))
            past = None if previous is None else small(self._probe(previous['image'], reset_context=True)[0])
            statistic = reward(small(prediction), small(transformed), reference_small,
                               past, None if previous is None else previous['prediction'])
            statistics.append(statistic)
            results.append((super().snapshot(), logits, trace))
        kind = 'cons' if self.arm == 'PG_CONS' else ('structural' if self.arm == 'PG_STRUCT' else 'retain')
        rewards = [s[kind] for s in statistics]
        selected = int(torch.randint(4, (), generator=self.generator)) if self.arm == 'RANDOM4' else int(np.argmax(rewards))
        selected_state, logits, trace = results[selected]
        super().restore(selected_state)
        self.action = actions[selected].clone()
        if self.arm == 'RN_DPO':
            worst = int(np.argmin(rewards))
            # Replay from the identical pre-arrival state with one added region loss.
            super().restore(start)
            self.preference = (torch.logit(reference.clamp(1e-6, 1-1e-6)),
                               results[selected][1] >= 0, results[worst][1] >= 0)
            logits, trace = super().step(x)
            self.preference = None
        policy_diag = {}
        if self.arm in POLICY_ARMS:
            self.reward_scale, policy_diag = reinforce(self.policy, self.policy_optimizer, context,
                self.history, actions, rewards, self.reward_scale)
        self.history = torch.cat((actions[selected].tanh(), torch.tensor([rewards[selected]])))
        if (self.visits-1) % 16 == 0:
            self.memory.append(dict(image=x.detach().cpu().clone(), prediction=small(logits.sigmoid())))
            self.memory = self.memory[-4:]
        trace['diagnostics'].update(policy_diag, selected=selected, rewards=rewards,
            reward_components=statistics, reward_scale=self.reward_scale,
            proposal_spread=float(actions.std(0, unbiased=False).mean()),
            candidate_mask_disagreement=float(torch.stack([(a[1]>=0).float() for a in results]).var(0, unbiased=False).mean()),
            memory_size=len(self.memory), memory_max_lag=64, committed_updates=1,
            proposal_updates=4, reference_forwards=1, reward_probe_forwards=4+(4 if previous is not None else 0),
            replay_updates=int(self.arm == 'RN_DPO'), is_contextual_bandit=self.arm in POLICY_ARMS)
        trace['proposal_masks_b64'] = [base64.b64encode(np.packbits((small(a[1].sigmoid()) >= .5).numpy()).tobytes()).decode('ascii') for a in results]
        return logits, trace

    def snapshot(self):
        result = super().snapshot()
        result['policy_state'] = dict(policy=None if self.policy is None else clone(self.policy.state_dict()),
            optimizer=None if self.policy is None else clone(self.policy_optimizer.state_dict()),
            memory=clone(self.memory), history=self.history.clone(), reward_scale=self.reward_scale,
            generator=self.generator.get_state().clone(), action=self.action.clone())
        return result

    def restore(self, state):
        super().restore(state)
        p = state['policy_state']
        if self.policy is not None:
            self.policy.load_state_dict(p['policy']); self.policy_optimizer.load_state_dict(copy.deepcopy(p['optimizer']))
        self.memory = clone(p['memory']); self.history = p['history'].clone()
        self.reward_scale = p['reward_scale']; self.generator.set_state(p['generator']); self.action = p['action'].clone()
        self.preference = None
