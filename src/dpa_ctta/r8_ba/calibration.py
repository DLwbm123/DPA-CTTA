"""Independent 1024-step R8 source calibration and source-val selection."""
import copy
import math
import random

import numpy as np
import torch

from ..r7_shared.numerics import COUNTS, finite, temperature_initial
from ..r7_shared.source import RANGES, simulate
from .schedule import CURRICULA, anchors, episode_roles, episode_styles, generator
from .trainer import SAVE_STEPS


def calibration_styles(step, bank):
    if step not in range(1024) or bank.shape != (128, 9):
        raise ValueError("R8 calibration step/bank")
    rng = generator("calibration", step)
    a, b = torch.randperm(128, generator=rng)[:2].tolist()
    if step % 3 == 0:
        return (a, a, b, b)
    if step % 3 == 2:
        return (a, b, a, b)
    scale = torch.tensor([high - low for low, high in RANGES])
    selected = [a]
    for _ in range(3):
        last = selected[-1]
        selected.append(min((i for i in range(128) if i not in selected),
                            key=lambda i: (float(((((bank[i] - bank[last]) / scale) ** 2).sum())), i)))
    return tuple(selected)


class Calibrator:
    def __init__(self, segmenter, trained_method, data, cal_oracles, binding):
        if not binding:
            raise ValueError("R8 calibration binding")
        cal_oracles.validate(data)
        if cal_oracles.fold != "cal" or cal_oracles.amplitude != segmenter.amplitude:
            raise ValueError("R8 calibration source/amplitude")
        if trained_method.amplitude != segmenter.amplitude or not hasattr(trained_method, "cal_raw"):
            raise ValueError("R8 calibratable A/B only")
        self.segmenter, self.data, self.oracles = segmenter, data, cal_oracles
        self.binding = binding
        self.method = copy.deepcopy(trained_method)
        self.method.requires_grad_(False)
        with torch.no_grad():
            self.method.cal_raw.copy_(temperature_initial())
        self.method.set_stage("cal")
        self.optimizer = torch.optim.Adam([self.method.cal_raw], lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.bank, self.steps = anchors("cal"), 0

    def step(self):
        if self.steps >= 1024:
            raise ValueError("R8 calibration budget exhausted")
        ids = calibration_styles(self.steps, self.bank)
        names = sorted(self.data.folds["cal"])
        perm = torch.randperm(len(names), generator=generator("calibration_roles", self.steps)).tolist()
        supports = [names[i] for i in perm[:4]]
        state, losses = self.method.initial(), []
        self.optimizer.zero_grad(set_to_none=True)
        for visit, (anchor, name) in enumerate(zip(ids, supports)):
            row = self.data.get(name, "cal")
            image = simulate(row.image, self.bank[anchor], f"R8_CAL|{self.steps}|{visit}|{name}")
            with torch.no_grad():
                _, raw, tokens = self.segmenter(image, observe=True)
            state, audit = self.method.update(raw, tokens, state)
            zstar = self.method.project(self.oracles.values[:, anchor])
            losses.append(self.method.cal_loss(audit, zstar, state))
        loss = torch.stack(losses).mean()
        finite(loss)
        loss.backward()
        COUNTS["calibration_backward_calls"] += 1
        finite(self.method.cal_raw.grad)
        self.optimizer.step()
        COUNTS["calibration_Adam"] += 1
        finite(self.method.cal_raw)
        self.steps += 1
        return dict(step=self.steps, loss=float(loss.detach()))

    def run(self):
        while self.steps < 1024:
            self.step()
        self.method.freeze()
        return self.method

    def snapshot(self):
        if self.method.stage != "cal":
            raise ValueError("R8 calibration already frozen")
        return dict(schema="R8_SOURCE_CAL_SNAPSHOT_V1", binding=self.binding,
                    amplitude=self.segmenter.amplitude, steps=self.steps,
                    method=copy.deepcopy(self.method.state_dict()), optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    torch_rng=torch.random.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(), physical_counts=dict(COUNTS))

    def restore(self, snapshot):
        if (snapshot.get("schema") != "R8_SOURCE_CAL_SNAPSHOT_V1" or snapshot.get("binding") != self.binding or
                snapshot.get("amplitude") != self.segmenter.amplitude or self.method.stage != "cal" or
                type(snapshot.get("steps")) is not int or snapshot["steps"] not in range(1025)):
            raise ValueError("R8 calibration snapshot binding/step")
        self.method.load_state_dict(snapshot["method"], strict=True)
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.steps = snapshot["steps"]
        torch.random.set_rng_state(snapshot["torch_rng"])
        if snapshot["cuda_rng"] is not None:
            if not torch.cuda.is_available():
                raise ValueError("R8 CUDA RNG unavailable")
            torch.cuda.set_rng_state_all(snapshot["cuda_rng"])
        np.random.set_state(snapshot["numpy_rng"])
        random.setstate(snapshot["python_rng"])
        COUNTS.clear()
        COUNTS.update(snapshot["physical_counts"])


@torch.no_grad()
def validate_episode(segmenter, method, data, val_oracles, episode):
    val_oracles.validate(data)
    if val_oracles.fold != "val" or val_oracles.amplitude != segmenter.amplitude or method.stage != "online":
        raise ValueError("R8 validation source/amplitude/stage")
    ids, curriculum, severity = episode_styles("val", episode)
    roles, reuse = episode_roles(data.folds["val"], "val", episode, ids, val_oracles.support_pairs)
    bank = anchors("val")
    state, dice, proxy = method.initial(), [], []
    for visit, (anchor, (support_name, query_name)) in enumerate(zip(ids, roles)):
        support, query = data.get(support_name, "val"), data.get(query_name, "val")
        style = bank[anchor]
        image = simulate(support.image, style, f"R8_VAL_SUPPORT|{episode}|{visit}|{support_name}")
        _, raw, tokens = segmenter(image, observe=True)
        state, _ = method.update(raw, tokens, state)
        query_image = simulate(query.image, style, f"R8_VAL_QUERY|{episode}|{visit}|{query_name}")
        probability = segmenter(query_image, method.ambient(state)).sigmoid()
        channel = (2 * (probability * query.label).sum((0, 2, 3)) + 1e-6) / (
            probability.sum((0, 2, 3)) + query.label.sum((0, 2, 3)) + 1e-6)
        dice.append(channel.mean().item())
        zstar = method.project(val_oracles.values[:, anchor])
        proxy.append((method.code(state) - zstar).square().mean().item())
    return dict(episode=episode, curriculum=curriculum, severity=severity,
                soft_Dice=sum(dice) / 32, proxy_MSE=sum(proxy) / 32, group_reuse=reuse)


def validate_all(segmenter, method, data, val_oracles):
    return [validate_episode(segmenter, method, data, val_oracles, episode) for episode in range(64)]


def score_validation(two_seed_rows):
    if len(two_seed_rows) != 2 or any(len(rows) != 64 for rows in two_seed_rows):
        raise ValueError("R8 source selection needs two complete 64-episode seeds")
    for rows in two_seed_rows:
        if {row["episode"] for row in rows} != set(range(64)):
            raise ValueError("R8 validation episode identity")
        if any(sum(row["curriculum"] == mode for row in rows) != 16 for mode in CURRICULA):
            raise ValueError("R8 validation curriculum coverage")
    means = []
    for mode in CURRICULA:
        values = [row["soft_Dice"] for rows in two_seed_rows for row in rows if row["curriculum"] == mode]
        if len(values) != 32 or not all(math.isfinite(x) for x in values):
            raise ValueError("R8 validation missing/invalid curriculum")
        means.append(sum(values) / 32)
    return 0.5 * sum(means) / 4 + 0.5 * min(means)


def select_checkpoint(results_by_step):
    if set(results_by_step) != set(SAVE_STEPS):
        raise ValueError("all five source snapshots required")
    scores = {step: score_validation(rows) for step, rows in results_by_step.items()}
    best = max(scores.values())
    return min(step for step, score in scores.items() if best - score <= 1e-6), scores


def select_family(full_config_scores, recorded_cost):
    if not full_config_scores:
        return None
    if set(full_config_scores) != set(recorded_cost):
        raise ValueError("R8 family score/cost mismatch")
    for value in (*full_config_scores.values(), *recorded_cost.values()):
        if not math.isfinite(value):
            raise ValueError("R8 invalid source score/cost")
    best = max(full_config_scores.values())
    eligible = [name for name, score in full_config_scores.items() if best - score <= 1e-6]
    return min(eligible, key=lambda name: (recorded_cost[name], name))
