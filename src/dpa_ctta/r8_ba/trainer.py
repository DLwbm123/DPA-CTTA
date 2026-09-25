"""R8 source fit: 32 visits, four 8-visit TBPTT updates per episode."""
import copy
import math

import torch

from ..r7_shared.numerics import COUNTS, finite, seg_loss
from ..r7_shared.source import simulate
from .schedule import anchors, episode_roles, episode_styles
from .rng import capture as capture_rng, restore as restore_rng
from .methods import CurrentMLP

from .scope import SCREEN
MAX_STEPS = 4000 if SCREEN else 16000
SAVE_STEPS = (1000, 4000) if SCREEN else (1000, 4000, 8000, 12000, 16000)


def lr_at(step):
    if type(step) is not int or step < 1 or step > MAX_STEPS:
        raise ValueError("R8 learning-rate step")
    if step <= 200:
        return 3e-4 * step / 200
    return 3e-5 + 0.5 * (3e-4 - 3e-5) * (1 + math.cos(math.pi * (step - 200) / 15800))


def detached_state(state):
    return {k: v.detach().clone() if isinstance(v, torch.Tensor) else v for k, v in state.items()}


def method_config(method):
    return (type(method).__name__, method.rank, method.amplitude, method.static,
            getattr(method, "observation", None), getattr(method, "aux_multiplier", None))


class SourceTrainer:
    def __init__(self, segmenter, method, data, oracles, source_seed, binding):
        if source_seed not in range(20260924, 20260929) or not binding:
            raise ValueError("R8 seed/binding")
        oracles.validate(data)
        if oracles.fold != "fit" or oracles.amplitude != segmenter.amplitude or method.amplitude != segmenter.amplitude:
            raise ValueError("R8 fit oracle/amplitude mismatch")
        if not method.observer.fitted:
            raise ValueError("fit-only observer scaler required")
        self.segmenter, self.method, self.data, self.oracles = segmenter, method, data, oracles
        self.source_seed, self.binding, self.steps = source_seed, binding, 0
        self.state = method.initial()
        method.requires_grad_(True)
        method.set_stage("fit")
        self.optimizer = torch.optim.AdamW([p for p in method.parameters() if p.requires_grad],
                                           lr=3e-4, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)
        self.style_bank = anchors("fit")

    def fit_step(self):
        if self.steps >= MAX_STEPS:
            raise ValueError("R8 full fit budget exhausted")
        episode, chunk = divmod(self.steps, 4)
        if chunk == 0:
            self.state = self.method.initial()
        style_ids, curriculum, _ = episode_styles("fit", episode, self.source_seed)
        roles, reused = episode_roles(self.data.folds["fit"], "fit", episode,
                                      style_ids, self.oracles.support_pairs, self.source_seed)
        losses = []
        for visit in range(chunk * 8, (chunk + 1) * 8):
            anchor = style_ids[visit]
            support_name, query_name = roles[visit]
            support = self.data.get(support_name, "fit")
            query = self.data.get(query_name, "fit")
            style = self.style_bank[anchor]
            with torch.no_grad():
                if self.method.group == "A":
                    _, clean_raw, clean_tokens = self.segmenter(support.image, observe=True)
                styled_support = simulate(support.image, style,
                                          f"R8_FIT_SUPPORT|{self.source_seed}|{episode}|{visit}|{support_name}")
                _, raw, tokens = self.segmenter(styled_support, observe=True)
                clean = self.method.observe(clean_raw, clean_tokens) if self.method.group == "A" else None
            state, audit = self.method.update(raw, tokens, self.state)
            self.state = state
            styled_query = simulate(query.image, style,
                                    f"R8_FIT_QUERY|{self.source_seed}|{episode}|{visit}|{query_name}")
            logits = self.segmenter(styled_query, self.method.ambient(state))
            zstar = None if isinstance(self.method, CurrentMLP) else self.method.project(self.oracles.values[:, anchor])
            losses.append(seg_loss(logits, query.label) + self.method.fit_loss(audit, clean, zstar, state))
        loss = torch.stack(losses).mean()
        finite(loss)
        step = self.steps + 1
        for group in self.optimizer.param_groups:
            group["lr"] = lr_at(step)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        COUNTS["source_backward_calls"] += 1
        torch.nn.utils.clip_grad_norm_([p for p in self.method.parameters() if p.requires_grad],
                                       1.0, error_if_nonfinite=True)
        self.optimizer.step()
        COUNTS["source_AdamW"] += 1
        finite(*self.method.parameters())
        self.steps = step
        self.state = detached_state(self.state) if step % 4 else self.method.initial()
        return dict(step=step, loss=float(loss.detach()), curriculum=curriculum,
                    group_reuse=reused, query_visits=8, learning_rate=lr_at(step))

    def snapshot(self):
        if self.steps < 0 or self.steps > MAX_STEPS:
            raise ValueError("R8 fit step outside budget")
        return dict(schema="R8_SOURCE_FIT_SNAPSHOT_V1", binding=self.binding, source_seed=self.source_seed,
                    steps=self.steps, static=self.method.static, amplitude=self.segmenter.amplitude,
                    method_config=method_config(self.method), method_digest=self.method.digest(),
                    method=copy.deepcopy(self.method.state_dict()), optimizer=copy.deepcopy(self.optimizer.state_dict()),
                    episode_state=detached_state(self.state), rng=capture_rng(), physical_counts=dict(COUNTS))

    def restore(self, snapshot):
        if (snapshot.get("schema") != "R8_SOURCE_FIT_SNAPSHOT_V1" or snapshot.get("binding") != self.binding or
                snapshot.get("source_seed") != self.source_seed or snapshot.get("static") != self.method.static or
                snapshot.get("amplitude") != self.segmenter.amplitude or
                snapshot.get("method_config") != method_config(self.method) or
                type(snapshot.get("steps")) is not int or snapshot["steps"] not in range(MAX_STEPS + 1)):
            raise ValueError("R8 source snapshot binding/step")
        if snapshot["episode_state"]["counter"] != 8 * (snapshot["steps"] % 4):
            raise ValueError("R8 source episode position mismatch")
        self.method.load_state_dict(snapshot["method"], strict=True)
        if self.method.digest() != snapshot["method_digest"]:
            raise ValueError("R8 source snapshot method digest")
        self.optimizer.load_state_dict(snapshot["optimizer"])
        self.method.validate_state(snapshot["episode_state"])
        self.state = detached_state(snapshot["episode_state"])
        self.steps = snapshot["steps"]
        restore_rng(snapshot["rng"])
        COUNTS.clear()
        COUNTS.update(snapshot["physical_counts"])
