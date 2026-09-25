"""Four-visit source-cal LR evaluation with disjoint, gradient-free queries."""
import math

import torch

from ..r7_shared.source import simulate
from .calibration import calibration_styles
from .gradient import LR, STEPS
from .schedule import anchors, episode_roles, generator


def episode(host, data, oracles, index, source_seed):
    if (oracles.fold != "cal" or not 0 <= index < 64 or host.visits != 0 or
            source_seed not in (20260924, 20260925)):
        raise ValueError("R8 gradient source-cal episode identity")
    oracles.validate(data)
    bank = anchors("cal")
    ids = calibration_styles(index, bank)
    roles, _ = episode_roles(data.folds["cal"], "cal", index, ids * 8,
                             oracles.support_pairs)
    # The final current-image FiLM is reused for query evaluation, including COLD_G3,
    # whose corrected code intentionally is not committed as adaptive history.
    ambient = []
    def remember(_module, args):
        if len(args) > 1 and args[1] is not None:
            ambient[:] = [args[1].detach().clone()]
    hook = host.segmenter.register_forward_pre_hook(remember)
    values = []
    try:
        for visit, (anchor, (support_name, query_name)) in enumerate(zip(ids, roles)):
            support, query = data.get(support_name, "cal"), data.get(query_name, "cal")
            support_image = simulate(support.image, bank[anchor],
                                     f"R8_GRAD_LR_SUPPORT|{index}|{visit}|{support_name}")
            host.step(support_image)
            if len(ambient) != 1:
                raise ValueError("R8 source query missing committed intervention")
            query_image = simulate(query.image, bank[anchor],
                                   f"R8_GRAD_LR_QUERY|{index}|{visit}|{query_name}")
            with torch.no_grad():
                p = host.segmenter(query_image, ambient[0]).sigmoid()
                dice = (2 * (p * query.label).sum((0, 2, 3)) + 1e-6) / (
                    p.sum((0, 2, 3)) + query.label.sum((0, 2, 3)) + 1e-6)
            values.append(float(dice.mean()))
        return dict(episode=index, source_seed=source_seed, arm=host.arm, lr=host.lr,
                    soft_Dice=sum(values) / 4, support_query_groups=[list(pair) for pair in roles[:4]])
    finally:
        hook.remove()


def augmentation_seed(episode_index, source_seed):
    # Candidate LRs and arms share the same source-cal augmentation schedule.
    return generator("gradient_lr_augmentation", episode_index, source_seed).initial_seed() % 2**32


def select(rows):
    expected = {(arm, lr, seed, index) for arm in STEPS for lr in LR
                for seed in (20260924, 20260925) for index in range(64)}
    keyed = {(row["arm"], row["lr"], row["source_seed"], row["episode"]): row for row in rows}
    if len(rows) != len(expected) or set(keyed) != expected:
        raise ValueError("R8 complete 1152 source-cal gradient episodes required")
    if any(not math.isfinite(row["soft_Dice"]) or not 0 <= row["soft_Dice"] <= 1 for row in rows):
        raise ValueError("R8 invalid source-cal query Dice")
    scores = {arm: {lr: sum(row["soft_Dice"] for row in rows
                             if row["arm"] == arm and row["lr"] == lr) / 128
                    for lr in sorted(LR)} for arm in STEPS}
    return dict(selected_lr={arm: min(values, key=lambda lr: (-values[lr], lr))
                             for arm, values in scores.items()}, scores=scores)
