"""Frozen source-only style and 32-visit role schedules for R8."""
import hashlib

import torch

from ..r7_shared.source import IDENTITY, RANGES

SIZES = dict(fit=512, cal=128, val=128)
CURRICULA = ("abrupt_16_16", "gradual_32", "recurrence_8_8_8_8", "iid_style_32")


def generator(*parts):
    raw = "|".join(map(str, ("R8_BA_V1", 20260924, *parts))).encode()
    return torch.Generator().manual_seed(int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63))


def anchors(fold):
    if fold not in SIZES:
        raise ValueError("source fold")
    rows = []
    for index in range(SIZES[fold]):
        row = torch.tensor(IDENTITY)
        if index:
            if fold == "val":
                active = (1, 2)[(index - 1) % 2] if index < 64 else (5, 6)[(index - 64) % 2]
            else:
                active = 1 + (index - 1) % 4
            rng = generator("style", fold, index)
            for dim in torch.randperm(9, generator=rng)[:active].tolist():
                low, high = RANGES[dim]
                row[dim] = low + (high - low) * torch.rand((), generator=rng)
        rows.append(row)
    return torch.stack(rows)


def _choices(fold, episode):
    if fold != "val":
        return list(range(SIZES[fold])), "mixed"
    within = episode % 16
    if within < 4:
        return [0], "clean"
    if within < 10:
        return list(range(1, 64)), "mild"
    return list(range(64, 128)), "compound"


def episode_styles(fold, episode):
    if fold not in SIZES or episode < 0:
        raise ValueError("source fold/episode")
    mode = CURRICULA[episode % 4] if fold != "val" else CURRICULA[episode // 16 % 4]
    choices, severity = _choices(fold, episode)
    rng = generator("episode", fold, episode)
    if len(choices) == 1:
        return [choices[0]] * 32, mode, severity
    perm = torch.randperm(len(choices), generator=rng).tolist()
    ordered = [choices[i] for i in perm]
    if mode == "abrupt_16_16":
        ids = [ordered[0]] * 16 + [ordered[1]] * 16
    elif mode == "recurrence_8_8_8_8":
        ids = [ordered[0]] * 8 + [ordered[1]] * 8 + [ordered[0]] * 8 + [ordered[1]] * 8
    elif mode == "iid_style_32":
        ids = [ordered[i] for i in torch.randint(len(ordered), (32,), generator=rng).tolist()]
    else:
        bank = anchors(fold)
        scale = torch.tensor([high - low for low, high in RANGES])
        ids = [ordered[0]]
        for _ in range(31):
            last = ids[-1]
            ids.append(min((i for i in choices if i not in ids),
                           key=lambda i: (float((((bank[i] - bank[last]) / scale) ** 2).sum()), i)))
    return ids, mode, severity


def oracle_roles(groups, fold, anchor):
    if fold not in SIZES or len(set(groups)) < 4:
        raise ValueError("oracle requires four source groups")
    ordered = sorted(groups)
    perm = torch.randperm(len(ordered), generator=generator("oracle_roles", fold, anchor)).tolist()
    return tuple(ordered[i] for i in perm[:4])  # two support, two held-out query


def episode_roles(groups, fold, episode, style_ids, oracle_support):
    """64 roles; query never uses the current anchor's oracle support groups."""
    if fold not in SIZES or len(style_ids) != 32 or len(set(groups)) < 4:
        raise ValueError("episode roles input")
    ordered = sorted(groups)
    perm = torch.randperm(len(ordered), generator=generator("episode_roles", fold, episode)).tolist()
    ordered = [ordered[i] for i in perm]
    used, roles = set(), []
    for anchor in style_ids:
        pair = set(oracle_support[anchor])
        support = next((g for g in ordered if g not in used), None)
        if support is None:
            support = next(g for g in ordered if not roles or g != roles[-1][0])
        used.add(support)
        query = next((g for g in ordered if g not in used and g not in pair and g != support), None)
        if query is None:
            query = next(g for g in ordered if g not in pair and g != support)
        used.add(query)
        roles.append((support, query))
    return roles, len({g for pair in roles for g in pair}) < 64
