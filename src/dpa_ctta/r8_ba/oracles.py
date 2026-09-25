"""R8 source-only ambient oracles and nested A/B bases."""
from dataclasses import dataclass

import torch

from ..r7_shared.numerics import COUNTS, finite, predictive_basis, seg_loss, svd_basis
from ..r7_shared.source import pooled_jacobian, simulate
from .schedule import SIZES, anchors, generator, oracle_roles


@dataclass(frozen=True)
class Oracles:
    fold: str
    amplitude: float
    values: torch.Tensor
    support_pairs: tuple
    query_pairs: tuple
    diagnostics: tuple

    def validate(self, data):
        n = SIZES[self.fold]
        if self.amplitude not in (0.1, 0.3) or self.values.shape != (1024, n):
            raise ValueError("R8 oracle shape/amplitude")
        if len(self.support_pairs) != n or len(self.query_pairs) != n or len(self.diagnostics) != n:
            raise ValueError("R8 oracle metadata length")
        finite(self.values)
        valid = set(data.folds[self.fold])
        for support, query in zip(self.support_pairs, self.query_pairs):
            if len(set((*support, *query))) != 4 or not set((*support, *query)) <= valid:
                raise ValueError("R8 support/query group leakage")
        return self


def oracle_one(segmenter, data, fold, index, on_step=None):
    if fold not in SIZES or index not in range(SIZES[fold]):
        raise ValueError("R8 oracle fold/index")
    style = anchors(fold)[index]
    names = oracle_roles(data.folds[fold], fold, index)
    supports = [data.get(name, fold) for name in names[:2]]
    queries = [data.get(name, fold) for name in names[2:]]
    support_images = [simulate(r.image, style, f"R8_ORACLE_SUPPORT|{fold}|{index}|{r.group}") for r in supports]
    query_images = [simulate(r.image, style, f"R8_ORACLE_QUERY|{fold}|{index}|{r.group}") for r in queries]
    v = torch.zeros(1024, requires_grad=True)
    optimizer = torch.optim.Adam([v], lr=0.03, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    diagnostics = []
    for step in range(1, 257):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(seg_loss(segmenter(image, v), row.label) for image, row in zip(support_images, supports)) / 2
        loss = loss + 1e-3 * v.square().mean()
        finite(loss)
        loss.backward()
        COUNTS["source_backward_calls"] += 1
        finite(v.grad)
        optimizer.step()
        COUNTS["source_Adam"] += 1
        finite(v)
        if step in (16, 64, 256):
            with torch.no_grad():
                dice = []
                for image, row in zip(query_images, queries):
                    prediction = segmenter(image, v).sigmoid()
                    score = (2 * (prediction * row.label).sum((0, 2, 3)) + 1e-6) / (
                        prediction.sum((0, 2, 3)) + row.label.sum((0, 2, 3)) + 1e-6)
                    dice.append(score.tolist())
            diagnostics.append(dict(step=step, support_objective=float(loss.detach()), query_Dice=dice,
                                    ambient_norm=float(v.detach().norm()), film_amplitude=segmenter.amplitude))
        if on_step is not None:
            on_step(step)
    return v.detach(), names[:2], names[2:], diagnostics


def oracle_all(segmenter, data, fold):
    if segmenter.amplitude not in (0.1, 0.3):
        raise ValueError("R8 source segmenter amplitude")
    rows = [oracle_one(segmenter, data, fold, index) for index in range(SIZES[fold])]
    return Oracles(fold, segmenter.amplitude, torch.stack([r[0] for r in rows], dim=1),
                   tuple(r[1] for r in rows), tuple(r[2] for r in rows),
                   tuple(tuple(r[3]) for r in rows)).validate(data)


def b_bases(fit_oracles, data):
    fit_oracles.validate(data)
    if fit_oracles.fold != "fit":
        raise ValueError("B basis must use fit oracles")
    u64 = svd_basis(fit_oracles.values, 64)
    return {32: u64[:, :32], 64: u64}


def a_bases(segmenter, data):
    groups = sorted(data.folds["fit"], key=lambda group: generator("basis", group).initial_seed())
    if len(groups) < 32:
        raise ValueError("A basis requires 32 fit groups")
    cov, readout = pooled_jacobian(segmenter, data, groups[:16])
    probe, _ = pooled_jacobian(segmenter, data, groups[16:32])
    curvature = readout.sigmoid() * (1 - readout.sigmoid())
    u32, covariance = predictive_basis(cov, probe, curvature, rank=32)
    return {16: u32[:, :16], 32: u32}, dict(covariance_groups=groups[:16], probe_groups=groups[16:32],
                                             covariance=covariance)
