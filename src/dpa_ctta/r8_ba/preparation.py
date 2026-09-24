"""Shared fit-only observations and alpha-specific R8 source preparation."""
import torch

from ..r7_shared.source import simulate
from .oracles import a_bases, b_bases, oracle_all
from .schedule import anchors


@torch.no_grad()
def scaler_observations(segmenter, data):
    """One zero-FiLM fit-only raw-observer cache shared by all 65 models."""
    bank = anchors("fit")
    values = []
    for index, style in enumerate(bank):
        for group in data.folds["fit"]:
            row = data.get(group, "fit")
            image = simulate(row.image, style, f"R8_SCALER|{index}|{group}")
            _, raw, _ = segmenter(image, observe=True)
            values.append(raw.detach())
    return torch.stack(values)


def prepare_alpha(segmenter, data):
    """No target access; caller owns persistence, resource caps, and receipts."""
    oracles = {fold: oracle_all(segmenter, data, fold) for fold in ("fit", "cal", "val")}
    b = b_bases(oracles["fit"], data)
    a, audit = a_bases(segmenter, data)
    return dict(amplitude=segmenter.amplitude, oracles=oracles, B_basis=b, A_basis=a, A_basis_audit=audit)
