"""Shared fit-only observations and alpha-specific R8 source preparation."""
import torch

from ..r7_shared.source import simulate
from ..r7_shared.numerics import finite, shape
from .oracles import a_bases, b_bases, oracle_all
from .schedule import anchors


@torch.no_grad()
def scaler_anchor(segmenter, data, index, style, on_group=lambda _: None):
    """One complete fit-only anchor; independent boundaries permit exact replay."""
    values = []
    for group in data.folds["fit"]:
        row = data.get(group, "fit")
        image = simulate(row.image, style, f"R8_SCALER|{index}|{group}")
        _, raw, _ = segmenter(image, observe=True)
        shape(raw, (134,))
        values.append(raw.detach().cpu())
        on_group(group)
    return torch.stack(values)


@torch.no_grad()
def scaler_observations(segmenter, data):
    """One zero-FiLM fit-only raw-observer cache shared by all 65 models."""
    bank = anchors("fit")
    return torch.cat([scaler_anchor(segmenter, data, index, style)
                      for index, style in enumerate(bank)])


def prepare_alpha(segmenter, data):
    """No target access; caller owns persistence, resource caps, and receipts."""
    oracles = {fold: oracle_all(segmenter, data, fold) for fold in ("fit", "cal", "val")}
    b = b_bases(oracles["fit"], data)
    a, audit = a_bases(segmenter, data)
    return dict(amplitude=segmenter.amplitude, oracles=oracles, B_basis=b, A_basis=a, A_basis_audit=audit)


def gradient_scale(fit_oracles, basis):
    """Per-coordinate source-fit proxy RMS for the three B gradient arms."""
    if (fit_oracles.fold != "fit" or fit_oracles.values.shape != (1024, 512) or
            basis.ndim != 2 or basis.shape[0] != 1024 or basis.shape[1] not in (32, 64)):
        raise ValueError("R8 gradient scale fit/rank identity")
    zstar = basis.T.double() @ fit_oracles.values.double()
    finite(zstar)
    return zstar.square().mean(1).sqrt().clamp_min(1e-3)
