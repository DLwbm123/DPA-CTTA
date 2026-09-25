"""Shared fit-only observations and alpha-specific R8 source preparation."""
import hashlib
import io
import json
from pathlib import Path

import torch

from ..r7_shared.source import simulate
from ..r7_shared.numerics import finite, shape
from .oracles import a_bases, b_bases, oracle_all
from .schedule import anchors
from .journal import _replace


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


def _check_bases(payload, identity):
    if (payload.get("schema") != "R8_SOURCE_BASES_V1" or
            payload.get("identity") != identity or
            set(payload.get("A_basis", {})) != {16, 32} or
            set(payload.get("B_basis", {})) != {32, 64} or
            set(payload.get("gradient_scale", {})) != {32, 64}):
        raise ValueError("R8 basis artifact identity")
    for name, ranks in (("A_basis", (16, 32)), ("B_basis", (32, 64))):
        for rank in ranks:
            value = payload[name][rank]
            if (value.shape != (1024, rank) or value.dtype != torch.float64 or
                    value.device.type != "cpu"):
                raise ValueError("R8 basis shape/dtype")
            finite(value)
        if not torch.equal(payload[name][ranks[-1]][:, :ranks[0]],
                           payload[name][ranks[0]]):
            raise ValueError("R8 nested basis prefix")
    for rank in (32, 64):
        scale = payload["gradient_scale"][rank]
        if (scale.shape != (rank,) or scale.dtype != torch.float64 or
                scale.device.type != "cpu" or (scale < 1e-3).any()):
            raise ValueError("R8 gradient scale shape")
        finite(scale)
    audit = payload.get("A_basis_audit")
    if (not isinstance(audit, dict) or
            len(audit.get("covariance_groups", ())) != 16 or
            len(audit.get("probe_groups", ())) != 16 or
            set(audit["covariance_groups"]) & set(audit["probe_groups"]) or
            audit.get("covariance").shape != (1024, 1024)):
        raise ValueError("R8 basis source group audit")
    finite(audit["covariance"])
    return payload


def save_bases(root, identity, a, b, audit, scale, counts):
    """Seal both nested bases only after the registered source scope rechecks inputs."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    payload = _check_bases(dict(schema="R8_SOURCE_BASES_V1", identity=identity,
                                A_basis=a, B_basis=b, A_basis_audit=audit,
                                gradient_scale=scale), identity)
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    raw = buffer.getvalue()
    _replace(root / "bases.pt", raw)
    receipt = dict(schema="R8_SOURCE_BASES_COMPLETE_V1", identity=identity,
                   bases_sha256=hashlib.sha256(raw).hexdigest(), physical_counts=dict(counts))
    _replace(root / "bases_complete.json",
             (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
    return receipt


def load_bases(root, identity):
    root = Path(root)
    receipt = json.loads((root / "bases_complete.json").read_text())
    raw = (root / "bases.pt").read_bytes()
    if (receipt.get("schema") != "R8_SOURCE_BASES_COMPLETE_V1" or
            receipt.get("identity") != identity or
            receipt.get("bases_sha256") != hashlib.sha256(raw).hexdigest()):
        raise ValueError("R8 basis completion identity/digest")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    return _check_bases(payload, identity)
