"""Pinned, CPU-only bridge to DLwbm123/CTTA without copying model code."""

from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch import nn

from ..hooks import directional_logits_jvp, enumerate_4d_outputs
from ..manifold import MultiScaleLatentFiLM


REFERENCE_REPO = "DLwbm123/CTTA"
REFERENCE_COMMIT = "dbff0d985c6c95345d9fb78f5b1daef57b392564"
REFERENCE_PACKAGE = "ctta-repro-suite"
INPUT_SIZES = {"fundus": 512, "polyp": 352}


def checkout_root(root=None, require_pinned=True):
    value = root or os.environ.get("DPA_CTTA_BASE_ROOT")
    if not value:
        raise RuntimeError("DPA_CTTA_BASE_ROOT is required for external integration")
    path = Path(value).expanduser().resolve()
    if not (path / ".git").exists() or not (path / REFERENCE_PACKAGE / "src").is_dir():
        raise RuntimeError("DPA_CTTA_BASE_ROOT must be the DLwbm123/CTTA checkout root")
    commit = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if require_pinned and commit != REFERENCE_COMMIT:
        raise RuntimeError(f"reference commit mismatch: expected {REFERENCE_COMMIT}, got {commit}")
    return path, commit


def _load_models(root):
    suite_src = root / REFERENCE_PACKAGE / "src"
    if str(suite_src) not in sys.path:
        sys.path.insert(0, str(suite_src))
    return importlib.import_module("ctta_suite.models")


def load_reference_interfaces(root=None):
    root, _commit = checkout_root(root)
    models = _load_models(root)
    data = importlib.import_module("ctta_suite.data")
    metrics = importlib.import_module("ctta_suite.metrics")
    return models, data, metrics


@contextmanager
def _dependency_view(root, models):
    """Map the public monorepo VPTTA tree to ctta_suite's omitted third_party view."""
    suite = root / REFERENCE_PACKAGE
    if (suite / "third_party").is_dir():
        old_root = models.ROOT
        models.ROOT = suite
        try:
            yield
        finally:
            models.ROOT = old_root
        return
    with tempfile.TemporaryDirectory(prefix="dpa_ctta_deps_") as temporary:
        view = Path(temporary)
        upstream = view / "third_party" / "BIBM2025-MGIPT"
        upstream.mkdir(parents=True)
        for task in ("OPTIC", "POLYP"):
            source = root / "VPTTA" / task
            if not source.is_dir():
                raise RuntimeError(f"pinned model source missing: VPTTA/{task}")
            (upstream / task).symlink_to(source, target_is_directory=True)
        old_root = models.ROOT
        models.ROOT = view
        try:
            yield
        finally:
            models.ROOT = old_root
            prefix = str(view)
            sys.path[:] = [entry for entry in sys.path if not entry.startswith(prefix)]


def build_reference_model(task, root=None):
    if task not in INPUT_SIZES:
        raise ValueError("task must be 'fundus' or 'polyp'")
    root, _commit = checkout_root(root)
    models = _load_models(root)
    with _dependency_view(root, models):
        model = models.build_model(task, "mgipt", source_only=True).cpu().eval().requires_grad_(False)
    return model, models.model_logits


def _select_descriptor_feature(records, model, input_size):
    modules = dict(model.named_modules())
    for record in records:
        shape = record["shape"]
        if (
            isinstance(modules[record["path"]], nn.Conv2d)
            and shape[1] >= 32
            and shape[-1] < input_size
        ):
            return record
    raise RuntimeError("no frozen early-feature module found")


def _select_tail_scales(records):
    tokens = ("up", "bnout", "seg_head", "agg", "rfb", "ra")
    candidates = [
        record
        for record in records
        if not record["path"].startswith(("res.", "resnet."))
        and any(token in record["path"].lower() for token in tokens)
    ]
    areas = sorted({record["shape"][-2] * record["shape"][-1] for record in candidates})
    if len(areas) < 3:
        raise RuntimeError("fewer than three decoder/tail spatial scales found")
    selected_areas = (areas[0], areas[len(areas) // 2], areas[-1])
    return [
        [r for r in candidates if r["shape"][-2] * r["shape"][-1] == area][-1]
        for area in selected_areas
    ]


def discover_injection_points(task, root=None):
    root, commit = checkout_root(root)
    size = INPUT_SIZES[task]
    torch.manual_seed(20260903)
    model, model_logits = build_reference_model(task, root)
    image = torch.linspace(-1, 1, 3 * size * size).reshape(1, 3, size, size)
    with torch.inference_mode():
        baseline = model_logits(model, image).clone()
        records = enumerate_4d_outputs(model, lambda: model_logits(model, image))
    descriptor_feature = _select_descriptor_feature(records, model, size)
    selected = _select_tail_scales(records)
    for record in selected:
        jvp = directional_logits_jvp(model, record["path"], image, model_logits)
        record["jvp_norm"] = float(jvp.norm())
        record["jvp_finite"] = bool(torch.isfinite(jvp).all())
        if not record["jvp_finite"] or record["jvp_norm"] <= 0:
            raise RuntimeError(f"zero/nonfinite logits JVP at {record['path']}")

    channels = {record["path"]: record["shape"][1] for record in selected}
    torch.manual_seed(20260903)
    wrapper = MultiScaleLatentFiLM(model, channels)
    try:
        with torch.inference_mode():
            zero = model_logits(wrapper, image).clone()
            changed = wrapper(image, torch.ones(1, 16))
            changed = changed[0] if isinstance(changed, tuple) else changed
        zero_identity = torch.equal(zero, baseline)
        nonzero_change = not torch.equal(changed, baseline)
    finally:
        wrapper.close()
    if not zero_identity or not nonzero_change:
        raise RuntimeError("real-model FiLM identity/effect contract failed")
    return {
        "task": task,
        "reference_repo": REFERENCE_REPO,
        "reference_commit": commit,
        "model": "ResUNet34" if task == "fundus" else "PraNet",
        "input_shape": [1, 3, size, size],
        "descriptor_feature": descriptor_feature,
        "injection_points": selected,
        "zero_state_bitwise_identity": zero_identity,
        "nonzero_state_logits_change": nonzero_change,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def load_integration_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("reference_commit") != REFERENCE_COMMIT:
        raise ValueError("integration config reference commit mismatch")
    points = config.get("injection_points", [])
    if len(points) != 3 or len({tuple(point["shape"][-2:]) for point in points}) != 3:
        raise ValueError("integration config must contain three distinct spatial scales")
    return config
