"""Pinned, CPU-only bridge to DLwbm123/CTTA without copying model code."""

from contextlib import contextmanager
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import tarfile

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
    verify_tracked_files(root, [f"{REFERENCE_PACKAGE}/src/ctta_suite/{name}.py"
                                for name in ("__init__", "models", "data", "metrics")])
    check_module_origins(("ctta_suite",), suite_src)
    sys.path.insert(0, str(suite_src))
    try:
        models = importlib.import_module("ctta_suite.models")
        check_module_origins(("ctta_suite",), suite_src)
        return models
    finally:
        sys.path.remove(str(suite_src))


def verify_tracked_files(root, paths):
    """Compare the actual critical source bytes with HEAD, even with index skip flags."""
    archive = subprocess.run(["git", "-C", str(root), "archive", "HEAD", "--", *paths],
                             capture_output=True, check=True).stdout
    tracked = set()
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            if not member.isfile() or not member.name.endswith(".py"):
                continue
            path = root / member.name
            if path.is_symlink() or not path.is_file() or path.read_bytes() != stream.extractfile(member).read():
                raise RuntimeError(f"dirty pinned scientific source: {member.name}")
            tracked.add(path)
    for relative in paths:
        path = root / relative
        candidates = path.rglob("*.py") if path.is_dir() else [path]
        if any(p not in tracked for p in candidates):
            raise RuntimeError("untracked/missing Python source in dependency import scope")


def check_module_origins(prefixes, directory):
    for name, module in tuple(sys.modules.items()):
        if any(name == p or name.startswith(p + ".") for p in prefixes):
            file = getattr(module, "__file__", None)
            stem = directory / name.replace(".", "/")
            expected = (stem.with_suffix(".py").resolve(), (stem / "__init__.py").resolve())
            if not file or Path(file).resolve() not in expected:
                raise RuntimeError(f"foreign module cache: {name}; use a fresh interpreter")


@contextmanager
def native_imports(root, task):
    """Scoped upstream absolute imports; never evict an existing user module."""
    folder = "OPTIC" if task == "fundus" else "POLYP"
    directory = root / "VPTTA" / folder
    verify_tracked_files(root, [f"VPTTA/{folder}/{p}" for p in ("networks", "utils", "vptta.py", "dataloaders")])
    prefixes = ("networks", "utils", "dataloaders", "config")
    if any(name.split(".")[0] in prefixes for name in sys.modules):
        raise RuntimeError("foreign native module cache; use a fresh interpreter")
    old_path = sys.path[:]
    sys.path.insert(0, str(directory))
    try:
        yield directory
        check_module_origins(prefixes, directory)
    finally:
        # Only these aliases created in this context can exist; preexisting ones were rejected.
        for name in tuple(sys.modules):
            if name.split(".")[0] in prefixes:
                del sys.modules[name]
        sys.path[:] = old_path


def load_reference_interfaces(root=None):
    root, _commit = checkout_root(root)
    models = _load_models(root)
    data = importlib.import_module("ctta_suite.data")
    metrics = importlib.import_module("ctta_suite.metrics")
    check_module_origins(("ctta_suite",), root / REFERENCE_PACKAGE / "src")
    return models, data, metrics


@contextmanager
def _dependency_view(root, models):
    """Map the public monorepo VPTTA tree to ctta_suite's omitted third_party view."""
    suite = root / REFERENCE_PACKAGE
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
    with _dependency_view(root, models), native_imports(root, task):
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
