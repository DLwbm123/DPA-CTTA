"""Run existing CPU tests and focused publication checks; never load data/weights."""

import json
import os
from pathlib import Path
import platform
import sys
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]
os.environ["PYTHONPATH"] = str(ROOT / "src")

import torch
import torch.utils.model_zoo
from PIL import Image

from common import make_online
from dpa_ctta.integrations.ctta_suite import build_reference_model, INPUT_SIZES
from dpa_ctta.manifold import MultiScaleLatentFiLM


def forbidden(*args, **kwargs):
    raise AssertionError("Publication checks forbid image/weight loading and downloads")


def focused_checks():
    findings = {}
    torch.manual_seed(20260907)
    adapter = make_online()
    image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    try:
        queries = []
        for previous in (0.0, 3.0, -2.0):
            adapter.z_prev.fill_(previous)
            logits, feature = adapter._zero_pass(image)
            queries.append(adapter.atlas.descriptor(image, logits.sigmoid(), feature))
        difference = max(float((q - queries[0]).abs().max()) for q in queries)
        assert difference == 0.0
        findings["descriptor_z_prev_max_abs_difference"] = difference
    finally:
        adapter.injected_model.close()

    for task, size in INPUT_SIZES.items():
        torch.manual_seed(20260907)
        source, _ = build_reference_model(task)
        config = json.loads((ROOT / "configs" / f"{task}_integration.json").read_text())
        channels = {p["path"]: p["shape"][1] for p in config["injection_points"]}
        before = {name: tensor.detach().clone() for name, tensor in source.state_dict().items()}
        with MultiScaleLatentFiLM(source, channels) as wrapper:
            image = torch.linspace(-1, 1, 3 * size * size).reshape(1, 3, size, size).requires_grad_()
            state = torch.zeros(1, 16, requires_grad=True)
            output = wrapper(image, state)
            logits = output[0] if isinstance(output, tuple) else output
            input_grad, latent_grad = torch.autograd.grad(logits.square().mean(), (image, state))
            for gradient in (input_grad, latent_grad):
                assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
            assert all(p.grad is None and not p.requires_grad for p in source.parameters())
            assert all(torch.equal(before[name], value) for name, value in source.state_dict().items())
            findings[task] = {
                "input_gradient_norm": float(input_grad.norm()),
                "zero_latent_gradient_norm": float(latent_grad.norm()),
                "source_parameters_and_registered_buffers_unchanged": True,
                "scope": "random CPU source_only model; custom BN replaced by standard BN upstream",
            }
    return findings


def main():
    torch.set_num_threads(2)
    torch.manual_seed(20260907)
    assert not torch.cuda.is_initialized()
    with patch.object(torch, "load", forbidden), patch.object(Image, "open", forbidden), \
         patch.object(torch.hub, "download_url_to_file", forbidden), \
         patch.object(torch.utils.model_zoo, "load_url", forbidden):
        suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        findings = focused_checks() if result.wasSuccessful() else {}
    record = {
        "python": platform.python_version(), "torch": torch.__version__,
        "tests_run": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "skipped": len(result.skipped),
        "focused_checks": findings, "cuda_initialized": torch.cuda.is_initialized(),
        "training_executed": False, "real_data_accessed": False,
        "source_checkpoint_loaded": False,
        "note": "Existing synthetic oracle unit checks and M0 CLI rejection checks are not experiment runs.",
    }
    assert not record["cuda_initialized"]
    (ROOT / "audit/publication_checks.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
