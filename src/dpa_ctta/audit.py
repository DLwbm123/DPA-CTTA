"""Small deterministic audit helpers used by repository scripts."""

import hashlib
import json
from pathlib import Path
import subprocess

import torch

from .anchors import DistilledPotentialAnchor
from .atlas import DistilledPotentialAtlas
from .descriptors import DescriptorScaler, FrozenDescriptor
from .proximal import closed_form_update


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_payload(seed=20260903):
    torch.manual_seed(seed)
    descriptor = FrozenDescriptor("polyp", feature_channels=4)
    scaler = DescriptorScaler(descriptor.output_dim)
    anchors = []
    for offset in (-0.3, 0.4):
        anchors.append(
            DistilledPotentialAnchor(
                torch.randn(1, 3, 16, 16) + offset,
                torch.randn(1, 1, 16, 16) - offset,
                torch.linspace(-0.2, 0.2, 16) + offset,
                torch.randn(16, descriptor.output_dim) * 0.01,
                torch.linspace(-1, 1, 16),
            )
        )
    atlas = DistilledPotentialAtlas(anchors, descriptor, scaler, temperature=0.7)
    projection = torch.linspace(-0.1, 0.1, 12).reshape(4, 3, 1, 1)

    def frozen_feature(image):
        return torch.nn.functional.conv2d(image, projection)

    query_image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
    query_mask = torch.sigmoid(torch.linspace(-2, 2, 16 * 16).reshape(1, 1, 16, 16))
    query = descriptor(query_image, query_mask, frozen_feature(query_image))
    result = atlas(query, frozen_feature)
    update = closed_form_update(torch.zeros(1, 16), result, 1.0, 1e-3)
    return {
        "seed": seed,
        "descriptor_dim": descriptor.output_dim,
        "weights": [round(float(value), 9) for value in result.weights[0]],
        "state": [round(float(value), 9) for value in update.state[0]],
        "precision_min": round(float(result.precision.min()), 9),
        "contraction_bound": update.contraction_bound,
        "cuda_initialized": torch.cuda.is_initialized(),
    }


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def file_manifest(root):
    root = Path(root)
    excluded = {"audit/file_manifest.json", "audit/pretraining_audit.json"}
    entries = []
    public_paths = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for relative in sorted(public_paths):
        path = root / relative
        if relative in excluded:
            continue
        entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {
        "algorithm": "sha256",
        "scope": "repository files excluding generated audit JSON files and Git metadata",
        "files": entries,
    }
