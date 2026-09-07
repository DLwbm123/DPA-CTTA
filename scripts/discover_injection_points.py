#!/usr/bin/env python3
"""Discover and record real decoder/tail injection points on CPU synthetic inputs."""

import argparse
import json
from pathlib import Path

import torch

from dpa_ctta.integrations.ctta_suite import discover_injection_points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", help="Pinned DLwbm123/CTTA checkout; defaults to DPA_CTTA_BASE_ROOT")
    parser.add_argument("--output-dir", default="configs")
    args = parser.parse_args()
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA was initialized before discovery")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for task in ("fundus", "polyp"):
        result = discover_injection_points(task, args.root)
        (output / f"{task}_integration.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )
    if torch.cuda.is_initialized():
        raise RuntimeError("CUDA initialized during CPU discovery")


if __name__ == "__main__":
    main()
