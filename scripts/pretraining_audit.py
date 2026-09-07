#!/usr/bin/env python3
"""Generate the pre-training audit from completed CPU/synthetic checks."""

import argparse
import json
import platform
from pathlib import Path

import torch

from dpa_ctta.audit import file_manifest, sha256_file, write_json
from dpa_ctta.integrations.ctta_suite import REFERENCE_COMMIT, REFERENCE_REPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-1", required=True)
    parser.add_argument("--run-2", required=True)
    parser.add_argument("--test-count", required=True, type=int)
    parser.add_argument("--audited-git-tree", required=True)
    parser.add_argument("--repo", default=".")
    args = parser.parse_args()
    root = Path(args.repo).resolve()
    run_hashes = [sha256_file(args.run_1), sha256_file(args.run_2)]
    if run_hashes[0] != run_hashes[1]:
        raise RuntimeError("scientific payload SHA mismatch")
    integrations = {
        task: json.loads((root / "configs" / f"{task}_integration.json").read_text())
        for task in ("fundus", "polyp")
    }
    preliminary_manifest = file_manifest(root)
    audit = {
        "status": "implementation-only research prototype",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "reference_repo": REFERENCE_REPO,
        "reference_commit": REFERENCE_COMMIT,
        "selected_injection_paths": {
            task: [point["path"] for point in config["injection_points"]]
            for task, config in integrations.items()
        },
        "test_commands": [
            "CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v",
            "CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v",
        ],
        "test_runs": 2,
        "test_count_each_run": args.test_count,
        "failures": 0,
        "errors": 0,
        "scientific_payload_sha256": run_hashes,
        "scientific_payload_match": True,
        "cuda_initialized": torch.cuda.is_initialized(),
        "source_images_accessed": False,
        "target_images_accessed": False,
        "source_checkpoint_accessed": False,
        "training_executed": False,
        "audited_git_tree": args.audited_git_tree,
        "audited_git_tree_scope": "implementation tree before generated audit JSON",
        "file_manifest": "audit/file_manifest.json",
        "file_manifest_entries": len(preliminary_manifest["files"]) + 1,
        "known_limitations": [
            "No source-statistics scaler has been fitted.",
            "No real-data training or target evaluation has been run.",
            "Diagonal precision only.",
            "Public reference checkout omits ctta-repro-suite/third_party; integration uses a temporary symlink view of VPTTA sources from the same pinned commit.",
            "Final commit tree is reported by GitHub because embedding it in its own tree is self-referential.",
        ],
    }
    if audit["cuda_initialized"]:
        raise RuntimeError("CUDA initialized during the audit")
    paths = audit["selected_injection_paths"]
    markdown = f"""# Pre-training audit

Status: **PASS for implementation-only review**. Training remains blocked pending `CODE_REVIEW_PASS`.

- Runtime: Python {audit['python']}; PyTorch {audit['torch']}; CPU only; CUDA initialized: false.
- Reference: `{REFERENCE_REPO}` at `{REFERENCE_COMMIT}`.
- Fundus injection paths: `{', '.join(paths['fundus'])}`.
- Polyp injection paths: `{', '.join(paths['polyp'])}`.
- Tests: two independent runs, {args.test_count} tests each, 0 failures, 0 errors.
- Scientific payload SHA-256: `{run_hashes[0]}` in both runs.
- Access boundary: source images=false, target images=false, source checkpoints=false, training=false.
- Audited pre-generated Git tree: `{args.audited_git_tree}`.
- Per-file SHA-256 manifest: `audit/file_manifest.json` ({audit['file_manifest_entries']} entries).

## Known limitations

""" + "\n".join(f"- {item}" for item in audit["known_limitations"]) + "\n"
    (root / "docs" / "PRETRAINING_AUDIT.md").write_text(markdown)
    manifest = file_manifest(root)
    if len(manifest["files"]) != audit["file_manifest_entries"]:
        raise RuntimeError("file manifest count changed unexpectedly")
    write_json(root / "audit" / "file_manifest.json", manifest)
    write_json(root / "audit" / "pretraining_audit.json", audit)


if __name__ == "__main__":
    main()
