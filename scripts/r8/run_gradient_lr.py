"""All three arms, three LRs and two source seeds; freeze before target scoring."""
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from dpa_ctta.b1_host import GRATA_COMMIT
from dpa_ctta.r7_shared.context import tensor_digest
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r8_ba.context import capture
from dpa_ctta.r8_ba.gradient import GradientHost, LR, STEPS
from dpa_ctta.r8_ba.gradient_calibration import augmentation_seed, episode, select
from dpa_ctta.r8_ba.independent_journal import IndependentJournal
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.journal import _replace, _reject_recorded_noninfra_failure
from dpa_ctta.r8_ba.oracle_journal import load_oracles
from dpa_ctta.r8_ba.paths import owned_source_path
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.target_factory import load_deployed


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if config.get("schema") != "R8_GRADIENT_LR_WORK_V1" or config.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("R8 gradient LR worker config")
    root = owned_source_path(config["job_root"])
    _reject_recorded_noninfra_failure(root)
    ref = config["source_index"]
    index = json.loads(verified(owned_source_path(ref["path"]), ref["sha256"], 16 * 1024**2))
    if (index.get("schema") != "R8_SOURCE_ASSET_INDEX_V1" or index.get("code_sha") != config["code_sha"] or
            index.get("protocol_sha256") != PROTOCOL_SHA256 or index.get("refs") != config["refs"]):
        raise ValueError("R8 gradient LR source index identity")
    selected = {}
    for seed in (20260924, 20260925):
        matches = [row for row in index["source_jobs"].values() if row["source_seed"] == seed and
                   row["mode"] == "FULL" and row["config"]["id"] == index["selected_config"]["B"]]
        if len(matches) != 1:
            raise ValueError("R8 gradient LR requires two source-selected B models")
        selected[seed] = matches[0]
    candidate = selected[20260924]["config"]
    bound = bind_metadata(config["refs"])
    common = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256, refs=config["refs"],
                  checkpoint_sha256=bound["docs"]["manifest"]["checkpoint"]["sha256"],
                  amplitude=candidate["film_amplitude"])
    oracle_root = owned_source_path(config["oracle_root"])
    oracle_sha = hashlib.sha256((oracle_root / "oracle_complete.json").read_bytes()).hexdigest()
    marker = json.loads((oracle_root / "worker_complete.json").read_text())
    if (marker.get("schema") != "R8_ORACLE_WORK_COMPLETE_V1" or marker.get("identity") != common or
            marker.get("oracle_receipt_sha256") != oracle_sha or
            any(row["oracle_receipt_sha256"] != oracle_sha for row in selected.values())):
        raise ValueError("R8 gradient LR source oracle producer")
    identity = dict(**common, source_index_sha256=ref["sha256"], oracle_receipt_sha256=oracle_sha)
    items = [dict(arm=arm, lr=lr, source_seed=seed, episode=i) for arm in STEPS for lr in sorted(LR)
             for seed in (20260924, 20260925) for i in range(64)]
    journal = IndependentJournal(root, identity, items)
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 gradient LR wall cap reached")
    try:
        with open_source(bound, config["source_root"], config["checkpoint_path"], candidate["film_amplitude"],
                         config["physical_gpu"], 256 * 1024**2, 2 * 1024**3, guard) as (data, segmenter, io_counts):
            binding = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
            oracles = load_oracles(oracle_root, data, binding, candidate["film_amplitude"])
            methods = {seed: load_deployed(row, candidate, seed, "FULL") for seed, row in selected.items()}
            scales = {seed: torch.tensor(row["gradient_scale"], dtype=torch.float64) for seed, row in selected.items()}
            def run(item):
                seed = item["source_seed"]
                source = dict(checkpoint_sha256=common["checkpoint_sha256"],
                    source_manifest_sha256=config["refs"]["manifest"]["sha256"],
                    source_split_sha256=config["refs"]["split"]["sha256"], source_oracle_sha256=oracle_sha,
                    basis_sha256=selected[seed]["bases_receipt_sha256"], spec_sha256=index["spec_sha256"],
                    protocol_sha256=PROTOCOL_SHA256)
                method_config = dict(candidate, gradient_arm=item["arm"], gradient_lr=item["lr"],
                    scale_sha256=tensor_digest([("scale", scales[seed])]), grata_commit=GRATA_COMMIT)
                expected = capture(segmenter, methods[seed], method_config, source)
                host = GradientHost(segmenter, methods[seed], method_config, source, expected,
                                    item["arm"], scales[seed], item["lr"])
                fixed_seed = augmentation_seed(item["episode"], seed)
                random.seed(fixed_seed)
                np.random.seed(fixed_seed)
                torch.manual_seed(fixed_seed)
                return episode(host, data, oracles["cal"], item["episode"], seed)
            if root.exists():
                journal.recover_once(config.get("resume_failure"))
            else:
                journal.create()
            hook = segmenter.register_forward_pre_hook(lambda *_: guard())
            try:
                journal.run(run, guard)
            finally:
                hook.remove()
            selection = select([row["result"] for row in journal.rows])
        selection.update(schema="R8_GRADIENT_LR_SELECTION_V1", identity=identity)
        _replace(root / "lr_selection.json", json.dumps(selection, sort_keys=True).encode())
        _replace(root / "worker_complete.json", json.dumps(dict(schema="R8_GRADIENT_LR_WORK_COMPLETE_V1",
            identity=identity, receipt_sha256=hashlib.sha256((root / "complete.json").read_bytes()).hexdigest(),
            selection_sha256=hashlib.sha256((root / "lr_selection.json").read_bytes()).hexdigest(),
            source_io_counts=dict(io_counts)), sort_keys=True).encode())
    except BaseException as exc:
        if root.exists():
            with (root / "worker_failures.jsonl").open("a") as output:
                output.write(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)[:3000])) + "\n")
                output.flush()
                os.fsync(output.fileno())
        raise


if __name__ == "__main__":
    main()
