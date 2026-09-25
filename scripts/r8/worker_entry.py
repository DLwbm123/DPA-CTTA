"""Neutral-symlink process entry for an already admitted R8 worker."""
import hashlib
import json
import os
import runpy
from pathlib import Path

from dpa_ctta.r8_ba.ledger import Ledger
from dpa_ctta.r8_ba.paths import owned_source_path, owned_target_path
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.worker_budget import WorkerBudget
from dpa_ctta.r8_ba.launch_binding import verify

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = {"R8_ORACLE_WORK_V1": "run_oracles.py", "R8_SCALER_WORK_V1": "run_scaler.py",
           "R8_BASES_WORK_V1": "run_bases.py", "R8_SOURCE_JOB_WORK_V1": "run_source_job.py",
           "R8_TARGET_JOB_WORK_V1": "run_target_job.py", "R8_SCORE_JOB_WORK_V1": "run_score_job.py",
           "R8_CAPACITY_WORK_V1": "run_capacity.py", "R8_GRADIENT_LR_WORK_V1": "run_gradient_lr.py"}


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if config.get("schema") not in ENTRIES:
        raise ValueError("R8 worker entry not implemented for this stage")
    verify(config, ROOT)
    row = config["execution_ledger"]
    expected = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                    graph_sha256=hashlib.sha256((ROOT / "docs/review/r8/TASK_GRAPH.static.json").read_bytes()).hexdigest())
    ledger_root = Path(row["root"]).resolve()
    package_root = Path("/data_nas/jiangsuiyang/CTTA/r8-ba-performance-envelope-v1")
    if not ledger_root.is_relative_to(package_root / "runs") or row["identity"] != expected:
        raise ValueError("R8 aggregate ledger output/identity binding")
    root = (owned_target_path if config["schema"] in ("R8_TARGET_JOB_WORK_V1", "R8_SCORE_JOB_WORK_V1")
            else owned_source_path)(config["job_root"])
    ledger = Ledger(ledger_root, expected)
    admitted = ledger.snapshot()["attempts"][row["attempt_id"]]
    if admitted.get("config_sha256") != hashlib.sha256(Path(os.environ["R8_WORK_CONFIG"]).read_bytes()).hexdigest():
        ledger.stop("worker config differs from admitted job")
        raise ValueError("R8 worker config identity")
    with WorkerBudget(ledger, row["attempt_id"], root, row["budget"], config["maximum_seconds"]):
        runpy.run_path(str(ROOT / "scripts/r8" / ENTRIES[config["schema"]]), run_name="__main__")


if __name__ == "__main__":
    main()
