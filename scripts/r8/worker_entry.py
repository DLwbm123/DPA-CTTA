"""Neutral-symlink process entry for an already admitted R8 worker."""
import hashlib
import json
import os
import runpy
from pathlib import Path

from dpa_ctta.r8_ba.ledger import Ledger
from dpa_ctta.r8_ba.paths import owned_source_path
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.worker_budget import WorkerBudget

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = {"R8_ORACLE_WORK_V1": "run_oracles.py", "R8_SCALER_WORK_V1": "run_scaler.py",
           "R8_BASES_WORK_V1": "run_bases.py", "R8_SOURCE_JOB_WORK_V1": "run_source_job.py"}


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if config.get("schema") not in ENTRIES:
        raise ValueError("R8 worker entry not implemented for this stage")
    row = config["execution_ledger"]
    expected = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                    graph_sha256=hashlib.sha256((ROOT / "docs/review/r8/TASK_GRAPH.static.json").read_bytes()).hexdigest())
    ledger_root = Path(row["root"]).resolve()
    package_root = Path("/data_nas/jiangsuiyang/CTTA/r8-ba-performance-envelope-v1")
    if not ledger_root.is_relative_to(package_root / "runs") or row["identity"] != expected:
        raise ValueError("R8 aggregate ledger output/identity binding")
    root = owned_source_path(config["job_root"])
    ledger = Ledger(ledger_root, expected)
    with WorkerBudget(ledger, row["attempt_id"], root, row["budget"], config["maximum_seconds"]):
        runpy.run_path(str(ROOT / "scripts/r8" / ENTRIES[config["schema"]]), run_name="__main__")


if __name__ == "__main__":
    main()
