"""Fresh-process posthoc scoring after a fully sealed R8 online worker."""
import hashlib
import json
import os
import runpy
import time
from pathlib import Path

from dpa_ctta.r8_ba.journal import _replace
from dpa_ctta.r8_ba.paths import owned_target_path
from dpa_ctta.r8_ba.scoring import score
from dpa_ctta.r8_ba.streams import rows_sha


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_SCORE_JOB_WORK_V1" or
            type(config.get("maximum_seconds")) is not int or config["maximum_seconds"] <= 0):
        raise ValueError("R8 scoring worker config")
    root = owned_target_path(config["job_root"])
    if (root / "worker_complete.json").exists():
        raise ValueError("R8 scored job already complete")
    load = runpy.run_path(str(Path(__file__).resolve().with_name("run_target_job.py")))["inputs"]
    job, _, _, _, rows, _ = load(config)
    marker = json.loads((root / "online_worker_complete.json").read_text())
    if (marker.get("schema") != "R8_TARGET_ONLINE_WORK_COMPLETE_V1" or
            marker.get("job") != job or marker.get("code_sha") != config["code_sha"] or
            marker.get("artifact_lock_sha256") != config["artifact_lock"]["sha256"] or
            marker.get("rows_sha256") != rows_sha(rows) or
            marker.get("online_receipt_sha256") != hashlib.sha256((root / "online_complete.json").read_bytes()).hexdigest()):
        raise ValueError("R8 scoring requires verified online worker marker")
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 scoring worker wall cap reached")
    receipt = score(rows, config["target_root"], root, job["id"], marker["context_sha256"],
                    job["arm"], job["order"], 256 * 1024**2, guard, config.get("resume_failure"))
    load(config)  # Verify registered metadata and locked source identities again after mask reads.
    _replace(root / "worker_complete.json", json.dumps(dict(
        schema="R8_TARGET_JOB_WORK_COMPLETE_V1", code_sha=config["code_sha"], job=job,
        online_worker_sha256=hashlib.sha256((root / "online_worker_complete.json").read_bytes()).hexdigest(),
        score_receipt_sha256=hashlib.sha256((root / "score_complete.json").read_bytes()).hexdigest()),
        sort_keys=True).encode())
    return receipt


if __name__ == "__main__":
    main()
