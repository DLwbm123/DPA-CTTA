from dpa_ctta.r8_ba.scope import SCREEN, GRAPH_PATH, SPEC_PATH, SOURCE_JOBS, TARGET_JOBS
"""One registered R8 online trajectory, with no target mask reader in this phase."""
import hashlib
import json
import os
import time
from pathlib import Path

from dpa_ctta.r3.plan import stream
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r8_ba.inputs import _gpu_policy, bind_metadata
from dpa_ctta.r8_ba.journal import TargetJournal, _replace, _reject_recorded_noninfra_failure
from dpa_ctta.r8_ba.online import run
from dpa_ctta.r8_ba.paths import owned_source_path, owned_target_path
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.streams import bind, rows_sha, stress
from dpa_ctta.r8_ba.target_factory import construct
from dpa_ctta.r8_ba.target_jobs import resolve

ROOT = Path(__file__).resolve().parents[2]


def inputs(config):
    graph = json.loads(GRAPH_PATH.read_text())
    candidates = json.loads(SPEC_PATH.read_text())["configs"]
    matches = [job for job in graph["jobs"] if job["id"] == config.get("job_id") and job["arrivals"]]
    if (len(matches) != 1 or graph.get("protocol_sha256") != PROTOCOL_SHA256 or
            config.get("protocol_sha256") != PROTOCOL_SHA256):
        raise ValueError("R8 registered target graph identity")
    asset_ref = config["artifact_lock"]
    asset_path = owned_source_path(asset_ref["path"])
    assets = json.loads(verified(asset_path, asset_ref["sha256"], 16 * 1024**2))
    bound = bind_metadata(config["refs"])
    if (assets.get("schema") != "R8_TARGET_ARTIFACT_LOCK_V1" or
            assets.get("code_sha") != config["code_sha"] or
            assets.get("protocol_sha256") != PROTOCOL_SHA256 or
            assets.get("spec_sha256") != graph["spec_sha256"] or
            assets.get("refs") != config["refs"] or
            assets.get("checkpoint_sha256") != bound["docs"]["manifest"]["checkpoint"]["sha256"] or
            set(assets.get("source_jobs", {})) != {job["id"] for job in graph["jobs"] if not job["arrivals"]}):
        raise ValueError("R8 complete source artifact lock identity")
    job = matches[0]
    resolved = resolve(job, graph, candidates, assets["selected_config"])
    registration = bound["docs"]["target"]
    stream_binding = bind(registration)
    rows = stream(registration, job["order"] if job["order"] is not None else 0)
    if job["stage"].startswith("STRESS_"):
        rows = stress(rows, job["stage"].removeprefix("STRESS_"))
    if (len(rows) != job["arrivals"] or
            sum(row["subset"] == "remaining_dev" for row in rows) != job["scored_contents"]):
        raise ValueError("R8 target stream coverage")
    return job, resolved, assets, bound, rows, stream_binding


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_TARGET_JOB_WORK_V1" or
            type(config.get("maximum_seconds")) is not int or config["maximum_seconds"] <= 0):
        raise ValueError("R8 target worker config")
    root = owned_target_path(config["job_root"])
    _reject_recorded_noninfra_failure(root)
    if (root / "online_worker_complete.json").exists():
        raise ValueError("R8 target online worker already complete")
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 target worker wall cap reached")
    closer = None
    try:
        job, resolved, assets, bound, rows, stream_binding = inputs(config)
        _gpu_policy(config["physical_gpu"])
        raw = verified(config["checkpoint_path"], assets["checkpoint_sha256"], 256 * 1024**2)
        if len(raw) != bound["docs"]["manifest"]["checkpoint"]["bytes"]:
            raise ValueError("R8 target checkpoint bytes")
        host, closer = construct(job, resolved, assets, raw, config["refs"], config["code_sha"])
        journal = TargetJournal(root, host, job["id"], rows_sha(rows))
        if (root / "online_complete.json").exists():
            receipt = journal.verified_complete(len(rows))
        else:
            if root.exists():
                journal.recover_once(config.get("resume_failure"))
            else:
                if config.get("resume_failure") is not None:
                    raise ValueError("R8 target recovery has no prior output")
                journal.create()
                _replace(root / "deployment_context.json", json.dumps(host.context, sort_keys=True).encode())
            receipt = run(host, rows, config["target_root"], journal, 256 * 1024**2, guard)
        verified(config["checkpoint_path"], assets["checkpoint_sha256"], 256 * 1024**2)
        if bind_metadata(config["refs"]) != bound:
            raise ValueError("R8 target input metadata changed")
        verified(config["artifact_lock"]["path"], config["artifact_lock"]["sha256"], 16 * 1024**2)
        _replace(root / "online_worker_complete.json", json.dumps(dict(
            schema="R8_TARGET_ONLINE_WORK_COMPLETE_V1", code_sha=config["code_sha"], job=job,
            artifact_lock_sha256=config["artifact_lock"]["sha256"],
            context_sha256=host.context["sha256"], rows_sha256=rows_sha(rows),
            stream_binding=stream_binding,
            online_receipt_sha256=hashlib.sha256((root / "online_complete.json").read_bytes()).hexdigest()),
            sort_keys=True).encode())
        return receipt
    except BaseException as exc:
        if root.is_dir():
            with (root / "worker_failures.jsonl").open("a") as work:
                work.write(json.dumps(dict(stage="online", error_type=type(exc).__name__,
                                           error=str(exc)[:3000]), sort_keys=True) + "\n")
                work.flush()
                os.fsync(work.fileno())
        raise
    finally:
        if closer is not None:
            closer()


if __name__ == "__main__":
    main()
