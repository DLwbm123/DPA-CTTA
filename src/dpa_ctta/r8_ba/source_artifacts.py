"""Read completed source jobs without constructing models or opening target data."""
import hashlib
import json
from pathlib import Path

from .protocol import PROTOCOL_SHA256
from .source_select import select_grid
from .trainer import SAVE_STEPS
from .validation_journal import load_validation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_completed(root, job, config, code_sha, spec_sha256):
    root = Path(root)
    marker = json.loads((root / "worker_complete.json").read_text())
    payload = marker.get("binding_payload", {})
    binding = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    if (marker.get("schema") != "R8_SOURCE_JOB_WORK_COMPLETE_V1" or
            marker.get("binding") != binding or payload.get("job") != job or
            payload.get("config") != config or payload.get("code_sha") != code_sha or
            payload.get("graph_spec_sha256") != spec_sha256 or
            marker.get("fit_receipt_sha256") != digest(root / "fit_complete.json")):
        raise ValueError("R8 completed source job identity")
    fit = json.loads((root / "fit_complete.json").read_text())
    if (fit.get("binding") != binding or fit.get("source_seed") != job["source_seed"] or
            fit.get("steps") != 16000 or fit.get("physical_sha256") != digest(root / "physical.jsonl")):
        raise ValueError("R8 completed source fit identity")
    rows, artifacts = {}, {}
    for step in SAVE_STEPS:
        selected = root / f"selected.{step}.pt"
        if fit["selected_sha256"].get(str(step)) != digest(selected):
            raise ValueError("R8 selected source archive changed")
        if job["stage"] == "SOURCE_MLP":
            artifact, artifact_sha = selected, fit["selected_sha256"][str(step)]
        else:
            cal = root / f"calibration.{step}"
            receipt = json.loads((cal / "cal_complete.json").read_text())
            artifact, artifact_sha = cal / "calibrated.pt", receipt.get("artifact_sha256")
            if (marker["calibration_receipt_sha256"].get(str(step)) != digest(cal / "cal_complete.json") or
                    receipt.get("binding") != binding or receipt.get("source_step") != step or
                    receipt.get("selected_sha256") != fit["selected_sha256"][str(step)] or
                    receipt.get("steps") != 1024 or artifact_sha != digest(artifact) or
                    receipt.get("physical_sha256") != digest(cal / "physical.jsonl")):
                raise ValueError("R8 calibrated source archive changed")
        val = root / f"validation.{step}" / "val_complete.json"
        if marker["validation_receipt_sha256"].get(str(step)) != digest(val):
            raise ValueError("R8 source validation marker changed")
        rows[step] = load_validation(root, step, artifact_sha, binding)
        artifacts[step] = dict(path=str(artifact), sha256=artifact_sha)
    return dict(binding=binding, marker_sha256=digest(root / "worker_complete.json"),
                rows=rows, artifacts=artifacts)


def select_completed_grid(graph, configs, roots, code_sha, cost_by_config, cost_evidence):
    jobs = [row for row in graph["jobs"] if row["stage"] == "SOURCE_GRID"]
    if (len(jobs) != 48 or set(roots) != {job["id"] for job in jobs} or
            graph.get("protocol_sha256") != PROTOCOL_SHA256 or
            not isinstance(cost_evidence, dict) or not cost_evidence):
        raise ValueError("R8 complete discovery jobs and cost evidence required")
    candidates = {config["id"]: config for config in configs}
    rows, evidence = {}, {}
    for job in jobs:
        result = read_completed(roots[job["id"]], job, candidates[job["config"]],
                                code_sha, graph["spec_sha256"])
        rows[job["config"], job["mode"], job["source_seed"]] = result.pop("rows")
        evidence[job["id"]] = result
    selection = select_grid(configs, rows, cost_by_config)
    selection.update(code_sha=code_sha, graph_spec_sha256=graph["spec_sha256"],
                     source_jobs=evidence, cost_evidence=cost_evidence)
    return selection
