"""Read completed source jobs without constructing models or opening target data."""
import hashlib
import json
from pathlib import Path

from .protocol import PROTOCOL_SHA256
from .calibration import select_checkpoint
from .preparation import load_bases
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
            uncal = root / f"validation_uncalibrated.{step}" / "val_complete.json"
            if marker["uncalibrated_validation_receipt_sha256"].get(str(step)) != digest(uncal):
                raise ValueError("R8 uncalibrated validation marker changed")
            load_validation(root, step, fit["selected_sha256"][str(step)], binding, calibrated=False)
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


def source_index(graph, configs, selection, roots, code_sha, refs, checkpoint_sha256,
                 bases_by_amplitude, r7_inventory):
    """Freeze the five-seed model choices after all source producers complete."""
    jobs = [row for row in graph["jobs"] if not row["arrivals"]]
    if (len(jobs) != 65 or set(roots) != {row["id"] for row in jobs} or
            selection.get("code_sha") != code_sha or selection.get("protocol_sha256") != PROTOCOL_SHA256 or
            selection.get("graph_spec_sha256") != graph["spec_sha256"]):
        raise ValueError("R8 complete source index identity")
    candidates = {row["id"]: row for row in configs}
    completed, chosen = {}, {}
    for job in jobs:
        route = "B" if job["stage"] == "SOURCE_MLP" else job["arm"]
        config_id = job["config"] if job["stage"] == "SOURCE_GRID" else selection["selected_config"][route]
        candidate = candidates[config_id]
        result = read_completed(roots[job["id"]], job, candidate, code_sha, graph["spec_sha256"])
        if job["stage"] == "SOURCE_GRID" and (
                selection["source_jobs"][job["id"]]["marker_sha256"] != result["marker_sha256"]):
            raise ValueError("R8 discovery source changed after selection")
        completed[job["id"]], chosen[job["id"]] = result, candidate
    mlp_pair = [next(job for job in jobs if job["stage"] == "SOURCE_MLP" and job["source_seed"] == seed)
                for seed in (20260924, 20260925)]
    mlp_step, mlp_scores = select_checkpoint({step: tuple(completed[job["id"]]["rows"][step] for job in mlp_pair)
                                             for step in SAVE_STEPS})
    basis_cache, index = {}, {}
    for job in jobs:
        candidate, result = chosen[job["id"]], completed[job["id"]]
        step = mlp_step if job["stage"] == "SOURCE_MLP" else selection["per_config_mode"][candidate["id"]][job["mode"]]["source_step"]
        marker = json.loads((Path(roots[job["id"]]) / "worker_complete.json").read_text())
        payload = marker["binding_payload"]
        row = dict(config=candidate, source_seed=job["source_seed"], mode=job["mode"],
                   source_step=step, artifact=result["artifacts"][step], binding=result["binding"],
                   worker_marker_sha256=result["marker_sha256"],
                   oracle_receipt_sha256=payload["oracle_receipt_sha256"],
                   bases_receipt_sha256=payload["bases_receipt_sha256"])
        if candidate["route"] == "B":
            amplitude = str(candidate["film_amplitude"])
            entry = bases_by_amplitude[amplitude]
            if amplitude not in basis_cache:
                basis_cache[amplitude] = load_bases(entry["root"], entry["identity"])
            if digest(Path(entry["root"]) / "bases_complete.json") != row["bases_receipt_sha256"]:
                raise ValueError("R8 source index gradient basis identity")
            row["gradient_scale"] = basis_cache[amplitude]["gradient_scale"][candidate["rank"]].tolist()
        index[job["id"]] = row
    return dict(schema="R8_SOURCE_ASSET_INDEX_V1", code_sha=code_sha, protocol_sha256=PROTOCOL_SHA256,
                spec_sha256=graph["spec_sha256"], refs=refs, checkpoint_sha256=checkpoint_sha256,
                selected_config=selection["selected_config"], source_jobs=index, r7_inventory=r7_inventory,
                mlp_selection=dict(source_step=mlp_step, scores=mlp_scores))


def lock_targets(index, index_sha256, lr_root, capacity_roots):
    """Seal the deployment index only after complete source-only LR/capacity stages."""
    if index.get("schema") != "R8_SOURCE_ASSET_INDEX_V1" or set(capacity_roots) != {"0.1", "0.3"}:
        raise ValueError("R8 source artifact lock prerequisites")
    root = Path(lr_root)
    marker = json.loads((root / "worker_complete.json").read_text())
    selection = json.loads((root / "lr_selection.json").read_text())
    receipt = json.loads((root / "complete.json").read_text())
    identity = selection.get("identity", {})
    if (marker.get("schema") != "R8_GRADIENT_LR_WORK_COMPLETE_V1" or
            selection.get("schema") != "R8_GRADIENT_LR_SELECTION_V1" or
            marker.get("identity") != identity or identity.get("code_sha") != index["code_sha"] or
            identity.get("source_index_sha256") != index_sha256 or
            identity.get("protocol_sha256") != PROTOCOL_SHA256 or
            marker.get("receipt_sha256") != digest(root / "complete.json") or
            marker.get("selection_sha256") != digest(root / "lr_selection.json") or
            receipt.get("identity") != identity or receipt.get("completed") != 1152 or
            receipt.get("results_sha256") != digest(root / "results.json") or
            receipt.get("physical_sha256") != digest(root / "physical.jsonl")):
        raise ValueError("R8 gradient LR completion identity")
    from .gradient_calibration import select
    recomputed = select([row["result"] for row in json.loads((root / "results.json").read_text())])
    if recomputed["selected_lr"] != selection.get("selected_lr"):
        raise ValueError("R8 selected gradient LR differs from source-only results")
    capacity = {}
    for amplitude, directory in capacity_roots.items():
        directory = Path(directory)
        done = json.loads((directory / "worker_complete.json").read_text())
        probe = json.loads((directory / "complete.json").read_text())
        producer = done.get("identity", {})
        if (done.get("schema") != "R8_CAPACITY_WORK_COMPLETE_V1" or
                producer.get("code_sha") != index["code_sha"] or
                producer.get("protocol_sha256") != PROTOCOL_SHA256 or
                producer.get("refs") != index["refs"] or
                producer.get("amplitude") != float(amplitude) or
                done.get("receipt_sha256") != digest(directory / "complete.json") or
                probe.get("identity") != producer or probe.get("completed") != 256 or
                probe.get("results_sha256") != digest(directory / "results.json") or
                probe.get("physical_sha256") != digest(directory / "physical.jsonl")):
            raise ValueError("R8 capacity completion identity")
        capacity[amplitude] = digest(directory / "worker_complete.json")
    return dict(index, schema="R8_TARGET_ARTIFACT_LOCK_V1", source_index_sha256=index_sha256,
                gradient_lr_selection=recomputed["selected_lr"],
                gradient_lr_worker_sha256=digest(root / "worker_complete.json"),
                capacity_worker_sha256=capacity)
