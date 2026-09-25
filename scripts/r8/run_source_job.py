"""Run one frozen R8 source job through fit, calibration, and source validation."""
import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import torch

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.calibration import Calibrator
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.journal import SourceJournal, _replace
from dpa_ctta.r8_ba.methods import CurrentMLP, build, from_selected
from dpa_ctta.r8_ba.oracle_journal import load_oracles
from dpa_ctta.r8_ba.paths import owned_source_path
from dpa_ctta.r8_ba.preparation import load_bases
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.resources import CAPS
from dpa_ctta.r8_ba.scaler_journal import load_scaler
from dpa_ctta.r8_ba.source_run import run_calibration, run_fit
from dpa_ctta.r8_ba.trainer import SAVE_STEPS, SourceTrainer
from dpa_ctta.r8_ba.validation_journal import load_validation, run_validation

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _marker(root, schema, identity, receipt_name, hash_key):
    root = Path(root)
    marker = json.loads((root / "worker_complete.json").read_text())
    receipt_hash = digest(root / receipt_name)
    if (marker.get("schema") != schema or marker.get("identity") != identity or
            marker.get(hash_key) != receipt_hash):
        raise ValueError("R8 source prerequisite worker/receipt identity")
    return receipt_hash


def _job_and_config(config):
    graph = json.loads((ROOT / "docs/review/r8/TASK_GRAPH.static.json").read_text())
    spec = json.loads((ROOT / "docs/review/r8/input/R8_EXPERIMENT_SPEC.json").read_text())
    if (graph.get("schema") != "R8_STATIC_TASK_GRAPH_V1" or
            graph.get("source_training_jobs") != 65 or graph.get("target_jobs") != 724 or
            graph.get("protocol_sha256") != PROTOCOL_SHA256):
        raise ValueError("R8 full frozen task graph")
    matches = [row for row in graph["jobs"] if row["id"] == config.get("job_id")]
    if len(matches) != 1 or matches[0]["stage"] not in ("SOURCE_GRID", "SOURCE_FINAL", "SOURCE_MLP"):
        raise ValueError("R8 registered source job")
    job = matches[0]
    candidate = config.get("config")
    route = "B" if job["stage"] == "SOURCE_MLP" else job["arm"]
    if candidate not in spec["configs"] or candidate["route"] != route:
        raise ValueError("R8 listed source configuration")
    if job["stage"] == "SOURCE_GRID":
        if candidate["id"] != job["config"] or config.get("selection_path") is not None:
            raise ValueError("R8 discovery config identity")
        selection_sha256 = None
    else:
        selection_path = config.get("selection_path")
        if not selection_path:
            raise ValueError("R8 source-only selection required")
        selection = json.loads(Path(selection_path).read_text())
        route = "B" if job["stage"] == "SOURCE_MLP" else job["arm"]
        if (selection.get("schema") != "R8_SOURCE_ONLY_GRID_SELECTION_V1" or
                selection.get("protocol_sha256") != PROTOCOL_SHA256 or
                selection.get("selected_config", {}).get(route) != candidate["id"]):
            raise ValueError("R8 source-selected config identity")
        selection_sha256 = digest(selection_path)
    if (job["source_seed"] != config.get("source_seed") or
            job["mode"] != config.get("mode") or job["fit_steps"] != 16000):
        raise ValueError("R8 source seed/mode identity")
    return graph, job, candidate, selection_sha256


def _complete_fit(root, trainer):
    receipt = json.loads((root / "fit_complete.json").read_text())
    journal = SourceJournal(root, trainer)
    if (receipt.get("schema") != "R8_SOURCE_FIT_COMPLETE_V1" or
            receipt.get("binding") != trainer.binding or
            receipt.get("source_seed") != trainer.source_seed or
            receipt.get("steps") != 16000 or
            receipt.get("physical_sha256") != digest(root / "physical.jsonl")):
        raise ValueError("R8 fit completion identity")
    for step in SAVE_STEPS:
        journal.selected(step)
        if receipt["selected_sha256"].get(str(step)) != digest(root / f"selected.{step}.pt"):
            raise ValueError("R8 selected source snapshot receipt")
    return receipt, journal


def _complete_calibration(root, step, calibrator, receipt):
    cal_root = root / f"calibration.{step}"
    identity = dict(binding=calibrator.binding, source_step=step,
                    selected_sha256=receipt["selected_sha256"][str(step)])
    row = json.loads((cal_root / "cal_complete.json").read_text())
    raw = (cal_root / "calibrated.pt").read_bytes()
    if (row.get("schema") != "R8_CAL_COMPLETE_V1" or
            any(row.get(key) != value for key, value in identity.items()) or
            row.get("steps") != 1024 or row.get("artifact_sha256") != hashlib.sha256(raw).hexdigest() or
            row.get("physical_sha256") != digest(cal_root / "physical.jsonl")):
        raise ValueError("R8 calibration completion identity")
    artifact = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (artifact.get("schema") != "R8_CALIBRATED_METHOD_V1" or
            artifact.get("identity") != identity or
            artifact.get("method_digest") != row["method_digest"]):
        raise ValueError("R8 calibrated source method identity")
    calibrator.method.load_state_dict(artifact["method"], strict=True)
    calibrator.method.freeze()
    if calibrator.method.digest() != row["method_digest"]:
        raise ValueError("R8 calibrated source weights")
    return row


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_SOURCE_JOB_WORK_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            re.fullmatch(r"[0-9a-f]{40}", config.get("code_sha", "")) is None or
            config.get("protocol_sha256") != PROTOCOL_SHA256 or
            type(config.get("maximum_seconds")) is not int or
            not 0 < config["maximum_seconds"] <= CAPS["gpu_seconds"]):
        raise ValueError("R8 source job worker config")
    graph, job, candidate, selection_sha256 = _job_and_config(config)
    started = time.monotonic()
    root = owned_source_path(config["job_root"])
    if (root / "worker_complete.json").exists():
        raise ValueError("R8 source job already completed")
    bound = bind_metadata(config["refs"])
    amplitude = candidate["film_amplitude"]
    common = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                  refs=config["refs"],
                  checkpoint_sha256=bound["docs"]["manifest"]["checkpoint"]["sha256"])
    oracle_identity = dict(**common, amplitude=amplitude)
    oracle_root = owned_source_path(config["oracle_root"])
    oracle_receipt_sha256 = _marker(oracle_root, "R8_ORACLE_WORK_COMPLETE_V1",
                                    oracle_identity, "oracle_complete.json", "oracle_receipt_sha256")
    bases_root = owned_source_path(config["bases_root"])
    bases_identity = dict(**oracle_identity, oracle_receipt_sha256=oracle_receipt_sha256)
    bases_receipt_sha256 = _marker(bases_root, "R8_BASES_WORK_COMPLETE_V1",
                                   bases_identity, "bases_complete.json", "bases_receipt_sha256")
    scaler_root = owned_source_path(config["scaler_root"])
    scaler_identity = dict(**common, fold="fit", zero_film=True)
    scaler_receipt_sha256 = _marker(scaler_root, "R8_SCALER_WORK_COMPLETE_V1",
                                    scaler_identity, "scaler_complete.json", "scaler_receipt_sha256")
    binding_payload = dict(schema="R8_SOURCE_JOB_BINDING_V1", code_sha=config["code_sha"],
                           graph_spec_sha256=graph["spec_sha256"], job=job, config=candidate,
                           selection_sha256=selection_sha256,
                           oracle_receipt_sha256=oracle_receipt_sha256,
                           bases_receipt_sha256=bases_receipt_sha256,
                           scaler_receipt_sha256=scaler_receipt_sha256)
    binding = hashlib.sha256(json.dumps(binding_payload, sort_keys=True).encode()).hexdigest()
    oracle_binding = hashlib.sha256(json.dumps(oracle_identity, sort_keys=True).encode()).hexdigest()
    scaler_binding = hashlib.sha256(json.dumps(scaler_identity, sort_keys=True).encode()).hexdigest()
    failure = config.get("resume_failure")
    recovered_stage = None

    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 source job wall cap reached")

    try:
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         amplitude, config["physical_gpu"], 256 * 1024**2,
                         2 * 1024**3, guard) as (data, segmenter, io_counts):
            oracles = load_oracles(oracle_root, data, oracle_binding, amplitude)
            bases = load_bases(bases_root, bases_identity)
            scaler = load_scaler(scaler_root, data, scaler_binding)
            basis = bases[("B_basis" if candidate["route"] == "B" else "A_basis")][candidate["rank"]]
            if job["stage"] == "SOURCE_MLP":
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(job["source_seed"])
                    method = CurrentMLP(basis, amplitude, candidate["observer"])
            else:
                method = build(candidate, basis, static=job["mode"] == "STATIC",
                               seed=job["source_seed"])
            method.observer.fit_scaler(scaler, "fit")
            COUNTS.clear()
            trainer = SourceTrainer(segmenter, method, data, oracles["fit"],
                                    job["source_seed"], binding)
            if not (root / "fit_complete.json").exists():
                resume_fit = root.exists()
                if resume_fit and failure is None:
                    raise ValueError("R8 incomplete fit requires evidenced recovery")
                run_fit(trainer, root, guard, failure if resume_fit else None)
                if resume_fit:
                    recovered_stage = "fit"
            fit_receipt, fit_journal = _complete_fit(root, trainer)
            val_hashes = {}
            cal_hashes = {}
            for step in SAVE_STEPS:
                guard()
                selected = fit_journal.selected(step)
                selected_method = from_selected(selected, candidate, basis,
                                                job["mode"] == "STATIC" if job["stage"] != "SOURCE_MLP" else False,
                                                job["source_seed"], binding,
                                                mlp=job["stage"] == "SOURCE_MLP")
                if job["stage"] == "SOURCE_MLP":
                    selected_method.freeze()
                    deployed = selected_method
                    artifact_sha = fit_receipt["selected_sha256"][str(step)]
                else:
                    calibrator = Calibrator(segmenter, selected_method, data, oracles["cal"], binding)
                    cal_root = root / f"calibration.{step}"
                    if not (cal_root / "cal_complete.json").exists():
                        resume_cal = cal_root.exists()
                        if resume_cal and (failure is None or recovered_stage is not None):
                            raise ValueError("R8 incomplete calibration requires one evidenced recovery")
                        run_calibration(calibrator, root, step, guard,
                                        failure if resume_cal else None)
                        if resume_cal:
                            recovered_stage = "calibration"
                    row = _complete_calibration(root, step, calibrator, fit_receipt)
                    deployed, artifact_sha = calibrator.method, row["artifact_sha256"]
                    cal_hashes[str(step)] = digest(cal_root / "cal_complete.json")
                val_root = root / f"validation.{step}"
                if not (val_root / "val_complete.json").exists():
                    resume_val = val_root.exists()
                    if resume_val and (failure is None or recovered_stage is not None):
                        raise ValueError("R8 incomplete validation requires one evidenced recovery")
                    run_validation(root, step, artifact_sha, binding, segmenter, deployed,
                                   data, oracles["val"], guard,
                                   failure if resume_val else None)
                    if resume_val:
                        recovered_stage = "validation"
                load_validation(root, step, artifact_sha, binding)
                val_hashes[str(step)] = digest(val_root / "val_complete.json")
            if failure is not None and recovered_stage is None:
                raise ValueError("R8 unused recovery request")
        _replace(root / "worker_complete.json", json.dumps(dict(
            schema="R8_SOURCE_JOB_WORK_COMPLETE_V1", binding=binding,
            binding_payload=binding_payload, fit_receipt_sha256=digest(root / "fit_complete.json"),
            calibration_receipt_sha256=cal_hashes, validation_receipt_sha256=val_hashes,
            source_io_counts=dict(io_counts), recovered_stage=recovered_stage),
            sort_keys=True, allow_nan=False).encode())
    except BaseException as exc:
        if root.is_dir():
            with (root / "worker_failures.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)[:3000],
                                             binding=binding), sort_keys=True) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        raise


if __name__ == "__main__":
    main()
