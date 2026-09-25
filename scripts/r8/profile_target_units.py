"""Bounded target-method timing on a registered source image, without target labels."""
import io
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from dpa_ctta.b1_host import GRATA_COMMIT, Host as GraTaHost
from dpa_ctta.hosts.vptta import VPTTAHost
from dpa_ctta.p1_analysis import evaluate
from dpa_ctta.r7_shared.context import tensor_digest
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r7_source_prep.runner import load_artifact, load_model
from dpa_ctta.r8_ba.context import SOURCE_KEYS, capture
from dpa_ctta.r8_ba.gradient import GradientHost
from dpa_ctta.r8_ba.gradient_calibration import episode as gradient_cal_episode
from dpa_ctta.r8_ba.host import OnlineHost
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.methods import CurrentMLP, build
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.native_host import NativeHost
from dpa_ctta.r8_ba.r7_control import FrozenR7Host
from dpa_ctta.r8_ba.resources import json_size_bound
from dpa_ctta.source_pilot import SourceOnlyHost
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256


def main():
    config = json.loads(Path(os.environ["R8_PROFILE_CONFIG"]).read_text())
    if (config.get("schema") != "R8_TARGET_UNIT_PROFILE_CONFIG_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("maximum_seconds") != 1200 or
            not config.get("r7_artifact_root") or not config.get("r7_inventory_path") or
            not config.get("r7_inventory_sha256")):
        raise ValueError("R8 target profile config")
    started = time.monotonic()
    result = dict(schema="R8_TARGET_UNIT_PROFILE_V1", status="IN_PROGRESS",
                  code_sha=config["code_sha"], physical_gpu=config["physical_gpu"],
                  input="registered source RGB and labels for CPU scoring only; no target image or label", units={})
    COUNTS.clear()

    def guard():
        if time.monotonic() - started >= 1200:
            raise TimeoutError("R8 target profile wall cap")
        if COUNTS["backbone_forwards"] >= 300:
            raise RuntimeError("R8 target profile forward cap")

    def measure(name, sample_units, call):
        guard()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        before = COUNTS.copy()
        start = time.monotonic()
        detail = call()
        torch.cuda.synchronize()
        result["units"][name] = dict(sample_units=sample_units,
                                     seconds=time.monotonic() - start,
                                     counts=dict(COUNTS - before),
                                     peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                     detail=detail)
        records = detail if isinstance(detail, list) else [detail]
        if records and all(isinstance(record, dict) for record in records):
            result.setdefault("record_bytes", {})[name] = max(json_size_bound(record) + 1 for record in records)

    def storage(name, worker, native_memory=False):
        buffer = io.BytesIO()
        torch.save(dict(schema="R8_TARGET_JOURNAL_V1", host=worker.snapshot(), identity={}), buffer)
        result.setdefault("storage", {})[name] = len(buffer.getvalue()) + 4096 + (128 * 1024 if native_memory else 0)
        result.setdefault("context_bytes", {})[name] = json_size_bound(worker.context) + 4096

    try:
        bound = bind_metadata(config["refs"])
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         0.3, config["physical_gpu"], 256 * 1024**2,
                         2 * 1024**3, guard) as (data, segmenter, io_counts):
            image = data.get(sorted(data.folds["fit"])[0], "fit").image
            scaler = []
            with torch.no_grad():
                for name in sorted(data.folds["fit"])[:4]:
                    _, raw, _ = segmenter(data.get(name, "fit").image, observe=True)
                    scaler.append(raw)
            basis64 = torch.eye(1024, dtype=torch.float64)[:, :64]
            basis32 = basis64[:, :32]
            source = {key: "0" * 64 for key in SOURCE_KEYS}
            source["protocol_sha256"] = PROTOCOL_SHA256

            def host(method, method_config, ablation=None):
                method.observer.fit_scaler(torch.stack(scaler), "fit")
                method.freeze()
                expected = capture(segmenter, method, method_config, source)
                return OnlineHost(segmenter, method, method_config, source, expected, ablation)

            a_config = dict(id="A_r32_a0p3", route="A", rank=32, film_amplitude=0.3,
                            observer="R7_dual_codebook", aux_multiplier=1.0)
            b_config = dict(id="B_expanded_global_aux1p0", route="B", rank=64,
                            film_amplitude=0.3, observer="global", aux_multiplier=1.0)
            a = host(build(a_config, basis32), a_config)
            b = host(build(b_config, basis64), b_config)
            long_ista = host(build(b_config, basis64), b_config, "ISTA_20")
            mlp = host(CurrentMLP(basis64, 0.3, "global"), b_config)
            zero_config = dict(id="C0_CURRENT_STATS", film_amplitude=0.3)
            zero = OnlineHost(segmenter, None, zero_config, source,
                              capture(segmenter, None, zero_config, source))
            for name, worker in (("new_a_max", a), ("new_b_max", b),
                                 ("ista20", long_ista), ("mlp", mlp), ("zero", zero)):
                measure(name, 2, lambda worker=worker: [worker.step(image)[1] for _ in range(2)])
                storage(name, worker)

            scale = torch.ones(64, dtype=torch.float64)
            for arm, category in (("B_G1", "gradient_g1"), ("B_G3", "gradient_g3")):
                method = build(b_config, basis64)
                method.observer.fit_scaler(torch.stack(scaler), "fit")
                method.freeze()
                gradient_config = dict(b_config, gradient_arm=arm, gradient_lr=0.001,
                                       scale_sha256=tensor_digest([("scale", scale)]),
                                       grata_commit=GRATA_COMMIT)
                expected = capture(segmenter, method, gradient_config, source)
                worker = GradientHost(segmenter, method, gradient_config, source,
                                      expected, arm, scale, 0.001)
                measure(category, 1, lambda worker=worker: worker.step(image)[1])
                storage(category, worker)
            cal_image = data.get(sorted(data.folds["cal"])[0], "cal").image
            method = build(b_config, basis64)
            method.observer.fit_scaler(torch.stack(scaler), "fit")
            method.freeze()
            lr_config = dict(b_config, gradient_arm="B_G3", gradient_lr=0.001,
                             scale_sha256=tensor_digest([("scale", scale)]),
                             grata_commit=GRATA_COMMIT)
            lr_host = GradientHost(segmenter, method, lr_config, source,
                                   capture(segmenter, method, lr_config, source),
                                   "B_G3", scale, 0.001)
            cal_names = sorted(data.folds["cal"])
            cal_oracles = Oracles("cal", 0.3, torch.zeros(1024, 128),
                                  (tuple(cal_names[:2]),) * 128,
                                  (tuple(cal_names[2:4]),) * 128, ((),) * 128)
            measure("gradient_lr_visit", 4,
                    lambda: gradient_cal_episode(lr_host, data, cal_oracles, 0, 20260924))
            label = data.get(sorted(data.folds["fit"])[0], "fit").label
            result["io_counts"] = dict(io_counts)

        raw = verified(config["checkpoint_path"], config["checkpoint_sha256"], 256 * 1024**2)
        inventory = json.loads(verified(config["r7_inventory_path"],
                                        config["r7_inventory_sha256"], 1024**2))
        r7_segmenter = load_model(raw, device="cuda:0")
        try:
            r7_cost = []
            for mode in ("FULL", "STATIC"):
                r7_host = FrozenR7Host(load_artifact(r7_segmenter, "C_" + mode,
                                        config["r7_artifact_root"], inventory), "R7_C_" + mode,
                                        config["code_sha"], config["r7_inventory_sha256"])
                name = "r7_c_" + mode.lower()
                measure(name, 2, lambda host=r7_host: [host.step(image)[1] for _ in range(2)])
                storage(name, r7_host)
                r7_cost.append(result["units"].pop(name))
            result["units"]["r7_c"] = max(r7_cost, key=lambda row: row["seconds"] / row["sample_units"])
            result["units"]["r7_c"]["detail"] = "slower of exact frozen C_FULL/C_STATIC artifacts"
            for field in ("storage", "context_bytes", "record_bytes"):
                result[field]["r7_c"] = max(result[field]["r7_c_full"], result[field]["r7_c_static"])
        finally:
            r7_segmenter.close()
            del r7_segmenter
            torch.cuda.empty_cache()

        # Target scoring runs only after the online receipt seals and releases its GPU.
        # Source masks exercise the same CPU evaluator without exposing a target label.
        probability = torch.full_like(label, 0.5)
        start_score = time.monotonic()
        score_detail = evaluate(probability, label, "fundus")
        result["units"]["score_visit"] = dict(sample_units=1,
                                                seconds=time.monotonic() - start_score,
                                                counts={}, peak_gpu_bytes=0,
                                                gpu_seconds=0,
                                                detail=score_detail)
        graph = json.loads((Path(__file__).resolve().parents[2] / "docs/review/r8/TASK_GRAPH.static.json").read_text())
        result.setdefault("record_bytes", {})["score_visit"] = json_size_bound(dict(
            visit=1, cycle=1, cycle_visit=1, arm="X" * max(len(job["arm"]) for job in graph["jobs"]),
            order=1, content="0" * 64, domain="X" * 64, subset="remaining_dev", metrics=score_detail)) + 1
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        for arm, name in (("VPTTA_NATIVE", "vptta"), ("C_CTTA_FIXED_LR", "c"),
                          ("G_CTTA_RELEASE_TRANSFER", "g"), ("N_SOURCE_EVAL", "n_source")):
            torch.manual_seed(20260907)
            np.random.seed(20260907)
            if arm == "VPTTA_NATIVE":
                native = VPTTAHost("fundus", source_state=state, device="cuda:0")
            elif arm == "N_SOURCE_EVAL":
                native = SourceOnlyHost("fundus", state, device="cuda:0")
            else:
                native = GraTaHost(arm[0], state=state, device="cuda:0")
            identity = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                            checkpoint_sha256=config["checkpoint_sha256"],
                            registration_sha256=config["refs"]["target"]["sha256"],
                            seed=None if arm == "N_SOURCE_EVAL" else 20260907)
            worker = NativeHost(native, arm, identity)
            count = 6 if name == "vptta" else 2
            measure(name, count, lambda worker=worker: [worker.step(image)[1] for _ in range(count)])
            storage(name, worker, native_memory=name == "vptta")
            worker.close()
            if name in ("c", "g"):
                native.finish(state)
            del worker
            torch.cuda.empty_cache()
        n = result["units"].pop("n_source")
        # N and C0 share the zero-update category; use the slower measured path.
        if n["seconds"] / n["sample_units"] > result["units"]["zero"]["seconds"] / result["units"]["zero"]["sample_units"]:
            result["units"]["zero"] = n
        for field in ("storage", "context_bytes", "record_bytes"):
            result[field]["zero"] = max(result[field]["zero"], result[field]["n_source"])
        result["status"] = "MEASURED_SURROGATES_ONLY"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["physical_counts"] = dict(COUNTS)
        with Path(config["output"]).open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
