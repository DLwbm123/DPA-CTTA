"""Measure R8 source unit costs on registered inputs and one physical GPU."""
import json
import io
import os
import time
from pathlib import Path

import torch

from dpa_ctta.r7_shared.numerics import COUNTS, finite, seg_loss
from dpa_ctta.r7_shared.source import pooled_jacobian, simulate
from dpa_ctta.r8_ba.calibration import Calibrator, validate_episode
from dpa_ctta.r8_ba.capacity import capacity_one
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.methods import CurrentMLP, build
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.schedule import anchors
from dpa_ctta.r8_ba.trainer import SourceTrainer
from dpa_ctta.r8_ba.resources import json_size_bound


def main():
    config = json.loads(Path(os.environ["R8_PROFILE_CONFIG"]).read_text())
    if (config.get("schema") != "R8_SOURCE_UNIT_PROFILE_CONFIG_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("maximum_seconds") != 1200):
        raise ValueError("R8 source profile config")
    started = time.monotonic()

    def guard():
        if time.monotonic() - started >= 1200:
            raise TimeoutError("R8 source profile wall cap")
        if COUNTS["backbone_forwards"] >= 2000:
            raise RuntimeError("R8 source profile forward cap")

    result = dict(schema="R8_SOURCE_UNIT_PROFILE_V1", status="IN_PROGRESS",
                  code_sha=config["code_sha"], physical_gpu=config["physical_gpu"], units={})
    COUNTS.clear()
    try:
        bound = bind_metadata(config["refs"])
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         0.3, config["physical_gpu"], 256 * 1024**2,
                         2 * 1024**3, guard) as (data, segmenter, io_counts):
            hook = segmenter.register_forward_pre_hook(lambda *_: guard())
            try:
                def measure(name, sample_units, call):
                    guard()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    before = COUNTS.copy()
                    start = time.monotonic()
                    detail = call()
                    torch.cuda.synchronize()
                    row = dict(sample_units=sample_units, seconds=time.monotonic() - start,
                               counts=dict(COUNTS - before),
                               peak_gpu_bytes=torch.cuda.max_memory_allocated())
                    result.setdefault("observations", {}).setdefault(name, []).append(row)
                    previous = result["units"].get(name)
                    if previous is not None:
                        if previous["sample_units"] != sample_units:
                            raise ValueError("R8 profile variant sample units")
                        row = dict(sample_units=sample_units,
                                   seconds=max(previous["seconds"], row["seconds"]),
                                   peak_gpu_bytes=max(previous["peak_gpu_bytes"], row["peak_gpu_bytes"]),
                                   counts={key: max(previous["counts"].get(key, 0), row["counts"].get(key, 0))
                                           for key in previous["counts"].keys() | row["counts"].keys()})
                    result["units"][name] = row
                    records = detail if isinstance(detail, list) else [detail]
                    if records and all(isinstance(record, dict) for record in records):
                        per_unit = {key: value // sample_units for key, value in row["counts"].items()}
                        bound = max(json_size_bound(dict(record, counts=per_unit)) + 1 for record in records)
                        result.setdefault("record_bytes", {})[name] = max(
                            result.get("record_bytes", {}).get(name, 0), bound)

                def storage(name, payload):
                    buffer = io.BytesIO()
                    torch.save(payload, buffer)
                    result.setdefault("storage", {})[name] = max(
                        result.get("storage", {}).get(name, 0), len(buffer.getvalue()) + 4096)

                names = {fold: sorted(data.folds[fold]) for fold in ("fit", "cal", "val")}
                def synthetic_oracles(fold):
                    pair, query = tuple(names[fold][:2]), tuple(names[fold][2:4])
                    size = {"fit": 512, "cal": 128, "val": 128}[fold]
                    return Oracles(fold, 0.3, torch.zeros(1024, size),
                                   (pair,) * size, (query,) * size, ((),) * size)

                fit, cal, val = (synthetic_oracles(fold) for fold in ("fit", "cal", "val"))
                row = data.get(names["val"][0], "val")
                image = simulate(row.image, anchors("val")[1], "R8_PROFILE_ORACLE_SUPPORT")
                optimizer_v = torch.zeros(1024, requires_grad=True)
                optimizer = torch.optim.Adam([optimizer_v], lr=0.03,
                                             betas=(0.9, 0.999), eps=1e-8, weight_decay=0)

                def oracle_steps():
                    for _ in range(16):
                        optimizer.zero_grad(set_to_none=True)
                        loss = (seg_loss(segmenter(image, optimizer_v), row.label) +
                                seg_loss(segmenter(image, optimizer_v), row.label)) / 2
                        loss = loss + 1e-3 * optimizer_v.square().mean()
                        finite(loss)
                        loss.backward()
                        COUNTS["source_backward_calls"] += 1
                        optimizer.step()
                        COUNTS["source_Adam"] += 1
                        finite(optimizer_v)

                measure("oracle_step", 16, oracle_steps)
                measure("oracle_query", 1,
                        lambda: segmenter(image, optimizer_v.detach()))
                basis = torch.eye(1024, dtype=torch.float64)[:, :64]
                measure("capacity_step", 128,
                        lambda: capacity_one(segmenter, data, val, basis, "B64", 1))
                scaler_rows = []

                def observe_scaler():
                    with torch.no_grad():
                        for name in names["fit"][:4]:
                            _, raw, _ = segmenter(data.get(name, "fit").image, observe=True)
                            scaler_rows.append(raw)

                measure("scaler_observation", 4, observe_scaler)
                measure("basis_vjp", 32,
                        lambda: pooled_jacobian(segmenter, data, names["fit"][:1]))
                # A has three backbone forwards per visit; B and MLP have two.
                # Measure each family rather than treating the B64 step as universal.
                for route, observation in (("a", "R7_dual_codebook"), ("b", "global"),
                                           ("b", "current_tokens"), ("mlp", "global"),
                                           ("mlp", "current_tokens")):
                    rank = 32 if route == "a" else 64
                    method_config = dict(id=f"PROFILE_{route}", route=route.upper(), rank=rank,
                                         film_amplitude=0.3, observer=observation, aux_multiplier=1.0)

                    def new_method():
                        method = (CurrentMLP(basis, 0.3, observation) if route == "mlp" else
                                  build(method_config, basis[:, :rank]))
                        method.observer.fit_scaler(torch.stack(scaler_rows), "fit")
                        return method

                    trainer = SourceTrainer(segmenter, new_method(), data, fit, 20260924,
                                            "R8_PROFILE_SYNTHETIC_ORACLE")
                    # Four consecutive chunks cover a complete recurrent episode.
                    measure(f"source_fit_{route}_step", 4,
                            lambda: [trainer.fit_step() for _ in range(4)])
                    storage(f"fit_{route}", dict(schema="R8_SOURCE_JOURNAL_V1", snapshot=trainer.snapshot()))
                    if route != "mlp":
                        calibrator = Calibrator(segmenter, new_method(), data, cal,
                                                "R8_PROFILE_SYNTHETIC_ORACLE")
                        measure(f"source_cal_{route}_step", 1, calibrator.step)
                        storage(f"cal_{route}", dict(schema="R8_CAL_JOURNAL_V1", identity={},
                                                    snapshot=calibrator.snapshot()))
                        calibrator.method.freeze()
                        storage(f"method_{route}", dict(schema="R8_CALIBRATED_METHOD_V1", identity={},
                            method_digest=calibrator.method.digest(), method=calibrator.method.state_dict()))
                    online_method = new_method()
                    online_method.freeze()
                    measure(f"source_val_{route}_visit", 32,
                            lambda: validate_episode(segmenter, online_method, data, val, 0))
                result["io_counts"] = dict(io_counts)
            finally:
                hook.remove()
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
