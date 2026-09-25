"""Measure R8 source unit costs on registered inputs and one physical GPU."""
import json
import os
import time
from pathlib import Path

import torch

from dpa_ctta.r7_shared.numerics import COUNTS, finite, seg_loss
from dpa_ctta.r7_shared.source import pooled_jacobian, simulate
from dpa_ctta.r8_ba.calibration import Calibrator, validate_episode
from dpa_ctta.r8_ba.capacity import capacity_one
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.methods import build
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.schedule import anchors
from dpa_ctta.r8_ba.trainer import SourceTrainer


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
                    call()
                    torch.cuda.synchronize()
                    result["units"][name] = dict(sample_units=sample_units,
                                                  seconds=time.monotonic() - start,
                                                  counts=dict(COUNTS - before),
                                                  peak_gpu_bytes=torch.cuda.max_memory_allocated())

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
                method_config = dict(id="B_expanded_global_aux1p0", route="B", rank=64,
                                     film_amplitude=0.3, observer="global", aux_multiplier=1.0)

                def new_method():
                    method = build(method_config, basis)
                    method.observer.fit_scaler(torch.stack(scaler_rows), "fit")
                    return method

                trainer = SourceTrainer(segmenter, new_method(), data, fit, 20260924,
                                        "R8_PROFILE_SYNTHETIC_ORACLE")
                measure("source_fit_step", 1, trainer.fit_step)
                calibrator = Calibrator(segmenter, new_method(), data, cal,
                                        "R8_PROFILE_SYNTHETIC_ORACLE")
                measure("source_cal_step", 1, calibrator.step)
                online_method = new_method()
                online_method.freeze()
                measure("source_val_visit", 32,
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
