"""Bounded real-GPU preflight for R8 backbone and maximum-rank source fit."""
import json
import os
import time
from pathlib import Path

import torch

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.methods import build
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.trainer import SourceTrainer


def main():
    config = json.loads(Path(os.environ["R8_PROFILE_CONFIG"]).read_text())
    if (config.get("schema") != "R8_CRITICAL_PROFILE_CONFIG_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config.get("physical_gpu") != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("maximum_seconds") != 1200):
        raise ValueError("R8 bounded critical profile config")
    started = time.monotonic()

    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise TimeoutError("R8 critical profile wall cap")
        if COUNTS["backbone_forwards"] >= 100:
            raise RuntimeError("R8 critical profile forward cap")

    result = dict(schema="R8_CRITICAL_PROFILE_V1", code_sha=config["code_sha"],
                  physical_gpu=config["physical_gpu"],
                  basis="synthetic rank64 memory and timing surrogate; not source-fit basis qualification")
    COUNTS.clear()
    try:
        refs = config["refs"]
        bound = bind_metadata(refs)
        result["source_metadata_groups"] = bound["audit"]["groups"]
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         0.3, config["physical_gpu"], 256 * 1024**2,
                         2 * 1024**3, guard) as (data, segmenter, io_counts):
            names = sorted(data.folds["fit"])
            torch.cuda.reset_peak_memory_stats()
            before = COUNTS.copy()
            start = time.monotonic()
            with torch.no_grad():
                segmenter(data.get(names[0], "fit").image)
            torch.cuda.synchronize()
            result["zero_forward"] = dict(seconds=time.monotonic() - start,
                                          counts=dict(COUNTS - before),
                                          peak_gpu_bytes=torch.cuda.max_memory_allocated())
            scaler = []
            with torch.no_grad():
                for name in names[:4]:
                    _, raw, _ = segmenter(data.get(name, "fit").image, observe=True)
                    scaler.append(raw)
            basis = torch.eye(1024, dtype=torch.float64)[:, :64]
            config_b = dict(id="B_expanded_global_aux1p0", route="B", rank=64,
                            film_amplitude=0.3, observer="global", aux_multiplier=1.0)
            method = build(config_b, basis)
            method.observer.fit_scaler(torch.stack(scaler), "fit")
            support, query = tuple(names[:2]), tuple(names[2:4])
            oracles = Oracles("fit", 0.3, torch.zeros(1024, 512),
                              (support,) * 512, (query,) * 512, ((),) * 512)
            trainer = SourceTrainer(segmenter, method, data, oracles, 20260924,
                                    "R8_PROFILE_SYNTHETIC_ORACLE")
            torch.cuda.reset_peak_memory_stats()
            before = COUNTS.copy()
            start = time.monotonic()
            trace = trainer.fit_step()
            torch.cuda.synchronize()
            result["max_rank_source_fit_step"] = dict(
                seconds=time.monotonic() - start, counts=dict(COUNTS - before),
                peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                optimizer_step=trace["step"], query_visits=trace["query_visits"])
            result["io_counts"] = dict(io_counts)
        result["status"] = "MEASURED_SURROGATE_ONLY"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        result["physical_counts"] = dict(COUNTS)
        path = Path(config["output"])
        with path.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
