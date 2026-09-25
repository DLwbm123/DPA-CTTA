"""Bounded real target-RGB read, R8 B64 inference, and 50-visit journal profile."""
import io
import json
import os
import time
from pathlib import Path

import torch

from dpa_ctta.integrations.ctta_suite import build_reference_model
from dpa_ctta.r3.plan import stream
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r8_ba.context import SOURCE_KEYS, capture
from dpa_ctta.r8_ba.host import OnlineHost
from dpa_ctta.r8_ba.inputs import _gpu_policy, bind_metadata
from dpa_ctta.r8_ba.journal import TargetJournal
from dpa_ctta.r8_ba.methods import R8Segmenter, build
from dpa_ctta.r8_ba.online import run
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.streams import rows_sha


def main():
    config = json.loads(Path(os.environ["R8_PROFILE_CONFIG"]).read_text())
    if (config.get("schema") != "R8_ONLINE_IO_PROFILE_CONFIG_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("maximum_seconds") != 1200):
        raise ValueError("R8 online IO profile config")
    _gpu_policy(config["physical_gpu"])
    started = time.monotonic()

    def guard():
        if time.monotonic() - started >= 1200:
            raise TimeoutError("R8 online IO profile wall cap")
        if COUNTS["backbone_forwards"] >= 150:
            raise RuntimeError("R8 online IO profile forward cap")

    result = dict(schema="R8_ONLINE_IO_PROFILE_V1", status="IN_PROGRESS",
                  code_sha=config["code_sha"], physical_gpu=config["physical_gpu"],
                  arm="UNTRAINED_B64_PROFILE_ONLY", target_labels_read=0)
    COUNTS.clear()
    segmenter = None
    try:
        bound = bind_metadata(config["refs"])
        rows = stream(bound["docs"]["target"], 0)[:50]
        raw = verified(config["checkpoint_path"], config["checkpoint_sha256"], 256 * 1024**2)
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        model, _ = build_reference_model("fundus")
        model.load_state_dict(state, strict=True)
        segmenter = R8Segmenter(model, 0.3, device="cuda:0")
        torch.manual_seed(20260924)
        method_config = dict(id="B_expanded_global_aux1p0", route="B", rank=64,
                             film_amplitude=0.3, observer="global", aux_multiplier=1.0)
        method = build(method_config, torch.eye(1024, dtype=torch.float64)[:, :64])
        method.observer.fit_scaler(torch.randn(4, 134), "fit")
        method.freeze()
        source = {key: "0" * 64 for key in SOURCE_KEYS}
        source["protocol_sha256"] = PROTOCOL_SHA256
        context = capture(segmenter, method, method_config, source)
        host = OnlineHost(segmenter, method, method_config, source, context)
        journal = TargetJournal(config["job_root"], host, "R8_PROFILE_IO_NOT_SCIENTIFIC_JOB",
                                rows_sha(rows))
        journal.create()
        hook = segmenter.register_forward_pre_hook(lambda *_: guard())
        try:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            before = COUNTS.copy()
            start = time.monotonic()
            receipt = run(host, rows, config["target_root"], journal,
                          256 * 1024**2, guard)
            torch.cuda.synchronize()
            result["unit"] = dict(sample_units=50, seconds=time.monotonic() - start,
                                  counts=dict(COUNTS - before),
                                  peak_gpu_bytes=torch.cuda.max_memory_allocated(),
                                  prediction_bytes=receipt["prediction_bytes"],
                                  trace_bytes=receipt["trace_bytes"],
                                  physical_bytes=journal.physical.stat().st_size,
                                  checkpoint_bytes=sum(p.stat().st_size for p in journal.root.glob("checkpoint.*")))
        finally:
            hook.remove()
        result["status"] = "MEASURED_SURROGATE_ONLY"
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = dict(type=type(exc).__name__, message=str(exc)[:2000])
        raise
    finally:
        if segmenter is not None:
            segmenter.close()
        result["elapsed_seconds"] = time.monotonic() - started
        result["physical_counts"] = dict(COUNTS)
        with Path(config["output"]).open("x") as file:
            json.dump(result, file, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
