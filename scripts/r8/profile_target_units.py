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
from dpa_ctta.r7_shared.context import tensor_digest
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r8_ba.context import SOURCE_KEYS, capture
from dpa_ctta.r8_ba.gradient import GradientHost
from dpa_ctta.r8_ba.host import OnlineHost
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.methods import CurrentMLP, build
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256


def main():
    config = json.loads(Path(os.environ["R8_PROFILE_CONFIG"]).read_text())
    if (config.get("schema") != "R8_TARGET_UNIT_PROFILE_CONFIG_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("maximum_seconds") != 1200):
        raise ValueError("R8 target profile config")
    started = time.monotonic()
    result = dict(schema="R8_TARGET_UNIT_PROFILE_V1", status="IN_PROGRESS",
                  code_sha=config["code_sha"], physical_gpu=config["physical_gpu"],
                  input="registered source RGB; no target image or label", units={})
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
            result["io_counts"] = dict(io_counts)

        raw = verified(config["checkpoint_path"], config["checkpoint_sha256"], 256 * 1024**2)
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        torch.manual_seed(20260907)
        np.random.seed(20260907)
        native = VPTTAHost("fundus", source_state=state, device="cuda:0")
        forwards = [0]
        hook = native.model.register_forward_hook(lambda *_: forwards.__setitem__(0, forwards[0] + 1))
        try:
            measure("vptta", 6, lambda: [native.step(image).shape for _ in range(6)])
            result["units"]["vptta"]["native_model_forwards"] = forwards[0]
        finally:
            hook.remove()
        del native
        torch.cuda.empty_cache()
        for arm, name in (("C", "c"), ("G", "g")):
            torch.manual_seed(20260907)
            np.random.seed(20260907)
            worker = GraTaHost(arm, state=state, device="cuda:0")
            measure(name, 1, lambda worker=worker: worker.step(image)[1])
            worker.finish(state)
            del worker
            torch.cuda.empty_cache()
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
