"""Bounded source-RGB native snapshot/continuation check; never a target trajectory."""
import io
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from dpa_ctta.b1_host import Host as GraTaHost
from dpa_ctta.host_diagnostic import close
from dpa_ctta.hosts.vptta import VPTTAHost
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_source_prep.registry import verified
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.native_host import NativeHost
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.source_pilot import SourceOnlyHost


def main():
    config = json.loads(Path(os.environ["R8_PROFILE_CONFIG"]).read_text())
    if (config.get("schema") != "R8_NATIVE_RECOVERY_PROFILE_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("maximum_seconds") != 1200):
        raise ValueError("R8 native profile config")
    started = time.monotonic()
    result = dict(schema="R8_NATIVE_RECOVERY_PROFILE_RESULT_V1", status="IN_PROGRESS",
                  code_sha=config["code_sha"], physical_gpu=config["physical_gpu"],
                  input="registered source RGB only", arms={})

    def guard():
        if time.monotonic() - started >= 1200:
            raise RuntimeError("R8 native profile wall cap")

    def build(arm, state, identity):
        random.seed(20260907)
        np.random.seed(20260907)
        torch.manual_seed(20260907)
        if arm == "VPTTA_NATIVE":
            native = VPTTAHost("fundus", source_state=state, device="cuda:0")
        elif arm == "N_SOURCE_EVAL":
            native = SourceOnlyHost("fundus", state, device="cuda:0")
        else:
            native = GraTaHost(arm[0], state=state, device="cuda:0")
        return NativeHost(native, arm, identity)

    COUNTS.clear()
    try:
        bound = bind_metadata(config["refs"])
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         0.3, config["physical_gpu"], 256 * 1024**2,
                         2 * 1024**3, guard) as (data, _, io_counts):
            images = [data.get(name, "fit").image for name in sorted(data.folds["fit"])[:18]]
            result["source_io_counts"] = dict(io_counts)
        raw = verified(config["checkpoint_path"], config["checkpoint_sha256"], 256 * 1024**2)
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        for arm in ("N_SOURCE_EVAL", "C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER", "VPTTA_NATIVE"):
            identity = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                            checkpoint_sha256=config["checkpoint_sha256"],
                            registration_sha256=config["refs"]["target"]["sha256"],
                            seed=None if arm == "N_SOURCE_EVAL" else 20260907)
            worker = build(arm, state, identity)
            hook = worker.native.model.register_forward_pre_hook(lambda *_: guard())
            count = 16 if arm == "VPTTA_NATIVE" else 2
            before = COUNTS.copy()
            start = time.monotonic()
            for image in images[:count]:
                worker.step(image)
            torch.cuda.synchronize()
            elapsed = time.monotonic() - start
            forward_cost = dict(COUNTS - before)
            snapshot = worker.snapshot()
            buffer = io.BytesIO()
            torch.save(snapshot, buffer)
            frozen = buffer.getvalue()
            expected, trace = worker.step(images[count])
            expected_state = worker.snapshot()
            worker.close()
            hook.remove()
            del worker
            torch.cuda.empty_cache()
            resumed = build(arm, state, identity)
            resumed.restore(torch.load(io.BytesIO(frozen), map_location="cpu", weights_only=True))
            actual, actual_trace = resumed.step(images[count])
            close(expected, actual, exact=True)
            close(trace, actual_trace, exact=True)
            close(expected_state, resumed.snapshot(), exact=True)
            result["arms"][arm] = dict(status="EXACT_CONTINUATION_MATCH", prefix_visits=count,
                                      prefix_seconds=elapsed, prefix_counts=forward_cost,
                                      snapshot_bytes=len(frozen), continuation_visits=1)
            resumed.close()
            del resumed
            torch.cuda.empty_cache()
        result["status"] = "SOURCE_RGB_RECOVERY_CHECK_PASSED"
    except BaseException as exc:
        result.update(status="FAILED", error=dict(type=type(exc).__name__, message=str(exc)[:3000]))
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        with Path(config["output"]).open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)


if __name__ == "__main__":
    main()
