"""Run the shared R8 source-fit scaler cache from registered source inputs."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.journal import _replace
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.resources import CAPS
from dpa_ctta.r8_ba.scaler_journal import run_scaler
from dpa_ctta.r8_ba.paths import owned_source_path


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_SCALER_WORK_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            re.fullmatch(r"[0-9a-f]{40}", config.get("code_sha", "")) is None or
            config.get("protocol_sha256") != PROTOCOL_SHA256 or
            type(config.get("maximum_seconds")) is not int or
            not 0 < config["maximum_seconds"] <= CAPS["gpu_seconds"]):
        raise ValueError("R8 scaler worker config")
    started = time.monotonic()
    root = owned_source_path(config["job_root"])
    bound = bind_metadata(config["refs"])
    identity = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                    refs=config["refs"],
                    checkpoint_sha256=bound["docs"]["manifest"]["checkpoint"]["sha256"],
                    fold="fit", zero_film=True)
    binding = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 scaler worker wall cap reached")

    first = None
    try:
        # The zero-FiLM backbone is independent of amplitude; choose the frozen 0.1 constructor.
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         0.1, config["physical_gpu"], 256 * 1024**2,
                         2 * 1024**3, guard) as (data, segmenter, io_counts):
            receipt = run_scaler(root, segmenter, data, binding, guard,
                                 config.get("resume_failure"))
        _replace(root / "worker_complete.json", json.dumps(dict(
            schema="R8_SCALER_WORK_COMPLETE_V1", identity=identity,
            scaler_receipt_sha256=hashlib.sha256(
                (root / "scaler_complete.json").read_bytes()).hexdigest(),
            source_io_counts=dict(io_counts)), sort_keys=True, allow_nan=False).encode())
        return receipt
    except BaseException as exc:
        first = exc
        raise
    finally:
        if first is not None and root.is_dir():
            status = dict(schema="R8_SCALER_WORK_FAILURE_V1", identity=identity,
                          error_type=type(first).__name__, error=str(first)[:3000])
            with (root / "worker_failures.jsonl").open("a") as stream:
                stream.write(json.dumps(status, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
