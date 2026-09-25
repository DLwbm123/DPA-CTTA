"""Run one full R8 source oracle amplitude from registered source inputs."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.journal import _replace
from dpa_ctta.r8_ba.oracle_shards import run as run_oracles
from dpa_ctta.r8_ba.paths import owned_source_path
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.resources import CAPS


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_ORACLE_SHARD_WORK_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("amplitude") not in (0.1, 0.3) or
            not isinstance(config.get("code_sha"), str) or
            re.fullmatch(r"[0-9a-f]{40}", config["code_sha"]) is None or
            config.get("protocol_sha256") != PROTOCOL_SHA256 or
            type(config.get("maximum_seconds")) is not int or
            not 0 < config["maximum_seconds"] <= CAPS["gpu_seconds"]):
        raise ValueError("R8 oracle worker config")
    started = time.monotonic()
    root = owned_source_path(config["job_root"])
    bound = bind_metadata(config["refs"])
    identity = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                    refs=config["refs"], checkpoint_sha256=bound["docs"]["manifest"]["checkpoint"]["sha256"],
                    amplitude=config["amplitude"], shard=config["shard"])
    binding = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 oracle worker wall cap reached")

    first = None
    try:
        with open_source(bound, config["source_root"], config["checkpoint_path"],
                         config["amplitude"], config["physical_gpu"],
                         256 * 1024**2, 2 * 1024**3, guard) as (data, segmenter, io_counts):
            receipt = run_oracles(root, segmenter, data, identity, guard,
                                  config.get("resume_failure"))
        _replace(root / "worker_complete.json", json.dumps(dict(
            schema="R8_ORACLE_SHARD_WORK_COMPLETE_V1", identity=identity,
            shard_receipt_sha256=hashlib.sha256(
                (root / "shard_complete.json").read_bytes()).hexdigest(),
            source_io_counts=dict(io_counts)), sort_keys=True, allow_nan=False).encode())
        return receipt
    except BaseException as exc:
        first = exc
        raise
    finally:
        if first is not None and root.is_dir():
            status = dict(schema="R8_ORACLE_WORK_FAILURE_V1", identity=identity,
                          error_type=type(first).__name__, error=str(first)[:3000])
            with (root / "worker_failures.jsonl").open("a") as stream:
                stream.write(json.dumps(status, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())


if __name__ == "__main__":
    main()
