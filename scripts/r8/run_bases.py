"""Produce one amplitude's A/B bases and B source-fit gradient scales."""
import hashlib
import json
import os
import re
import time
from pathlib import Path

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.journal import _replace
from dpa_ctta.r8_ba.oracle_journal import load_oracles
from dpa_ctta.r8_ba.oracles import a_bases, b_bases
from dpa_ctta.r8_ba.preparation import gradient_scale, save_bases
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.resources import CAPS


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_BASES_WORK_V1" or
            config.get("physical_gpu") not in (5, 6, 7) or
            config["physical_gpu"] != int(os.environ["CUDA_VISIBLE_DEVICES"]) or
            config.get("amplitude") not in (0.1, 0.3) or
            re.fullmatch(r"[0-9a-f]{40}", config.get("code_sha", "")) is None or
            config.get("protocol_sha256") != PROTOCOL_SHA256 or
            type(config.get("maximum_seconds")) is not int or
            not 0 < config["maximum_seconds"] <= CAPS["gpu_seconds"]):
        raise ValueError("R8 bases worker config")
    started = time.monotonic()
    root = Path(config["job_root"])
    bound = bind_metadata(config["refs"])
    oracle_identity = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256,
                           refs=config["refs"],
                           checkpoint_sha256=bound["docs"]["manifest"]["checkpoint"]["sha256"],
                           amplitude=config["amplitude"])
    oracle_root = Path(config["oracle_root"])
    worker = json.loads((oracle_root / "worker_complete.json").read_text())
    receipt_hash = hashlib.sha256((oracle_root / "oracle_complete.json").read_bytes()).hexdigest()
    if (worker.get("schema") != "R8_ORACLE_WORK_COMPLETE_V1" or
            worker.get("identity") != oracle_identity or
            worker.get("oracle_receipt_sha256") != receipt_hash):
        raise ValueError("R8 completed source oracle worker required")
    oracle_binding = hashlib.sha256(json.dumps(oracle_identity, sort_keys=True).encode()).hexdigest()
    identity = dict(**oracle_identity, oracle_receipt_sha256=receipt_hash)

    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 bases worker wall cap reached")

    with open_source(bound, config["source_root"], config["checkpoint_path"],
                     config["amplitude"], config["physical_gpu"], 256 * 1024**2,
                     2 * 1024**3, guard) as (data, segmenter, io_counts):
        fit = load_oracles(oracle_root, data, oracle_binding, config["amplitude"])["fit"]
        COUNTS.clear()
        hook = segmenter.register_forward_pre_hook(lambda *_: guard())
        try:
            b = b_bases(fit, data)
            guard()
            a, audit = a_bases(segmenter, data)
            guard()
            scale = {rank: gradient_scale(fit, b[rank]) for rank in (32, 64)}
            counts = COUNTS.copy()
        finally:
            hook.remove()
    receipt = save_bases(root, identity, a, b, audit, scale, counts)
    _replace(root / "worker_complete.json", json.dumps(dict(
        schema="R8_BASES_WORK_COMPLETE_V1", identity=identity,
        bases_receipt_sha256=hashlib.sha256((root / "bases_complete.json").read_bytes()).hexdigest(),
        source_io_counts=dict(io_counts)), sort_keys=True, allow_nan=False).encode())
    return receipt


if __name__ == "__main__":
    main()
