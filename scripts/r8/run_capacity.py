"""All four latent spaces and 64 fixed source-val probes for one amplitude."""
import hashlib
import json
import os
import time
from pathlib import Path

from dpa_ctta.r8_ba.capacity import ANCHORS, SPACES, capacity_one
from dpa_ctta.r8_ba.independent_journal import IndependentJournal
from dpa_ctta.r8_ba.inputs import bind_metadata, open_source
from dpa_ctta.r8_ba.journal import _replace, _reject_recorded_noninfra_failure
from dpa_ctta.r8_ba.oracle_journal import load_oracles
from dpa_ctta.r8_ba.paths import owned_source_path
from dpa_ctta.r8_ba.preparation import load_bases
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256


def main():
    config = json.loads(Path(os.environ["R8_WORK_CONFIG"]).read_text())
    if (config.get("schema") != "R8_CAPACITY_WORK_V1" or config.get("amplitude") not in (0.1, 0.3) or
            config.get("protocol_sha256") != PROTOCOL_SHA256):
        raise ValueError("R8 capacity worker config")
    root = owned_source_path(config["job_root"])
    _reject_recorded_noninfra_failure(root)
    bound = bind_metadata(config["refs"])
    common = dict(code_sha=config["code_sha"], protocol_sha256=PROTOCOL_SHA256, refs=config["refs"],
                  checkpoint_sha256=bound["docs"]["manifest"]["checkpoint"]["sha256"],
                  amplitude=config["amplitude"])
    oracle_root, bases_root = owned_source_path(config["oracle_root"]), owned_source_path(config["bases_root"])
    oracle_sha = hashlib.sha256((oracle_root / "oracle_complete.json").read_bytes()).hexdigest()
    oracle_marker = json.loads((oracle_root / "worker_complete.json").read_text())
    bases_identity = dict(**common, oracle_receipt_sha256=oracle_sha)
    bases_sha = hashlib.sha256((bases_root / "bases_complete.json").read_bytes()).hexdigest()
    bases_marker = json.loads((bases_root / "worker_complete.json").read_text())
    if (oracle_marker.get("schema") != "R8_ORACLE_WORK_COMPLETE_V1" or
            oracle_marker.get("identity") != common or oracle_marker.get("oracle_receipt_sha256") != oracle_sha or
            bases_marker.get("schema") != "R8_BASES_WORK_COMPLETE_V1" or
            bases_marker.get("identity") != bases_identity or bases_marker.get("bases_receipt_sha256") != bases_sha):
        raise ValueError("R8 capacity source producer identity")
    identity = dict(**bases_identity, bases_receipt_sha256=bases_sha)
    items = [dict(space=space, anchor=anchor) for space in SPACES for anchor in ANCHORS]
    started = time.monotonic()
    def guard():
        if time.monotonic() - started >= config["maximum_seconds"]:
            raise RuntimeError("R8 capacity worker wall cap reached")
    journal = IndependentJournal(root, identity, items)
    try:
        with open_source(bound, config["source_root"], config["checkpoint_path"], config["amplitude"],
                         config["physical_gpu"], 256 * 1024**2, 2 * 1024**3, guard) as (data, segmenter, io_counts):
            binding = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
            oracles = load_oracles(oracle_root, data, binding, config["amplitude"])
            bases = load_bases(bases_root, bases_identity)
            if root.exists():
                journal.recover_once(config.get("resume_failure"))
            else:
                journal.create()
            hook = segmenter.register_forward_pre_hook(lambda *_: guard())
            try:
                receipt = journal.run(lambda item: capacity_one(segmenter, data, oracles["val"],
                    bases[item["space"][0] + "_basis"][SPACES[item["space"]]],
                    item["space"], item["anchor"]), guard)
            finally:
                hook.remove()
        _replace(root / "worker_complete.json", json.dumps(dict(schema="R8_CAPACITY_WORK_COMPLETE_V1",
            identity=identity, receipt_sha256=hashlib.sha256((root / "complete.json").read_bytes()).hexdigest(),
            source_io_counts=dict(io_counts)), sort_keys=True).encode())
        return receipt
    except BaseException as exc:
        if root.exists():
            with (root / "worker_failures.jsonl").open("a") as output:
                output.write(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)[:3000])) + "\n")
                output.flush()
                os.fsync(output.fileno())
        raise


if __name__ == "__main__":
    main()
