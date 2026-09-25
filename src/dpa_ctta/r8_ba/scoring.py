"""Label-capable R8 phase, entered only after a verified online seal."""
import json
import hashlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from ..p1_analysis import evaluate
from ..r7_target_screen.runner import TargetReader
from .journal import _digest, _replace, _reject_recorded_noninfra_failure, verify_online_complete
from .streams import rows_sha


def score(rows, target_root, job_root, job_id, context_sha256, arm, order,
          max_asset_bytes, guard, resume_failure=None):
    if not rows or not callable(guard) or not job_id or not arm:
        raise ValueError("R8 scoring binding")
    root = Path(job_root)
    rows_sha256 = rows_sha(rows)
    online = verify_online_complete(root, job_id, context_sha256, rows_sha256, len(rows))
    scalars = root / "scalars.private.jsonl"
    if (root / "score_complete.json").exists():
        raise ValueError("R8 scoring output already exists")
    identity = dict(job_id=job_id, context_sha256=context_sha256, rows_sha256=rows_sha256,
                    prediction_sha256=online["prediction_sha256"], arm=arm, order=order)
    checkpoint = root / "score_checkpoint.json"
    physical = root / "score_physical.jsonl"
    hasher, first_index = hashlib.sha256(), 0
    if scalars.exists():
        _reject_recorded_noninfra_failure(root)
        if (not isinstance(resume_failure, dict) or resume_failure.get("class") != "INFRASTRUCTURE" or
                not resume_failure.get("reason") or not resume_failure.get("evidence") or
                (root / "recovery.json").exists() or not checkpoint.is_file()):
            raise ValueError("R8 scoring requires one evidenced infrastructure recovery")
        saved = json.loads(checkpoint.read_text())
        size, first_index = saved["bytes"], saved["visits"]
        if (saved.get("identity") != identity or type(first_index) is not int or
                not 0 <= first_index <= len(rows) or type(size) is not int or size < 0 or
                _digest(scalars, size) != saved["sha256"]):
            raise ValueError("R8 scoring checkpoint identity/prefix")
        with scalars.open("rb") as stream:
            prefix = stream.read(size)
        records = prefix.splitlines()
        if (len(records) != first_index or (prefix and not prefix.endswith(b"\n")) or
                any(json.loads(line).get("visit") != index + 1 or
                    json.loads(line).get("content") != rows[index]["group_id"]
                    for index, line in enumerate(records))):
            raise ValueError("R8 scoring committed row coverage")
        hasher.update(prefix)
        _replace(root / "recovery.json", json.dumps(dict(schema="R8_SCORE_RECOVERY_V1",
            identity=identity, visits=first_index, failure=resume_failure,
            discarded_scalar_bytes=scalars.stat().st_size - size), sort_keys=True).encode())
        with scalars.open("r+b") as stream:
            stream.truncate(size)
            stream.flush()
            os.fsync(stream.fileno())
    else:
        if resume_failure is not None or checkpoint.exists() or physical.exists():
            raise ValueError("R8 scoring start has inconsistent prior state")
        scalars.touch(exist_ok=False)
        physical.touch(exist_ok=False)
        _replace(checkpoint, json.dumps(dict(identity=identity, visits=0, bytes=0,
                                             sha256=hasher.hexdigest()), sort_keys=True).encode())
    reader = TargetReader(target_root, max_asset_bytes, "mask")
    first = None
    try:
        with (root / "predictions.bits").open("rb") as predictions, scalars.open("ab") as output:
            predictions.seek(first_index * 65536)
            for index in range(first_index, len(rows)):
                row = rows[index]
                guard()
                started = time.monotonic()
                raw = predictions.read(65536)
                if len(raw) != 65536:
                    raise ValueError("R8 prediction dimensions")
                bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).copy()
                probability = torch.from_numpy(bits).float().reshape(1, 2, 512, 512)
                metrics = evaluate(probability, reader.read(row), "fundus")
                record = dict(visit=index + 1, cycle=index // 1951 + 1,
                              cycle_visit=index % 1951 + 1, arm=arm, order=order,
                              content=row["group_id"], domain=row["domain"],
                              subset=row["subset"], metrics=metrics)
                encoded = (json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode()
                with physical.open("a") as work:
                    work.write(json.dumps(dict(visit=index + 1, seconds=time.monotonic() - started,
                                               recovered=resume_failure is not None), sort_keys=True) + "\n")
                    work.flush()
                    os.fsync(work.fileno())
                output.write(encoded)
                hasher.update(encoded)
                if (index + 1) % 50 == 0 or index + 1 == len(rows):
                    output.flush()
                    os.fsync(output.fileno())
                    _replace(checkpoint, json.dumps(dict(identity=identity, visits=index + 1,
                        bytes=output.tell(), sha256=hasher.hexdigest()), sort_keys=True).encode())
                guard()
            output.flush()
            os.fsync(output.fileno())
    except BaseException as exc:
        first = exc
    finally:
        try:
            reader.after_check()
        except BaseException as exc:
            if first is None:
                first = exc
    if first is not None:
        with (root / "worker_failures.jsonl").open("a") as work:
            work.write(json.dumps(dict(stage="scoring", error_type=type(first).__name__,
                                       error=str(first)[:3000]), sort_keys=True) + "\n")
            work.flush()
            os.fsync(work.fileno())
        raise first
    receipt = dict(schema="R8_SCORE_COMPLETE_V1", job_id=job_id,
                   context_sha256=context_sha256, online_prediction_sha256=online["prediction_sha256"],
                   rows_sha256=rows_sha256,
                   visits=len(rows), scalar_bytes=scalars.stat().st_size,
                   scalar_sha256=_digest(scalars), physical_sha256=_digest(physical))
    if receipt["scalar_sha256"] != hasher.hexdigest():
        raise ValueError("R8 scoring output changed before seal")
    _replace(root / "score_complete.json", json.dumps(receipt, sort_keys=True).encode())
    return receipt
