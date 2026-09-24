"""Label-capable R8 phase, entered only after a verified online seal."""
import json
import os
from pathlib import Path

import numpy as np
import torch

from ..p1_analysis import evaluate
from ..r7_target_screen.runner import TargetReader
from .journal import _digest, _replace, verify_online_complete
from .streams import rows_sha


def score(rows, target_root, job_root, job_id, context_sha256, arm, order,
          max_asset_bytes, guard):
    if not rows or not callable(guard) or not job_id or not arm:
        raise ValueError("R8 scoring binding")
    root = Path(job_root)
    rows_sha256 = rows_sha(rows)
    online = verify_online_complete(root, job_id, context_sha256, rows_sha256, len(rows))
    scalars = root / "scalars.private.jsonl"
    if scalars.exists() or (root / "score_complete.json").exists():
        raise ValueError("R8 scoring output already exists")
    reader = TargetReader(target_root, max_asset_bytes, "mask")
    first = None
    try:
        with (root / "predictions.bits").open("rb") as predictions, scalars.open("xb") as output:
            for index, row in enumerate(rows):
                guard()
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
                output.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
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
        raise first
    receipt = dict(schema="R8_SCORE_COMPLETE_V1", job_id=job_id,
                   context_sha256=context_sha256, online_prediction_sha256=online["prediction_sha256"],
                   rows_sha256=rows_sha256,
                   visits=len(rows), scalar_bytes=scalars.stat().st_size,
                   scalar_sha256=_digest(scalars))
    _replace(root / "score_complete.json", json.dumps(receipt, sort_keys=True).encode())
    return receipt
