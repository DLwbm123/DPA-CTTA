"""R8 target online phase: current images in, committed packed predictions out."""
import numpy as np
import torch

from ..r7_target_screen.runner import TargetReader, image_records
from ..r7_shared.numerics import COUNTS


def run(host, rows, target_root, journal, max_asset_bytes, guard):
    if (not rows or journal.host is not host or journal.prediction_bytes != 65536 or
            host.visits < 0 or host.visits > len(rows) or not callable(guard) or
            journal.predictions.stat().st_size != host.visits * 65536):
        raise ValueError("R8 target online job binding")
    reader = TargetReader(target_root, max_asset_bytes, "image")
    first = None
    accounted = COUNTS.copy()
    index = host.visits
    try:
        for index in range(host.visits, len(rows)):
            guard()
            pixels = reader.read(image_records((rows[index],))[0])
            prediction, trace = host.step(pixels)
            if (prediction.requires_grad or prediction.shape != (1, 2, 512, 512) or
                    not torch.isfinite(prediction).all() or trace.get("visit") != index + 1):
                raise ValueError("R8 uncommitted online prediction")
            bits = np.packbits((prediction.sigmoid() >= 0.5).cpu().numpy().reshape(-1)).tobytes()
            journal.append(bits, trace)
            accounted = COUNTS.copy()
            guard()
        host.check_frozen(boundary=True)
    except BaseException as exc:
        first = exc
        delta = dict(COUNTS - accounted)
        if delta:
            try:
                journal.record_failed_call(index + 1, delta, exc)
            except BaseException:
                pass  # preserve the first failure; scheduler retains worker-level cost evidence
    finally:
        try:
            reader.after_check()
        except BaseException as exc:
            if first is None:
                first = exc
    if first is not None:
        raise first
    return journal.complete(len(rows))
