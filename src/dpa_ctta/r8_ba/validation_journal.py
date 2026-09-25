"""Source-val episode boundaries, append-only physical visits, one shared recovery."""
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path

from ..r7_shared.numerics import COUNTS
from .calibration import validate_episode
from .trainer import SAVE_STEPS
from .journal import _replace, _sync_dir, _reject_recorded_noninfra_failure
from .schedule import CURRICULA

EPISODES = 64
VISITS = 32


def _json(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


def _physical(raw, start, completed, tail=False):
    if raw and not raw.endswith(b"\n"):
        raise ValueError("R8 validation physical log truncated")
    rows = [json.loads(line) for line in raw.splitlines()]
    episode, visit = start, 0
    total = Counter()
    for position, row in enumerate(rows):
        if episode >= start + completed + (1 if tail else 0):
            raise ValueError("R8 validation extra physical work")
        counts = row.get("counts")
        if (row.get("episode") != episode or row.get("visit") != visit + 1 or
                not isinstance(counts, dict) or
                any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("R8 validation physical identity/cost")
        total.update(counts)
        if row.get("status") == "FAILED_CALL":
            if not tail or position != len(rows) - 1 or episode != start + completed:
                raise ValueError("R8 validation failed call position")
            continue
        visit += 1
        if visit == VISITS:
            episode, visit = episode + 1, 0
    if ((episode != start + completed and
         not (tail and episode == start + completed + 1 and visit == 0)) or
            (not tail and visit)):
        raise ValueError("R8 validation committed visit coverage")
    return total


class ValidationJournal:
    def __init__(self, job_root, source_step, method_sha256, binding, calibrated=True):
        if (source_step not in SAVE_STEPS or
                not isinstance(method_sha256, str) or len(method_sha256) != 64 or not binding):
            raise ValueError("R8 validation source artifact identity")
        self.job_root = Path(job_root)
        if type(calibrated) is not bool:
            raise ValueError("R8 validation calibration phase")
        self.calibrated = calibrated
        prefix = "validation" if calibrated else "validation_uncalibrated"
        self.root = self.job_root / f"{prefix}.{source_step}"
        self.identity = dict(source_step=source_step, method_sha256=method_sha256,
                             binding=binding, calibrated=calibrated)
        self.rows = self.root / "rows.jsonl"
        self.physical = self.root / "physical.jsonl"
        self.completed = 0
        self.active = False
        self.failed = False

    def create(self):
        self.root.mkdir(parents=False, exist_ok=False)
        self.rows.touch(exist_ok=False)
        self.physical.touch(exist_ok=False)
        _replace(self.root / "identity.json", _json(self.identity))
        _sync_dir(self.root)
        self.active = True

    def _rows(self):
        raw = self.rows.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("R8 validation rows truncated")
        rows = [json.loads(line) for line in raw.splitlines()]
        if len(rows) > EPISODES:
            raise ValueError("R8 validation too many episodes")
        for episode, row in enumerate(rows):
            if (row.get("episode") != episode or
                    row.get("curriculum") != CURRICULA[episode // 16] or
                    type(row.get("soft_Dice")) not in (int, float) or
                    not math.isfinite(row["soft_Dice"]) or
                    not 0 <= row["soft_Dice"] <= 1 or
                    type(row.get("proxy_MSE")) not in (int, float) or
                    not math.isfinite(row["proxy_MSE"]) or row["proxy_MSE"] < 0):
                raise ValueError("R8 validation row coverage/value")
        return rows, raw

    def _totals(self, raw, count):
        marker = self.job_root / "recovery.json"
        if not marker.exists():
            return _physical(raw, 0, count)
        recovery = json.loads(marker.read_text())
        if (recovery.get("stage") != "validation" or recovery.get("source_step") != self.identity["source_step"] or
                recovery.get("calibrated", True) != self.calibrated):
            return _physical(raw, 0, count)
        before, cut, boundary = (recovery[k] for k in
                                 ("snapshot_physical_bytes", "physical_log_bytes", "completed_episodes"))
        if (recovery.get("identity") != self.identity or
                not 0 <= before <= cut <= len(raw) or count < boundary):
            raise ValueError("R8 validation recovery identity/offset")
        total = _physical(raw[:before], 0, boundary)
        total.update(_physical(raw[before:cut], boundary, 0, tail=True))
        total.update(_physical(raw[cut:], boundary, count - boundary))
        return total

    def recover_once(self, failure):
        _reject_recorded_noninfra_failure(self.job_root)
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not isinstance(failure.get("evidence"), dict) or
                not failure["evidence"] or not self.root.is_dir() or
                (self.job_root / "recovery.json").exists() or
                (self.root / "val_complete.json").exists() or
                json.loads((self.root / "identity.json").read_text()) != self.identity):
            raise ValueError("R8 validation recovery unavailable")
        rows, _ = self._rows()
        raw = self.physical.read_bytes()
        boundary = len(rows) * VISITS
        physical_rows = raw.splitlines()
        if len(physical_rows) < boundary:
            raise ValueError("R8 validation physical prefix missing")
        before = sum(len(line) + 1 for line in physical_rows[:boundary])
        _physical(raw[:before], 0, len(rows))
        _physical(raw[before:], len(rows), 0, tail=True)
        if physical_rows:
            last = json.loads(physical_rows[-1])
            if (last.get("status") == "FAILED_CALL" and last.get("error_type") not in
                    ("OSError", "ConnectionError", "InterruptedError", "TimeoutError")):
                raise ValueError("R8 numerical/resource failure cannot recover")
        self.completed = len(rows)
        _replace(self.job_root / "recovery.json", _json(dict(
            schema="R8_SOURCE_RECOVERY_V1", stage="validation", identity=self.identity, calibrated=self.calibrated,
            source_step=self.identity["source_step"], completed_episodes=self.completed,
            snapshot_physical_bytes=before, physical_log_bytes=len(raw), failure=failure)))
        self.active = True
        return self.completed

    def run_next(self, segmenter, method, data, val_oracles, guard):
        if not self.active or self.failed or self.completed >= EPISODES or not callable(guard):
            raise ValueError("R8 validation next episode")
        episode = self.completed
        logged = 0
        previous = COUNTS.copy()

        def visit_done(visit):
            nonlocal logged, previous
            guard()
            if visit != logged + 1:
                raise ValueError("R8 validation visit order")
            with self.physical.open("ab") as stream:
                stream.write(_json(dict(episode=episode, visit=visit,
                                        counts=dict(COUNTS - previous))))
                stream.flush()
                os.fsync(stream.fileno())
            previous, logged = COUNTS.copy(), visit

        hook = segmenter.register_forward_pre_hook(lambda *_: guard())
        try:
            guard()
            result = validate_episode(segmenter, method, data, val_oracles,
                                      episode, on_visit=visit_done)
            if logged != VISITS:
                raise ValueError("R8 validation incomplete episode")
            with self.rows.open("ab") as stream:
                stream.write(_json(result))
                stream.flush()
                os.fsync(stream.fileno())
            self.completed += 1
            return self.completed
        except BaseException as exc:
            self.failed = True
            if logged < VISITS:
                with self.physical.open("ab") as stream:
                    stream.write(_json(dict(episode=episode, visit=logged + 1,
                                            status="FAILED_CALL", counts=dict(COUNTS - previous),
                                            error_type=type(exc).__name__, error=str(exc)[:3000])))
                    stream.flush()
                    os.fsync(stream.fileno())
            raise
        finally:
            hook.remove()

    def complete(self):
        rows, row_bytes = self._rows()
        if (len(rows) != EPISODES or self.completed != EPISODES or
                (self.root / "val_complete.json").exists() or
                json.loads((self.root / "identity.json").read_text()) != self.identity):
            raise ValueError("R8 validation incomplete")
        physical = self.physical.read_bytes()
        totals = self._totals(physical, EPISODES)
        receipt = dict(schema="R8_SOURCE_VAL_COMPLETE_V1", identity=self.identity,
                       episodes=EPISODES, rows_sha256=hashlib.sha256(row_bytes).hexdigest(),
                       physical_sha256=hashlib.sha256(physical).hexdigest(),
                       physical_counts=dict(totals),
                       recovered=(self.job_root / "recovery.json").exists())
        _replace(self.root / "val_complete.json", _json(receipt))
        return receipt


def run_validation(root, step, method_sha256, binding, segmenter, method,
                   data, val_oracles, guard, resume_failure=None, calibrated=True):
    job = ValidationJournal(root, step, method_sha256, binding, calibrated)
    if resume_failure is None:
        job.create()
    else:
        job.recover_once(resume_failure)
    while job.completed < EPISODES:
        job.run_next(segmenter, method, data, val_oracles, guard)
    return job.complete()


def load_validation(root, step, method_sha256, binding, calibrated=True):
    job = ValidationJournal(root, step, method_sha256, binding, calibrated)
    receipt = json.loads((job.root / "val_complete.json").read_text())
    rows, raw = job._rows()
    physical = job.physical.read_bytes()
    if (receipt.get("schema") != "R8_SOURCE_VAL_COMPLETE_V1" or
            receipt.get("identity") != job.identity or
            receipt.get("episodes") != EPISODES or len(rows) != EPISODES or
            receipt.get("rows_sha256") != hashlib.sha256(raw).hexdigest() or
            receipt.get("physical_sha256") != hashlib.sha256(physical).hexdigest() or
            receipt.get("physical_counts") != dict(job._totals(physical, EPISODES))):
        raise ValueError("R8 validation completion identity/digest")
    return rows
