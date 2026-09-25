"""Independent R8 calibration state and one shared source-job recovery allowance."""
import hashlib
import io
import json
import os
from pathlib import Path

import torch

from ..r7_shared.numerics import COUNTS
from .journal import SourceJournal, _digest, _replace, _sync_dir, _reject_recorded_noninfra_failure
from .trainer import SAVE_STEPS, MAX_STEPS


class CalibrationJournal:
    def __init__(self, job_root, source_step, calibrator):
        if source_step not in SAVE_STEPS:
            raise ValueError("R8 calibration source point")
        self.job_root = Path(job_root)
        self.root = self.job_root / f"calibration.{source_step}"
        self.source_step = source_step
        self.calibrator = calibrator
        self.physical = self.root / "physical.jsonl"
        self.previous_counts = COUNTS.copy()

    def _identity(self):
        fit = json.loads((self.job_root / "fit_complete.json").read_text())
        selected = self.job_root / f"selected.{self.source_step}.pt"
        if (fit.get("schema") != "R8_SOURCE_FIT_COMPLETE_V1" or
                fit.get("binding") != self.calibrator.binding or
                fit.get("steps") != MAX_STEPS or
                fit["selected_sha256"].get(str(self.source_step)) != _digest(selected)):
            raise ValueError("R8 calibration fit/source point binding")
        return dict(binding=self.calibrator.binding, source_step=self.source_step,
                    selected_sha256=fit["selected_sha256"][str(self.source_step)])

    def create(self):
        if self.calibrator.steps != 0:
            raise ValueError("R8 calibration start step")
        self._identity()
        self.root.mkdir(exist_ok=False)
        self.physical.touch(exist_ok=False)
        _sync_dir(self.root)
        self.checkpoint()

    def append(self, trace):
        if trace.get("step") != self.calibrator.steps or not 1 <= self.calibrator.steps <= 1024:
            raise ValueError("R8 calibration physical step")
        record = dict(trace, counts=dict(COUNTS - self.previous_counts))
        with self.physical.open("ab") as stream:
            stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        self.previous_counts = COUNTS.copy()
        if self.calibrator.steps % 250 == 0:
            self.checkpoint()

    def record_failed_call(self, step, counts, error):
        record = dict(status="FAILED_CALL", step=step, counts=counts,
                      error_type=type(error).__name__, error=str(error)[:3000])
        with self.physical.open("ab") as stream:
            stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def checkpoint(self):
        steps = self.calibrator.steps
        if steps % 250 or steps > 1000:
            raise ValueError("R8 calibration checkpoint interval")
        payload = dict(schema="R8_CAL_JOURNAL_V1", identity=self._identity(),
                       snapshot=self.calibrator.snapshot())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        slot = (steps // 250) % 2
        _replace(self.root / f"checkpoint.{slot}.pt", data)
        _replace(self.root / f"checkpoint.{slot}.json", json.dumps(
            dict(sha256=hashlib.sha256(data).hexdigest(), steps=steps), sort_keys=True).encode())

    def recover_once(self, failure):
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not isinstance(failure.get("evidence"), dict) or
                not failure["evidence"]):
            raise ValueError("R8 only evidenced infrastructure failure may recover")
        _reject_recorded_noninfra_failure(self.job_root)
        marker = self.job_root / "recovery.json"
        if (marker.exists() or not self.physical.is_file() or
                (self.root / "cal_complete.json").exists()):
            raise ValueError("R8 source job recovery unavailable or already used")
        identity = self._identity()
        valid = []
        for slot in (0, 1):
            try:
                data = (self.root / f"checkpoint.{slot}.pt").read_bytes()
                meta = json.loads((self.root / f"checkpoint.{slot}.json").read_text())
                if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                    continue
                payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                snapshot = payload["snapshot"]
                if (payload["schema"] == "R8_CAL_JOURNAL_V1" and payload["identity"] == identity and
                        type(snapshot.get("steps")) is int and snapshot["steps"] % 250 == 0 and
                        snapshot["steps"] == meta["steps"] and 0 <= snapshot["steps"] <= 1000):
                    valid.append(snapshot)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pass
        if not valid:
            raise ValueError("R8 no verified equivalent calibration snapshot")
        chosen = max(valid, key=lambda snapshot: snapshot["steps"])
        raw = self.physical.read_bytes()
        if SourceJournal._check_steps(raw, 1, True) < chosen["steps"] + 1:
            raise ValueError("R8 calibration snapshot physical prefix missing")
        if raw:
            last = json.loads(raw.splitlines()[-1])
            if (last.get("status") == "FAILED_CALL" and last.get("error_type") not in
                    ("OSError", "ConnectionError", "InterruptedError", "TimeoutError")):
                raise ValueError("R8 numerical/resource failure cannot recover")
        self.calibrator.restore(chosen)
        self.previous_counts = COUNTS.copy()
        _replace(marker, json.dumps(dict(schema="R8_SOURCE_RECOVERY_V1", stage="calibration",
                                        **identity, steps=self.calibrator.steps, failure=failure,
                                        physical_log_bytes=self.physical.stat().st_size),
                                    sort_keys=True).encode())
        return self.calibrator.steps

    def complete(self):
        if self.calibrator.steps != 1024 or (self.root / "cal_complete.json").exists():
            raise ValueError("R8 calibration not complete")
        identity = self._identity()
        raw = self.physical.read_bytes()
        marker = self.job_root / "recovery.json"
        if marker.exists():
            recovery = json.loads(marker.read_text())
            if recovery.get("stage") == "calibration" and recovery.get("source_step") == self.source_step:
                cut = recovery["physical_log_bytes"]
                if (recovery.get("binding") != identity["binding"] or
                        recovery.get("selected_sha256") != identity["selected_sha256"] or
                        recovery.get("failure", {}).get("class") != "INFRASTRUCTURE" or
                        type(cut) is not int or not 0 <= cut <= len(raw) or
                        SourceJournal._check_steps(raw[:cut], 1, True) < recovery["steps"] + 1):
                    raise ValueError("R8 calibration recovery/log mismatch")
                end = SourceJournal._check_steps(raw[cut:], recovery["steps"] + 1)
            else:
                end = SourceJournal._check_steps(raw, 1)
        else:
            end = SourceJournal._check_steps(raw, 1)
        if end != 1025:
            raise ValueError("R8 calibration physical completion coverage")
        self.calibrator.method.freeze()
        state = dict(schema="R8_CALIBRATED_METHOD_V1", identity=identity,
                     method_digest=self.calibrator.method.digest(),
                     method=self.calibrator.method.state_dict())
        buffer = io.BytesIO()
        torch.save(state, buffer)
        path = self.root / "calibrated.pt"
        if path.exists():
            old = torch.load(path, map_location="cpu", weights_only=True)
            if (old.get("schema") != state["schema"] or old.get("identity") != identity or
                    old.get("method_digest") != state["method_digest"] or
                    set(old["method"]) != set(state["method"]) or
                    any(not torch.equal(old["method"][key], value)
                        for key, value in state["method"].items())):
                raise ValueError("R8 existing calibrated artifact mismatch")
        else:
            _replace(path, buffer.getvalue())
        receipt = dict(schema="R8_CAL_COMPLETE_V1", **identity, steps=1024,
                       method_digest=state["method_digest"], artifact_sha256=_digest(path),
                       physical_bytes=len(raw), physical_sha256=hashlib.sha256(raw).hexdigest())
        _replace(self.root / "cal_complete.json", json.dumps(receipt, sort_keys=True).encode())
        return receipt
