"""Durable target output prefix and one bounded, verified recovery."""
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

import torch

from ..r7_shared.numerics import COUNTS
from .trainer import SAVE_STEPS


def _digest(path, size=None):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = size
        while remaining is None or remaining:
            block = stream.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not block:
                break
            h.update(block)
            if remaining is not None:
                remaining -= len(block)
    if remaining not in (None, 0):
        raise ValueError("R8 journal prefix missing")
    return h.hexdigest()


def _sync_dir(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
            _sync_dir(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def verify_online_complete(root, job_id, context_sha256, rows_sha256, expected_visits,
                           prediction_bytes=65536):
    """Verify the sealed online output without constructing a model or mask reader."""
    root = Path(root)
    receipt = json.loads((root / "online_complete.json").read_text())
    identity = dict(job_id=job_id, context_sha256=context_sha256, rows_sha256=rows_sha256,
                    prediction_bytes=prediction_bytes)
    predictions, visits = root / "predictions.bits", root / "visits.jsonl"
    if (receipt.get("schema") != "R8_ONLINE_COMPLETE_V1" or
            receipt.get("identity") != identity or
            receipt.get("visits") != expected_visits or
            receipt.get("prediction_bytes") != expected_visits * prediction_bytes or
            predictions.stat().st_size != receipt["prediction_bytes"] or
            visits.stat().st_size != receipt["trace_bytes"] or
            _digest(predictions) != receipt["prediction_sha256"] or
            _digest(visits) != receipt["trace_sha256"]):
        raise ValueError("R8 online completion receipt mismatch")
    return receipt


class TargetJournal:
    def __init__(self, root, host, job_id, rows_sha256, prediction_bytes=65536):
        self.root = Path(root)
        self.host = host
        self.job_id = job_id
        self.rows_sha256 = rows_sha256
        self.prediction_bytes = prediction_bytes
        self.predictions = self.root / "predictions.bits"
        self.visits = self.root / "visits.jsonl"
        self.physical = self.root / "physical.jsonl"

    def _identity(self):
        return dict(job_id=self.job_id, context_sha256=self.host.context["sha256"],
                    rows_sha256=self.rows_sha256,
                    prediction_bytes=self.prediction_bytes)

    def create(self):
        if (self.host.visits != 0 or not self.job_id or self.prediction_bytes <= 0 or
                not isinstance(self.rows_sha256, str) or len(self.rows_sha256) != 64):
            raise ValueError("R8 journal start identity")
        self.root.mkdir(parents=True, exist_ok=False)
        for path in (self.predictions, self.visits, self.physical):
            path.touch(exist_ok=False)
        _sync_dir(self.root)
        self.checkpoint()

    def append(self, packed_prediction, trace):
        if (not isinstance(packed_prediction, bytes) or len(packed_prediction) != self.prediction_bytes or
                trace.get("visit") != self.host.visits or
                self.predictions.stat().st_size != (self.host.visits - 1) * self.prediction_bytes):
            raise ValueError("R8 journal visit or prediction offset")
        line = (json.dumps(trace, sort_keys=True, allow_nan=False) + "\n").encode()
        # Physical work remains append-only even when an uncheckpointed output tail is replayed.
        with self.physical.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        with self.predictions.open("ab") as stream:
            stream.write(packed_prediction)
            stream.flush()
            os.fsync(stream.fileno())
        with self.visits.open("ab") as stream:
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        if self.host.visits % 50 == 0:
            self.checkpoint()

    def record_failed_call(self, visit, counts, error):
        record = dict(status="FAILED_CALL", visit=visit, counts=counts,
                      error_type=type(error).__name__, error=str(error)[:3000])
        with self.physical.open("ab") as stream:
            stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    def complete(self, expected_visits):
        if (self.host.visits != expected_visits or
                self.predictions.stat().st_size != expected_visits * self.prediction_bytes or
                (self.root / "online_complete.json").exists()):
            raise ValueError("R8 target completion offset")
        self.host.check_frozen(boundary=True)
        receipt = dict(schema="R8_ONLINE_COMPLETE_V1", identity=self._identity(),
                       visits=expected_visits, prediction_bytes=self.predictions.stat().st_size,
                       prediction_sha256=_digest(self.predictions),
                       trace_bytes=self.visits.stat().st_size,
                       trace_sha256=_digest(self.visits))
        _replace(self.root / "online_complete.json", json.dumps(receipt, sort_keys=True).encode())
        return receipt

    def verified_complete(self, expected_visits):
        return verify_online_complete(self.root, self.job_id, self.host.context["sha256"],
                                      self.rows_sha256, expected_visits, self.prediction_bytes)

    def checkpoint(self):
        self.host.check_frozen(boundary=True)
        visits = self.host.visits
        output_size = self.predictions.stat().st_size
        if output_size != visits * self.prediction_bytes:
            raise ValueError("R8 journal checkpoint output offset")
        log_size = self.visits.stat().st_size
        payload = dict(schema="R8_TARGET_JOURNAL_V1", identity=self._identity(),
                       host=self.host.snapshot(), output_size=output_size,
                       output_sha256=_digest(self.predictions), log_size=log_size,
                       log_sha256=_digest(self.visits))
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        slot = (visits // 50) % 2
        snapshot = self.root / f"checkpoint.{slot}.pt"
        metadata = self.root / f"checkpoint.{slot}.json"
        _replace(snapshot, data)
        _replace(metadata, json.dumps(dict(sha256=hashlib.sha256(data).hexdigest(),
                                           visits=visits), sort_keys=True).encode())

    def _validated(self, slot):
        snapshot = self.root / f"checkpoint.{slot}.pt"
        metadata = self.root / f"checkpoint.{slot}.json"
        if not snapshot.is_file() or not metadata.is_file():
            return None
        meta = json.loads(metadata.read_text())
        data = snapshot.read_bytes()
        if hashlib.sha256(data).hexdigest() != meta["sha256"]:
            return None
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        host = payload["host"]
        visits = host["visits"]
        if (payload.get("schema") != "R8_TARGET_JOURNAL_V1" or
                payload.get("identity") != self._identity() or
                meta["visits"] != visits or visits < 0 or visits % 50 != 0 or
                payload["output_size"] != visits * self.prediction_bytes or
                self.predictions.stat().st_size < payload["output_size"] or
                self.visits.stat().st_size < payload["log_size"] or
                _digest(self.predictions, payload["output_size"]) != payload["output_sha256"] or
                _digest(self.visits, payload["log_size"]) != payload["log_sha256"]):
            return None
        return payload

    def recover_once(self, failure):
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not isinstance(failure.get("evidence"), dict) or
                not failure["evidence"]):
            raise ValueError("R8 only evidenced infrastructure failure may recover")
        if (not self.root.is_dir() or not self.predictions.is_file() or
                not self.visits.is_file() or not self.physical.is_file() or
                (self.root / "recovery.json").exists()):
            raise ValueError("R8 recovery unavailable or already used")
        valid = []
        for slot in (0, 1):
            try:
                payload = self._validated(slot)
                if payload is not None:
                    valid.append(payload)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pass
        if not valid:
            raise ValueError("R8 no verified equivalent target snapshot")
        chosen = max(valid, key=lambda item: item["host"]["visits"])
        self.host.restore(chosen["host"])
        _replace(self.root / "recovery.json", json.dumps(
            dict(schema="R8_RECOVERY_V1", **self._identity(), visits=self.host.visits,
                 failure=failure,
                 discarded_output_bytes=self.predictions.stat().st_size - chosen["output_size"],
                 physical_log_bytes=self.physical.stat().st_size), sort_keys=True).encode())
        for path, size in ((self.predictions, chosen["output_size"]),
                           (self.visits, chosen["log_size"])):
            with path.open("r+b") as stream:
                stream.truncate(size)
                stream.flush()
                os.fsync(stream.fileno())
        return self.host.visits


class SourceJournal:
    """250-step source state with an append-only record of replayed work."""

    def __init__(self, root, trainer):
        self.root = Path(root)
        self.trainer = trainer
        self.physical = self.root / "physical.jsonl"
        self.previous_counts = COUNTS.copy()

    def create(self):
        if self.trainer.steps != 0:
            raise ValueError("R8 source journal start step")
        self.root.mkdir(parents=True, exist_ok=False)
        self.physical.touch(exist_ok=False)
        _sync_dir(self.root)
        self.checkpoint()

    def append(self, trace):
        if trace.get("step") != self.trainer.steps or self.trainer.steps < 1:
            raise ValueError("R8 source journal step")
        record = dict(trace, counts=dict(COUNTS - self.previous_counts))
        with self.physical.open("ab") as stream:
            stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        self.previous_counts = COUNTS.copy()
        if self.trainer.steps % 250 == 0:
            self.checkpoint()

    def record_failed_call(self, step, counts, error):
        record = dict(status="FAILED_CALL", step=step, counts=counts,
                      error_type=type(error).__name__, error=str(error)[:3000])
        with self.physical.open("ab") as stream:
            stream.write((json.dumps(record, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())

    @staticmethod
    def _check_steps(raw, start, allow_terminal_failure=False):
        if raw and not raw.endswith(b"\n"):
            raise ValueError("R8 source physical log truncated")
        expected = start
        lines = raw.splitlines()
        for index, line in enumerate(lines):
            row = json.loads(line)
            counts = row.get("counts")
            if (not isinstance(counts, dict) or
                    any(type(value) is not int or value < 0 for value in counts.values())):
                raise ValueError("R8 source physical cost row")
            if row.get("status") == "FAILED_CALL":
                if not allow_terminal_failure or index != len(lines) - 1:
                    raise ValueError("R8 source failed call not terminal")
                continue
            if row.get("step") != expected:
                raise ValueError("R8 source physical step coverage")
            expected += 1
        return expected

    def complete(self):
        if self.trainer.steps != 16000 or (self.root / "fit_complete.json").exists():
            raise ValueError("R8 source fit not complete")
        raw = self.physical.read_bytes()
        recovery_path = self.root / "recovery.json"
        if recovery_path.exists():
            recovery = json.loads(recovery_path.read_text())
            cut = recovery["physical_log_bytes"]
            if (recovery.get("schema") != "R8_SOURCE_RECOVERY_V1" or
                    recovery.get("binding") != self.trainer.binding or
                    recovery.get("source_seed") != self.trainer.source_seed or
                    recovery.get("failure", {}).get("class") != "INFRASTRUCTURE" or
                    type(cut) is not int or not 0 <= cut <= len(raw) or
                    type(recovery.get("steps")) is not int or recovery["steps"] % 250 or
                    self._check_steps(raw[:cut], 1, allow_terminal_failure=True) < recovery["steps"] + 1):
                raise ValueError("R8 source recovery/physical log mismatch")
            final = self._check_steps(raw[cut:], recovery["steps"] + 1)
        else:
            final = self._check_steps(raw, 1)
        if final != 16001:
            raise ValueError("R8 source physical completion coverage")
        selected = {}
        final_config = self.trainer.snapshot()["method_config"]
        for step in SAVE_STEPS:
            snapshot = self.selected(step)
            if snapshot["method_config"] != final_config:
                raise ValueError("R8 selected source configuration mismatch")
            if step == 16000 and snapshot["method_digest"] != self.trainer.method.digest():
                raise ValueError("R8 final source method digest mismatch")
            selected[str(step)] = _digest(self.root / f"selected.{step}.pt")
        receipt = dict(schema="R8_SOURCE_FIT_COMPLETE_V1", binding=self.trainer.binding,
                       source_seed=self.trainer.source_seed, steps=self.trainer.steps,
                       method_digest=self.trainer.method.digest(), selected_sha256=selected,
                       physical_bytes=len(raw), physical_sha256=hashlib.sha256(raw).hexdigest(),
                       recovered=recovery_path.exists())
        _replace(self.root / "fit_complete.json", json.dumps(receipt, sort_keys=True).encode())
        return receipt

    def checkpoint(self):
        steps = self.trainer.steps
        if steps % 250:
            raise ValueError("R8 source checkpoint interval")
        payload = dict(schema="R8_SOURCE_JOURNAL_V1", snapshot=self.trainer.snapshot())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        slot = (steps // 250) % 2
        _replace(self.root / f"checkpoint.{slot}.pt", data)
        _replace(self.root / f"checkpoint.{slot}.json", json.dumps(
            dict(sha256=hashlib.sha256(data).hexdigest(), steps=steps), sort_keys=True).encode())
        if steps in SAVE_STEPS:
            self._archive_selected(steps, data)

    def _archive_selected(self, steps, data):
        selected = self.root / f"selected.{steps}.pt"
        metadata = self.root / f"selected.{steps}.json"
        expected = dict(sha256=hashlib.sha256(data).hexdigest(), steps=steps)
        if selected.exists() or metadata.exists():
            if not selected.is_file() or _digest(selected) != expected["sha256"]:
                raise ValueError("R8 selected source snapshot changed")
            if metadata.exists() and json.loads(metadata.read_text()) != expected:
                raise ValueError("R8 selected source snapshot metadata mismatch")
            if not metadata.exists():
                _replace(metadata, json.dumps(expected, sort_keys=True).encode())
        else:
            _replace(selected, data)
            _replace(metadata, json.dumps(expected, sort_keys=True).encode())

    def selected(self, steps):
        if steps not in SAVE_STEPS:
            raise ValueError("R8 unplanned source snapshot")
        path = self.root / f"selected.{steps}.pt"
        metadata = json.loads((self.root / f"selected.{steps}.json").read_text())
        data = path.read_bytes()
        if metadata != dict(sha256=hashlib.sha256(data).hexdigest(), steps=steps):
            raise ValueError("R8 selected source snapshot digest mismatch")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        snapshot = payload["snapshot"]
        if (payload.get("schema") != "R8_SOURCE_JOURNAL_V1" or snapshot.get("steps") != steps or
                snapshot.get("binding") != self.trainer.binding or
                snapshot.get("source_seed") != self.trainer.source_seed):
            raise ValueError("R8 selected source snapshot identity mismatch")
        return snapshot

    def recover_once(self, failure):
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not isinstance(failure.get("evidence"), dict) or
                not failure["evidence"]):
            raise ValueError("R8 only evidenced infrastructure failure may recover")
        if (not self.root.is_dir() or not self.physical.is_file() or
                (self.root / "recovery.json").exists() or (self.root / "fit_complete.json").exists()):
            raise ValueError("R8 source recovery unavailable or already used")
        valid = []
        for slot in (0, 1):
            try:
                data = (self.root / f"checkpoint.{slot}.pt").read_bytes()
                meta = json.loads((self.root / f"checkpoint.{slot}.json").read_text())
                if hashlib.sha256(data).hexdigest() != meta["sha256"]:
                    continue
                payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
                snapshot = payload["snapshot"]
                if (payload["schema"] == "R8_SOURCE_JOURNAL_V1" and
                        type(snapshot.get("steps")) is int and snapshot["steps"] % 250 == 0 and
                        snapshot["steps"] == meta["steps"]):
                    valid.append((snapshot, data))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pass
        if not valid:
            raise ValueError("R8 no verified equivalent source snapshot")
        chosen, data = max(valid, key=lambda item: item[0]["steps"])
        if chosen["steps"] in SAVE_STEPS:
            self._archive_selected(chosen["steps"], data)
        self.trainer.restore(chosen)
        self.previous_counts = COUNTS.copy()
        _replace(self.root / "recovery.json", json.dumps(
            dict(schema="R8_SOURCE_RECOVERY_V1", binding=self.trainer.binding,
                 source_seed=self.trainer.source_seed, steps=self.trainer.steps,
                 failure=failure,
                 physical_log_bytes=self.physical.stat().st_size), sort_keys=True).encode())
        return self.trainer.steps
