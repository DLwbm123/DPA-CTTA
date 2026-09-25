"""Durable shared source-fit scaler observations with physical-work accounting."""
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import torch

from ..r7_shared.numerics import COUNTS
from .journal import _replace, _sync_dir
from .preparation import scaler_anchor
from .schedule import anchors


def _json(row):
    return (json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode()


def _scan(raw, start, completed, groups, tail=False):
    if raw and not raw.endswith(b"\n"):
        raise ValueError("R8 scaler physical log truncated")
    rows = [json.loads(line) for line in raw.splitlines()]
    expected = 0
    ordinal = start
    totals = Counter()
    for position, row in enumerate(rows):
        if ordinal >= start + completed + (1 if tail else 0):
            raise ValueError("R8 scaler extra physical work")
        counts = row.get("counts")
        if (row.get("anchor") != ordinal or row.get("group") != groups[expected] or
                not isinstance(counts, dict) or
                any(type(value) is not int or value < 0 for value in counts.values())):
            raise ValueError("R8 scaler physical identity/cost")
        totals.update(counts)
        if row.get("status") == "FAILED_CALL":
            if not tail or position != len(rows) - 1 or ordinal != start + completed:
                raise ValueError("R8 scaler failed call position")
            continue
        if row.get("visit") != expected + 1:
            raise ValueError("R8 scaler physical visit sequence")
        expected += 1
        if expected == len(groups):
            expected = 0
            ordinal += 1
    if ((ordinal != start + completed and
         not (tail and ordinal == start + completed + 1 and expected == 0)) or
            (not tail and expected)):
        raise ValueError("R8 scaler committed coverage")
    return totals


class ScalerJournal:
    def __init__(self, root, segmenter, data, binding):
        if not isinstance(binding, str) or len(binding) != 64 or not data.folds["fit"]:
            raise ValueError("R8 scaler identity")
        self.root = Path(root)
        self.segmenter, self.data = segmenter, data
        self.identity = dict(binding=binding, fold="fit", zero_film=True)
        self.bank = anchors("fit")
        self.groups = tuple(data.folds["fit"])
        self.physical = self.root / "physical.jsonl"
        self.results = []
        self.active = False
        self.failed = False

    def _totals(self, raw, count):
        recovery = self.root / "recovery.json"
        if not recovery.exists():
            return _scan(raw, 0, count, self.groups)
        row = json.loads(recovery.read_text())
        cut, boundary = row["physical_log_bytes"], row["next_anchor"]
        before = row["snapshot_physical_bytes"]
        if len(raw) <= before:
            return _scan(raw, 0, count, self.groups)
        if (row.get("schema") != "R8_SCALER_RECOVERY_V1" or
                row.get("identity") != self.identity or
                not 0 <= before <= cut <= len(raw) or count < boundary):
            raise ValueError("R8 scaler recovery identity/offset")
        totals = _scan(raw[:before], 0, boundary, self.groups)
        totals.update(_scan(raw[before:cut], boundary, 0, self.groups, tail=True))
        totals.update(_scan(raw[cut:], boundary, count - boundary, self.groups))
        return totals

    def _snapshot(self):
        raw = self.physical.read_bytes()
        if dict(self._totals(raw, len(self.results))) != dict(COUNTS):
            raise ValueError("R8 scaler physical counters disagree")
        payload = dict(schema="R8_SCALER_SNAPSHOT_V1", identity=self.identity,
                       groups=self.groups, next_anchor=len(self.results),
                       results=self.results, cpu_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all(), counts=dict(COUNTS),
                       physical_bytes=len(raw),
                       physical_sha256=hashlib.sha256(raw).hexdigest())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        slot = len(self.results) % 2
        _replace(self.root / f"checkpoint.{slot}.pt", data)
        _replace(self.root / f"checkpoint.{slot}.json", _json(dict(
            sha256=hashlib.sha256(data).hexdigest(), next_anchor=len(self.results))))

    def create(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.physical.touch(exist_ok=False)
        _sync_dir(self.root)
        COUNTS.clear()
        self._snapshot()
        self.active = True

    def _validated(self, slot):
        data = (self.root / f"checkpoint.{slot}.pt").read_bytes()
        meta = json.loads((self.root / f"checkpoint.{slot}.json").read_text())
        if meta.get("sha256") != hashlib.sha256(data).hexdigest():
            raise ValueError("R8 scaler snapshot digest")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        count = payload["next_anchor"]
        raw = self.physical.read_bytes()
        if (payload.get("schema") != "R8_SCALER_SNAPSHOT_V1" or
                payload.get("identity") != self.identity or
                tuple(payload.get("groups", ())) != self.groups or
                type(count) is not int or not 0 <= count <= len(self.bank) or
                len(payload["results"]) != count or meta.get("next_anchor") != count or
                len(raw) < payload["physical_bytes"] or
                hashlib.sha256(raw[:payload["physical_bytes"]]).hexdigest() !=
                payload["physical_sha256"]):
            raise ValueError("R8 scaler snapshot identity/prefix")
        if dict(self._totals(raw[:payload["physical_bytes"]], count)) != payload["counts"]:
            raise ValueError("R8 scaler snapshot counters")
        for result in payload["results"]:
            if (result.shape != (len(self.groups), 134) or result.dtype != torch.float32 or
                    result.device.type != "cpu" or not torch.isfinite(result).all()):
                raise ValueError("R8 scaler snapshot result")
        return payload

    def recover_once(self, failure):
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not isinstance(failure.get("evidence"), dict) or
                not failure["evidence"] or not self.physical.is_file() or
                (self.root / "recovery.json").exists() or
                (self.root / "scaler_complete.json").exists()):
            raise ValueError("R8 scaler recovery unavailable")
        valid = []
        for slot in (0, 1):
            try:
                valid.append(self._validated(slot))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pass
        if not valid:
            raise ValueError("R8 no equivalent scaler snapshot")
        chosen = max(valid, key=lambda row: row["next_anchor"])
        raw = self.physical.read_bytes()
        cut = chosen["physical_bytes"]
        tail_counts = _scan(raw[cut:], chosen["next_anchor"], 0, self.groups, tail=True)
        tail_rows = [json.loads(line) for line in raw[cut:].splitlines()]
        if (tail_rows and tail_rows[-1].get("status") == "FAILED_CALL" and
                tail_rows[-1].get("error_type") not in
                ("OSError", "ConnectionError", "InterruptedError", "TimeoutError")):
            raise ValueError("R8 numerical/resource failure cannot recover")
        self.results = chosen["results"]
        torch.set_rng_state(chosen["cpu_rng"])
        torch.cuda.set_rng_state_all(chosen["cuda_rng"])
        totals = Counter(chosen["counts"])
        totals.update(tail_counts)
        COUNTS.clear()
        COUNTS.update(totals)
        _replace(self.root / "recovery.json", _json(dict(
            schema="R8_SCALER_RECOVERY_V1", identity=self.identity,
            next_anchor=len(self.results), physical_log_bytes=len(raw),
            snapshot_physical_bytes=cut, failure=failure)))
        self.active = True
        return len(self.results)

    def run_next(self, guard):
        if (not self.active or self.failed or not callable(guard) or
                len(self.results) >= len(self.bank)):
            raise ValueError("R8 scaler next anchor")
        index = len(self.results)
        previous = COUNTS.copy()
        logged = 0

        def group_done(group):
            nonlocal previous, logged
            guard()
            counts = dict(COUNTS - previous)
            with self.physical.open("ab") as stream:
                stream.write(_json(dict(anchor=index, group=group,
                                        visit=logged + 1, counts=counts)))
                stream.flush()
                os.fsync(stream.fileno())
            previous, logged = COUNTS.copy(), logged + 1

        hook = self.segmenter.register_forward_pre_hook(lambda *_: guard())
        try:
            guard()
            result = scaler_anchor(self.segmenter, self.data, index,
                                   self.bank[index], on_group=group_done)
        except BaseException as exc:
            self.failed = True
            with self.physical.open("ab") as stream:
                stream.write(_json(dict(anchor=index, group=self.groups[logged],
                                        visit=logged + 1, status="FAILED_CALL",
                                        counts=dict(COUNTS - previous),
                                        error_type=type(exc).__name__, error=str(exc)[:3000])))
                stream.flush()
                os.fsync(stream.fileno())
            raise
        finally:
            hook.remove()
        self.results.append(result)
        try:
            self._snapshot()
        except BaseException:
            self.failed = True
            raise
        return index + 1

    def complete(self):
        if len(self.results) != len(self.bank) or (self.root / "scaler_complete.json").exists():
            raise ValueError("R8 scaler completion coverage")
        payload = self._validated(len(self.bank) % 2)
        values = torch.cat(payload["results"])
        buffer = io.BytesIO()
        torch.save(dict(schema="R8_SCALER_V1", identity=self.identity,
                        groups=self.groups, values=values), buffer)
        _replace(self.root / "scaler.pt", buffer.getvalue())
        raw = self.physical.read_bytes()
        receipt = dict(schema="R8_SCALER_COMPLETE_V1", identity=self.identity,
                       anchors=len(self.bank), observations=len(values),
                       scaler_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                       physical_bytes=len(raw),
                       physical_sha256=hashlib.sha256(raw).hexdigest(),
                       physical_counts=dict(COUNTS),
                       recovered=(self.root / "recovery.json").exists())
        _replace(self.root / "scaler_complete.json", _json(receipt))
        return receipt


def run_scaler(root, segmenter, data, binding, guard, resume_failure=None):
    job = ScalerJournal(root, segmenter, data, binding)
    if resume_failure is None:
        job.create()
    else:
        job.recover_once(resume_failure)
    while len(job.results) < len(job.bank):
        job.run_next(guard)
    return job.complete()


def load_scaler(root, data, binding):
    root = Path(root)
    receipt = json.loads((root / "scaler_complete.json").read_text())
    raw = (root / "scaler.pt").read_bytes()
    physical = (root / "physical.jsonl").read_bytes()
    identity = dict(binding=binding, fold="fit", zero_film=True)
    if (receipt.get("schema") != "R8_SCALER_COMPLETE_V1" or
            receipt.get("identity") != identity or
            receipt.get("anchors") != len(anchors("fit")) or
            receipt.get("observations") != len(anchors("fit")) * len(data.folds["fit"]) or
            receipt.get("scaler_sha256") != hashlib.sha256(raw).hexdigest() or
            receipt.get("physical_bytes") != len(physical) or
            receipt.get("physical_sha256") != hashlib.sha256(physical).hexdigest()):
        raise ValueError("R8 scaler completion identity/digest")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    values = payload["values"]
    if (payload.get("schema") != "R8_SCALER_V1" or payload.get("identity") != identity or
            tuple(payload.get("groups", ())) != tuple(data.folds["fit"]) or
            values.shape != (receipt["observations"], 134) or
            values.dtype != torch.float32 or not torch.isfinite(values).all()):
        raise ValueError("R8 scaler payload identity")
    return values
