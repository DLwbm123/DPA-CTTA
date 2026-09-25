"""Source-only oracle preparation with atomic anchor boundaries and physical costs."""
import hashlib
import io
import json
import os
from collections import Counter
from pathlib import Path

import torch

from ..r7_shared.numerics import COUNTS
from .journal import _replace, _sync_dir, _reject_recorded_noninfra_failure
from .oracles import Oracles, oracle_one
from .schedule import SIZES, oracle_roles

ORDER = tuple((fold, index) for fold in ("fit", "cal", "val")
              for index in range(SIZES[fold]))


def _json(data):
    return (json.dumps(data, sort_keys=True, allow_nan=False) + "\n").encode()


def _lines(raw, start, completed, allow_tail=False):
    if raw and not raw.endswith(b"\n"):
        raise ValueError("R8 oracle physical log truncated")
    expected = [0] * (len(ORDER) - start)
    rows = [json.loads(line) for line in raw.splitlines()]
    totals = Counter()
    previous_ordinal = start
    for i, row in enumerate(rows):
        ordinal = row.get("ordinal")
        counts = row.get("counts")
        if (type(ordinal) is not int or ordinal not in range(start, len(ORDER)) or
                row.get("fold") != ORDER[ordinal][0] or row.get("index") != ORDER[ordinal][1] or
                not isinstance(counts, dict) or
                any(type(v) is not int or v < 0 for v in counts.values())):
            raise ValueError("R8 oracle physical identity/cost")
        if (ordinal < previous_ordinal or ordinal > previous_ordinal + 1 or
                (ordinal > previous_ordinal and expected[previous_ordinal - start] != 256)):
            raise ValueError("R8 oracle physical anchor order")
        previous_ordinal = ordinal
        totals.update(counts)
        if row.get("status") == "FAILED_CALL":
            if not allow_tail or i != len(rows) - 1 or ordinal != start + completed:
                raise ValueError("R8 oracle failed call position")
            continue
        if row.get("step") != expected[ordinal - start] + 1:
            raise ValueError("R8 oracle physical step sequence")
        expected[ordinal - start] += 1
        if expected[ordinal - start] > 256:
            raise ValueError("R8 oracle repeated physical step")
    if expected[:completed] != [256] * completed or any(expected[completed + 1:]):
        raise ValueError("R8 oracle committed step coverage")
    if not allow_tail and (any(expected[completed:]) or len(rows) != completed * 256):
        raise ValueError("R8 oracle extra physical work")
    return totals


def _physical_totals(raw, count, recovery=None, allow_tail=False):
    if recovery is None or len(raw) <= recovery["snapshot_physical_bytes"]:
        return _lines(raw, 0, count, allow_tail)
    start = recovery["next_ordinal"]
    before = recovery["snapshot_physical_bytes"]
    cut = recovery["physical_log_bytes"]
    if not 0 <= before <= cut <= len(raw) or count < start:
        raise ValueError("R8 oracle recovery physical offset")
    totals = _lines(raw[:before], 0, start)
    totals.update(_lines(raw[before:cut], start, 0, allow_tail=True))
    totals.update(_lines(raw[cut:], start, count - start, allow_tail))
    return totals


class OracleJournal:
    def __init__(self, root, segmenter, data, binding):
        if (not isinstance(binding, str) or len(binding) != 64 or
                segmenter.amplitude not in (0.1, 0.3)):
            raise ValueError("R8 oracle identity")
        self.root = Path(root)
        self.segmenter, self.data = segmenter, data
        self.identity = dict(binding=binding, amplitude=segmenter.amplitude)
        self.physical = self.root / "physical.jsonl"
        self.results = []
        self.active = False
        self.failed = False

    def _snapshot(self):
        count = len(self.results)
        raw = self.physical.read_bytes()
        recovery_path = self.root / "recovery.json"
        recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
        totals = _physical_totals(raw, count, recovery)
        if dict(totals) != dict(COUNTS):
            raise ValueError("R8 oracle physical counters disagree")
        payload = dict(schema="R8_ORACLE_SNAPSHOT_V1", identity=self.identity,
                       next_ordinal=count, results=self.results,
                       # Every new anchor starts at zero latent and fresh Adam; no optimizer crosses this boundary.
                       cpu_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all(),
                       counts=dict(COUNTS), physical_bytes=len(raw),
                       physical_sha256=hashlib.sha256(raw).hexdigest())
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        data = buffer.getvalue()
        slot = count % 2
        _replace(self.root / f"checkpoint.{slot}.pt", data)
        _replace(self.root / f"checkpoint.{slot}.json", _json(dict(
            sha256=hashlib.sha256(data).hexdigest(), next_ordinal=count)))

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
            raise ValueError("R8 oracle snapshot digest")
        payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        count = payload["next_ordinal"]
        raw = self.physical.read_bytes()
        if (payload.get("schema") != "R8_ORACLE_SNAPSHOT_V1" or
                payload.get("identity") != self.identity or
                type(count) is not int or not 0 <= count <= len(ORDER) or
                len(payload["results"]) != count or meta["next_ordinal"] != count or
                len(raw) < payload["physical_bytes"] or
                hashlib.sha256(raw[:payload["physical_bytes"]]).hexdigest() !=
                payload["physical_sha256"]):
            raise ValueError("R8 oracle snapshot identity/prefix")
        recovery_path = self.root / "recovery.json"
        recovery = json.loads(recovery_path.read_text()) if recovery_path.exists() else None
        totals = _physical_totals(raw[:payload["physical_bytes"]], count, recovery)
        if dict(totals) != payload["counts"]:
            raise ValueError("R8 oracle snapshot counters")
        for ordinal, (value, support, query, diagnostics) in enumerate(payload["results"]):
            fold, index = ORDER[ordinal]
            if (value.shape != (1024,) or value.dtype != torch.float32 or
                    value.device.type != "cpu" or not torch.isfinite(value).all() or
                    tuple((*support, *query)) != oracle_roles(self.data.folds[fold], fold, index) or
                    [d["step"] for d in diagnostics] != [16, 64, 256]):
                raise ValueError("R8 oracle snapshot result identity")
        return payload

    def recover_once(self, failure):
        _reject_recorded_noninfra_failure(self.root)
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not isinstance(failure.get("evidence"), dict) or
                not failure["evidence"] or not self.physical.is_file() or
                (self.root / "recovery.json").exists() or
                (self.root / "oracle_complete.json").exists()):
            raise ValueError("R8 oracle recovery unavailable")
        valid = []
        for slot in (0, 1):
            try:
                valid.append(self._validated(slot))
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pass
        if not valid:
            raise ValueError("R8 no equivalent oracle snapshot")
        chosen = max(valid, key=lambda row: row["next_ordinal"])
        raw = self.physical.read_bytes()
        cut = chosen["physical_bytes"]
        # The uncommitted tail may contain only work on the next independent anchor.
        tail_counts = _lines(raw[cut:], chosen["next_ordinal"], 0, allow_tail=True)
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
            schema="R8_ORACLE_RECOVERY_V1", identity=self.identity,
            next_ordinal=len(self.results), physical_log_bytes=len(raw),
            snapshot_physical_bytes=cut, failure=failure)))
        self.active = True
        return len(self.results)

    def run_next(self, guard):
        if (not self.active or self.failed or not callable(guard) or
                len(self.results) >= len(ORDER)):
            raise ValueError("R8 oracle next anchor")
        ordinal = len(self.results)
        fold, index = ORDER[ordinal]
        previous = COUNTS.copy()
        logged = 0

        def step_done(step):
            nonlocal previous, logged
            guard()
            counts = dict(COUNTS - previous)
            with self.physical.open("ab") as stream:
                stream.write(_json(dict(ordinal=ordinal, fold=fold, index=index,
                                        step=step, counts=counts)))
                stream.flush()
                os.fsync(stream.fileno())
            previous, logged = COUNTS.copy(), step

        hook = self.segmenter.register_forward_pre_hook(lambda *_: guard())
        try:
            guard()
            result = oracle_one(self.segmenter, self.data, fold, index, on_step=step_done)
            self.results.append(result)
            self._snapshot()
            return ordinal + 1
        except BaseException as exc:
            self.failed = True
            with self.physical.open("ab") as stream:
                stream.write(_json(dict(ordinal=ordinal, fold=fold, index=index,
                                        step=logged + 1, status="FAILED_CALL",
                                        counts=dict(COUNTS - previous),
                                        error_type=type(exc).__name__, error=str(exc)[:3000])))
                stream.flush()
                os.fsync(stream.fileno())
            raise
        finally:
            hook.remove()

    def complete(self):
        if len(self.results) != len(ORDER) or (self.root / "oracle_complete.json").exists():
            raise ValueError("R8 oracle completion coverage")
        final = self._validated(len(ORDER) % 2)
        raw = self.physical.read_bytes()
        recovery = self.root / "recovery.json"
        record = json.loads(recovery.read_text()) if recovery.exists() else None
        if record is not None and (record.get("schema") != "R8_ORACLE_RECOVERY_V1" or
                                   record.get("identity") != self.identity):
            raise ValueError("R8 oracle recovery identity")
        if dict(_physical_totals(raw, len(ORDER), record)) != dict(COUNTS):
            raise ValueError("R8 oracle final physical counters")
        folds = {}
        for fold in SIZES:
            rows = [row for row, (name, _) in zip(final["results"], ORDER) if name == fold]
            folds[fold] = Oracles(fold, self.segmenter.amplitude,
                                  torch.stack([row[0] for row in rows], 1),
                                  tuple(row[1] for row in rows), tuple(row[2] for row in rows),
                                  tuple(tuple(row[3]) for row in rows)).validate(self.data)
        buffer = io.BytesIO()
        torch.save(dict(schema="R8_ORACLES_V1", identity=self.identity,
                        folds={fold: vars(rows) for fold, rows in folds.items()}), buffer)
        _replace(self.root / "oracles.pt", buffer.getvalue())
        receipt = dict(schema="R8_ORACLE_COMPLETE_V1", identity=self.identity,
                       anchors=len(ORDER), oracles_sha256=hashlib.sha256(buffer.getvalue()).hexdigest(),
                       physical_bytes=len(raw), physical_sha256=hashlib.sha256(raw).hexdigest(),
                       physical_counts=dict(COUNTS), recovered=recovery.exists())
        _replace(self.root / "oracle_complete.json", _json(receipt))
        return receipt


def run_oracles(root, segmenter, data, binding, guard, resume_failure=None):
    journal = OracleJournal(root, segmenter, data, binding)
    if resume_failure is None:
        journal.create()
    else:
        journal.recover_once(resume_failure)
    while len(journal.results) < len(ORDER):
        journal.run_next(guard)
    return journal.complete()


def load_oracles(root, data, binding, amplitude):
    root = Path(root)
    receipt = json.loads((root / "oracle_complete.json").read_text())
    raw = (root / "oracles.pt").read_bytes()
    physical = (root / "physical.jsonl").read_bytes()
    identity = dict(binding=binding, amplitude=amplitude)
    if (receipt.get("schema") != "R8_ORACLE_COMPLETE_V1" or
            receipt.get("identity") != identity or receipt.get("anchors") != len(ORDER) or
            receipt.get("oracles_sha256") != hashlib.sha256(raw).hexdigest() or
            receipt.get("physical_bytes") != len(physical) or
            receipt.get("physical_sha256") != hashlib.sha256(physical).hexdigest()):
        raise ValueError("R8 oracle completion identity/digest")
    payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (payload.get("schema") != "R8_ORACLES_V1" or
            payload.get("identity") != identity or set(payload["folds"]) != set(SIZES)):
        raise ValueError("R8 oracle payload identity")
    return {fold: Oracles(**payload["folds"][fold]).validate(data) for fold in SIZES}
