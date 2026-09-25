"""Durable boundaries for independently initialized capacity probes and LR episodes."""
import hashlib
import io
import json
import os
import time
from pathlib import Path

import torch

from ..r7_shared.numerics import COUNTS
from .journal import _digest, _replace, _reject_recorded_noninfra_failure
from .rng import capture as capture_rng, restore as restore_rng


class IndependentJournal:
    def __init__(self, root, identity, items):
        if not identity or not items or len({json.dumps(row, sort_keys=True) for row in items}) != len(items):
            raise ValueError("R8 independent stage identity/items")
        self.root, self.identity, self.items = Path(root), identity, items
        self.rows = []
        self.physical = self.root / "physical.jsonl"

    def create(self):
        self.root.mkdir(parents=True, exist_ok=False)
        self.physical.touch(exist_ok=False)
        self.checkpoint()

    def checkpoint(self):
        packet = dict(schema="R8_INDEPENDENT_BOUNDARY_V1", identity=self.identity, items=self.items,
                      rows=self.rows, rng=capture_rng(), counts=dict(COUNTS),
                      physical_bytes=self.physical.stat().st_size, physical_sha256=_digest(self.physical))
        buffer = io.BytesIO()
        torch.save(packet, buffer)
        raw = buffer.getvalue()
        slot = len(self.rows) % 2
        _replace(self.root / f"checkpoint.{slot}.pt", raw)
        _replace(self.root / f"checkpoint.{slot}.json", json.dumps(dict(
            sha256=hashlib.sha256(raw).hexdigest(), completed=len(self.rows)), sort_keys=True).encode())

    def recover_once(self, failure):
        _reject_recorded_noninfra_failure(self.root)
        if (not isinstance(failure, dict) or failure.get("class") != "INFRASTRUCTURE" or
                not failure.get("reason") or not failure.get("evidence") or
                (self.root / "recovery.json").exists() or (self.root / "complete.json").exists()):
            raise ValueError("R8 independent recovery unavailable")
        valid = []
        for slot in (0, 1):
            try:
                meta = json.loads((self.root / f"checkpoint.{slot}.json").read_text())
                raw = (self.root / f"checkpoint.{slot}.pt").read_bytes()
                if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
                    continue
                packet = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
                count = len(packet["rows"])
                if (packet.get("schema") == "R8_INDEPENDENT_BOUNDARY_V1" and
                        packet.get("identity") == self.identity and packet.get("items") == self.items and
                        0 <= count <= len(self.items) and meta["completed"] == count and
                        [row["item"] for row in packet["rows"]] == self.items[:count] and
                        _digest(self.physical, packet["physical_bytes"]) == packet["physical_sha256"]):
                    valid.append(packet)
            except (OSError, ValueError, KeyError, TypeError, RuntimeError):
                pass
        if not valid:
            raise ValueError("R8 no verified independent boundary")
        packet = max(valid, key=lambda row: len(row["rows"]))
        physical = self.physical.read_bytes()
        if not physical.endswith(b"\n") and physical:
            raise ValueError("R8 physical work log truncated")
        if physical:
            last = json.loads(physical.splitlines()[-1])
            if last.get("status") == "FAILED_CALL" and last.get("error_type") not in (
                    "OSError", "TimeoutError", "ConnectionError", "InterruptedError"):
                raise ValueError("R8 non-infrastructure independent failure")
        self.rows = packet["rows"]
        restore_rng(packet["rng"])
        COUNTS.clear()
        COUNTS.update(packet["counts"])
        _replace(self.root / "recovery.json", json.dumps(dict(identity=self.identity, failure=failure,
            completed=len(self.rows), physical_bytes=len(physical)), sort_keys=True).encode())

    def _append(self, row):
        with self.physical.open("a") as output:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

    def run(self, call, guard):
        while len(self.rows) < len(self.items):
            guard()
            index, before, started = len(self.rows), COUNTS.copy(), time.monotonic()
            item, logged = self.items[index], False
            try:
                result = call(item)
                json.dumps(result, allow_nan=False)
                self._append(dict(index=index, status="COMPLETE", counts=dict(COUNTS - before),
                                  seconds=time.monotonic() - started))
                logged = True
                self.rows.append(dict(item=item, result=result))
                self.checkpoint()
            except BaseException as exc:
                self._append(dict(index=index, status="FAILED_CALL", counts={} if logged else dict(COUNTS - before),
                                  error_type=type(exc).__name__, seconds=time.monotonic() - started))
                with (self.root / "worker_failures.jsonl").open("a") as output:
                    output.write(json.dumps(dict(error_type=type(exc).__name__, error=str(exc)[:3000])) + "\n")
                    output.flush()
                    os.fsync(output.fileno())
                raise
        _replace(self.root / "results.json", json.dumps(self.rows, sort_keys=True, allow_nan=False).encode())
        receipt = dict(schema="R8_INDEPENDENT_COMPLETE_V1", identity=self.identity, completed=len(self.items),
                       results_sha256=_digest(self.root / "results.json"), physical_sha256=_digest(self.physical))
        _replace(self.root / "complete.json", json.dumps(receipt, sort_keys=True).encode())
        return receipt
