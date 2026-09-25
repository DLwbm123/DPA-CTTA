"""Shared admission ledger. Unsettled attempts retain their entire reservation."""
import fcntl
import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

from .journal import _replace
from .resources import CAPS, check_caps


def _cost(row):
    if (not isinstance(row, dict) or set(row) != set(CAPS) or
            any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for v in row.values()) or
            any(type(row[k]) is not int for k in CAPS if k != "gpu_seconds")):
        raise ValueError("R8 complete nonnegative physical cost required")
    return row


class Ledger:
    """One NAS lock and atomic JSON state, shared by the three GPU workers.

    Reserve a bounded work unit BEFORE running it or writing its outputs. Settle
    only after durable physical evidence is available. A crashed attempt is never
    refunded by recovery; replay needs a distinct reservation. Disk costs include
    retained outputs and the maximum temporary write footprint.
    """

    def __init__(self, root, identity):
        self.root = Path(root)
        if (set(identity) != {"code_sha", "protocol_sha256", "graph_sha256"} or
                any(not isinstance(v, str) or re.fullmatch(
                    r"[0-9a-f]{40}" if k == "code_sha" else r"[0-9a-f]{64}", v) is None
                    for k, v in identity.items())):
            raise ValueError("R8 ledger identity")
        self.identity = identity.copy()

    def create(self, prior_cost, evidence):
        _cost(prior_cost)
        check_caps(prior_cost)
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("R8 prior physical evidence required")
        self.root.mkdir(parents=True, exist_ok=False)
        (self.root / "lock").touch(exist_ok=False)
        self._save(dict(schema="R8_AGGREGATE_LEDGER_V1", identity=self.identity,
                        caps=CAPS, prior_cost=prior_cost, prior_evidence=evidence,
                        attempts={}, stop=None))

    def _save(self, state):
        _replace(self.root / "state.json",
                 json.dumps(state, sort_keys=True, allow_nan=False).encode())

    @contextmanager
    def _locked(self):
        # ponytail: one lock for <= 789 jobs; keep reservations at bounded work-unit boundaries.
        with (self.root / "lock").open("r+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads((self.root / "state.json").read_text())
            if (state.get("schema") != "R8_AGGREGATE_LEDGER_V1" or
                    state.get("identity") != self.identity or state.get("caps") != CAPS):
                state["stop"] = "identity or resource-cap binding mismatch"
                self._save(state)
                raise RuntimeError("R8 GLOBAL STOP: ledger identity mismatch")
            if state["stop"] is not None:
                raise RuntimeError("R8 GLOBAL STOP: " + state["stop"])
            yield state

    @staticmethod
    def total(state):
        total = _cost(state["prior_cost"]).copy()
        for attempt in state["attempts"].values():
            cost = _cost(attempt["actual"] if attempt["actual"] is not None
                         else attempt["reserved"])
            for key in CAPS:
                total[key] += cost[key]
        return total

    def reserve(self, attempt_id, physical_gpu, budget):
        _cost(budget)
        if (not isinstance(attempt_id, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,128}", attempt_id) is None or
                type(physical_gpu) is not int or physical_gpu not in (5, 6, 7) or
                budget["gpu_seconds"] <= 0):
            raise ValueError("R8 reservation identity or GPU")
        with self._locked() as state:
            if attempt_id in state["attempts"]:
                raise ValueError("R8 physical attempt cannot be reused")
            total = self.total(state)
            try:
                check_caps({key: total[key] + budget[key] for key in CAPS})
            except RuntimeError as exc:
                state["stop"] = str(exc)
                self._save(state)
                raise
            state["attempts"][attempt_id] = dict(physical_gpu=physical_gpu,
                reserved=budget.copy(), actual=None, evidence=None)
            self._save(state)

    def guard(self, attempt_id, observed):
        """Caller supplies cumulative ATTEMPT costs, never restored logical counters."""
        _cost(observed)
        with self._locked() as state:
            attempt = state["attempts"][attempt_id]
            if attempt["actual"] is not None:
                raise ValueError("R8 settled attempt cannot execute")
            if any(observed[key] >= attempt["reserved"][key]
                   for key in CAPS if attempt["reserved"][key] > 0) or any(
                    observed[key] > 0 and attempt["reserved"][key] == 0 for key in CAPS):
                state["stop"] = "attempt reservation exhausted: " + attempt_id
                self._save(state)
                raise RuntimeError("R8 GLOBAL STOP: attempt reservation exhausted")
            check_caps(self.total(state))

    def settle(self, attempt_id, actual, evidence):
        _cost(actual)
        if not isinstance(evidence, str) or not evidence:
            raise ValueError("R8 durable physical evidence required")
        with self._locked() as state:
            attempt = state["attempts"][attempt_id]
            if attempt["actual"] is not None:
                raise ValueError("R8 attempt already settled")
            if any(actual[key] > attempt["reserved"][key] for key in CAPS):
                # Keep the overrun evidence even though no further work is admitted.
                attempt.update(actual=actual.copy(), evidence=evidence)
                state["stop"] = "physical cost exceeded reservation: " + attempt_id
                self._save(state)
                raise RuntimeError("R8 GLOBAL STOP: reservation exceeded")
            attempt.update(actual=actual.copy(), evidence=evidence)
            self._save(state)

    def stop(self, reason):
        if not isinstance(reason, str) or not reason:
            raise ValueError("R8 global stop reason")
        with self._locked() as state:
            state["stop"] = reason
            self._save(state)

    def snapshot(self):
        with self._locked() as state:
            return dict(state, committed_and_reserved=self.total(state))
