"""Attempt-local physical counters, independent of restored scientific counters."""
import json
import os
import signal
import threading
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from .ledger import Ledger, _cost
from .resources import CAPS

ACTIVE = None


class WorkerBudget:
    def __init__(self, ledger, attempt_id, root, budget, maximum_seconds):
        self.ledger, self.attempt_id, self.root = ledger, attempt_id, Path(root).resolve()
        self.budget = _cost(budget).copy()
        if (not 0 < maximum_seconds < self.budget["gpu_seconds"] or
                os.getpid() != os.getpgrp()):
            raise ValueError("R8 worker requires a dedicated process group and bounded deadline")
        self.maximum_seconds = maximum_seconds
        self.observed = dict.fromkeys(CAPS, 0)
        self.started = time.monotonic()
        self.baseline_disk_bytes = self._disk()
        self.last_shared_check = float("-inf")
        self.handles, self.models = [], set()
        self.done = threading.Event()

    def _disk(self):
        if not self.root.exists():
            return 0
        total = 0
        for path in self.root.rglob("*"):
            if path.is_symlink():
                self.ledger.stop("symlink in attempt output")
                raise ValueError("R8 output isolation failure")
            if path.is_file():
                total += path.stat().st_size
        return total

    def __call__(self):
        now = time.monotonic()
        self.observed["gpu_seconds"] = now - self.started
        if self.observed["gpu_seconds"] >= self.maximum_seconds:
            self.ledger.stop("worker deadline reached: " + self.attempt_id)
            raise RuntimeError("R8 GLOBAL STOP: worker deadline")
        if any(self.observed[key] >= self.budget[key] for key in CAPS if self.budget[key] > 0) or any(
                self.observed[key] > 0 and self.budget[key] == 0 for key in CAPS):
            self.ledger.stop("worker reservation exhausted: " + self.attempt_id)
            raise RuntimeError("R8 GLOBAL STOP: worker reservation")
        if now - self.last_shared_check >= 1:
            self.observed["disk_bytes"] = max(self.observed["disk_bytes"], self._disk() - self.baseline_disk_bytes)
            self.ledger.guard(self.attempt_id, self.observed)
            self.last_shared_check = now

    def operation(self, key):
        self.observed[key] += 1
        self()

    def attach_model(self, model):
        if id(model) not in self.models:
            self.models.add(id(model))
            self.handles.append(model.register_forward_pre_hook(lambda *_: self.operation("model_forwards")))

    def check_write(self, path, size):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            return  # Ledger metadata lives in its own pre-reserved launch directory.
        self.observed["disk_bytes"] = max(self.observed["disk_bytes"], self._disk() + size - self.baseline_disk_bytes)
        self()

    def _deadline(self):
        if not self.done.wait(self.maximum_seconds):
            # Stop this owned process group even if a GPU or NAS call never returns.
            # Reservations remain charged if the worker cannot seal its evidence.
            threading.Thread(target=lambda: self.ledger.stop("worker watchdog deadline: " + self.attempt_id),
                             daemon=True).start()
            os.killpg(os.getpgrp(), signal.SIGTERM)

    def __enter__(self):
        global ACTIVE
        if ACTIVE is not None:
            raise ValueError("R8 nested physical worker budget")
        state = self.ledger.snapshot()
        attempt = state["attempts"][self.attempt_id]
        if (attempt["reserved"] != self.budget or attempt["actual"] is not None or
                str(attempt["physical_gpu"]) != os.environ.get("CUDA_VISIBLE_DEVICES")):
            raise ValueError("R8 worker reservation identity")
        self()
        ACTIVE = self
        self.backward, self.grad = torch.autograd.backward, torch.autograd.grad
        def backward(*args, **kwargs):
            self.operation("backward_calls")
            return self.backward(*args, **kwargs)
        def grad(*args, **kwargs):
            self.operation("vjp_calls")
            return self.grad(*args, **kwargs)
        torch.autograd.backward, torch.autograd.grad = backward, grad
        self.handles.append(register_optimizer_step_pre_hook(
            lambda optimizer, *_: self.operation("optimizer_steps")
            if isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)) else None))
        self.watchdog = threading.Thread(target=self._deadline, daemon=True)
        self.watchdog.start()
        return self

    def __exit__(self, kind, error, traceback):
        global ACTIVE
        self.done.set()
        for handle in self.handles:
            handle.remove()
        torch.autograd.backward, torch.autograd.grad = self.backward, self.grad
        ACTIVE = None
        self.observed["gpu_seconds"] = time.monotonic() - self.started
        retained_disk = max(0, self._disk() - self.baseline_disk_bytes)
        peak_disk = max(self.observed["disk_bytes"], retained_disk)
        self.observed["disk_bytes"] = retained_disk
        # Failed attempts keep the full reservation. The measured attempt counters
        # are supplementary evidence; recovery must reserve fresh capacity.
        path = self.ledger.root / ("attempt-" + self.attempt_id + ".json")
        with path.open("x") as stream:
            json.dump(dict(schema="R8_PHYSICAL_ATTEMPT_V1", observed=self.observed,
                           baseline_disk_bytes=self.baseline_disk_bytes,
                           peak_disk_bytes=peak_disk,
                           status="COMPLETE" if kind is None else "FAILED",
                           error_type=None if kind is None else kind.__name__,
                           error=None if error is None else str(error)[:3000],
                           errno=getattr(error, "errno", None)), stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if kind is None:
            self.ledger.settle(self.attempt_id, self.observed, str(path))


def attach_model(model):
    if ACTIVE is not None:
        ACTIVE.attach_model(model)


def check_write(path, size):
    if ACTIVE is not None:
        ACTIVE.check_write(path, size)
