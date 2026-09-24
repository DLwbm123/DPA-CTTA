"""Image-only R8 A/B/MLP deployment with transactional state commit."""
import copy

import torch

from ..r7_shared.context import stamp
from ..r7_shared.numerics import COUNTS, finite
from .context import capture, require
from .rng import capture as capture_rng, restore as restore_rng

ABLATIONS = {"A": ("RESET_HISTORY", "ISOTROPIC_R"),
             "B": ("RESET_HISTORY", "PRED_ONLY", "ISTA_20")}


class OnlineHost:
    def __init__(self, segmenter, method, config, source, expected_context, ablation=None):
        self.segmenter, self.method, self.ablation = segmenter, method, ablation
        if method is None and ablation is not None:
            raise ValueError("C0 has no ablation")
        if method is not None and (ablation is not None and
                                   (ablation not in ABLATIONS.get(method.group, ()) or method.static)):
            raise ValueError("R8 FULL-only registered ablation")
        actual = capture(segmenter, method, config, source)
        require(actual, expected_context)
        self.context = copy.deepcopy(expected_context)
        self.amplitude = segmenter.amplitude
        self.science_signature = self._science_signature()
        self.stamp = stamp(segmenter, method, ablation)
        self.state = None if method is None else method.initial()
        self.visits = 0
        self.failed = False

    def check_frozen(self, boundary=False):
        if self.segmenter.amplitude != self.amplitude or self._science_signature() != self.science_signature:
            raise ValueError("R8 scientific configuration changed")
        if stamp(self.segmenter, self.method, self.ablation) != self.stamp:
            raise ValueError("R8 inference weights/policy changed")
        if boundary:
            actual = capture(self.segmenter, self.method, self.context["payload"]["config"],
                             self.context["payload"]["source"])
            require(actual, self.context)

    def _science_signature(self):
        if self.method is None:
            return None
        method = self.method
        return (type(method).__name__, method.rank, method.amplitude, method.static,
                getattr(method, "observation", None), getattr(method, "aux_multiplier", None))

    @torch.no_grad()
    def step(self, current_image):
        if self.failed:
            raise RuntimeError("R8 host stopped at first error")
        before = COUNTS.copy()
        try:
            self.check_frozen()
            if self.method is None:
                logits, next_state = self.segmenter(current_image), None
            else:
                _, raw, tokens = self.segmenter(current_image, observe=True)
                next_state, _ = self.method.update(raw, tokens, self.state, ablation=self.ablation)
                self.method.validate_state(next_state)
                logits = self.segmenter(current_image, self.method.ambient(next_state))
            finite(logits)
            if logits.shape != (1, 2, 512, 512):
                raise ValueError("R8 prediction shape")
            self.state = next_state
            self.visits += 1
            return logits.detach(), dict(visit=self.visits, state_committed=True, counts=dict(COUNTS - before))
        except Exception:
            self.failed = True
            raise

    def snapshot(self):
        if self.failed:
            raise ValueError("R8 failed host cannot snapshot")
        self.check_frozen(boundary=True)
        return dict(schema="R8_TARGET_HOST_SNAPSHOT_V1", context_sha256=self.context["sha256"],
                    visits=self.visits, state=copy.deepcopy(self.state),
                    rng=capture_rng(),
                    physical_counts=dict(COUNTS))

    def restore(self, snapshot):
        if (snapshot.get("schema") != "R8_TARGET_HOST_SNAPSHOT_V1" or
                snapshot.get("context_sha256") != self.context["sha256"] or
                type(snapshot.get("visits")) is not int or snapshot["visits"] < 0):
            raise ValueError("R8 target snapshot identity")
        self.check_frozen(boundary=True)
        state = snapshot["state"]
        if self.method is None:
            if state is not None:
                raise ValueError("C0 snapshot state")
        else:
            self.method.validate_state(state)
            if state["counter"] != snapshot["visits"]:
                raise ValueError("R8 target visit/state offset mismatch")
        self.state = copy.deepcopy(state)
        self.visits = snapshot["visits"]
        restore_rng(snapshot["rng"])
        COUNTS.clear()
        COUNTS.update(snapshot["physical_counts"])
        self.failed = False
