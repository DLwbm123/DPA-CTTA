"""R8 journal envelope around the unchanged, inventory-verified R7 C host."""
import copy

from ..r7_shared.context import json_digest
from ..r7_shared.numerics import COUNTS
from .protocol import PROTOCOL_SHA256
from .rng import capture as capture_rng, restore as restore_rng
from .worker_budget import attach_model


class FrozenR7Host:
    def __init__(self, native, arm, code_sha, inventory_sha256):
        if (arm not in ("R7_C_FULL", "R7_C_STATIC") or native.method.group != "C" or
                native.method.static != arm.endswith("STATIC") or native.visits != 0):
            raise ValueError("R8 frozen R7 control identity")
        self.native = native
        attach_model(native.segmenter.model)
        payload = dict(schema="R8_FROZEN_R7_CONTROL_V1", arm=arm, code_sha=code_sha,
                       protocol_sha256=PROTOCOL_SHA256, inventory_sha256=inventory_sha256,
                       native_context=copy.deepcopy(native._context))
        self.context = dict(payload=payload, sha256=json_digest(payload))

    @property
    def visits(self):
        return self.native.visits

    def step(self, image):
        return self.native.step(image)

    def check_frozen(self, boundary=False):
        self.native._check_frozen(boundary=boundary)

    def snapshot(self):
        if self.native.failed:
            raise ValueError("R8 failed control cannot snapshot")
        return dict(schema="R8_FROZEN_R7_SNAPSHOT_V1", context_sha256=self.context["sha256"],
                    visits=self.visits, native=self.native.save_state(), rng=capture_rng(),
                    physical_counts=dict(COUNTS))

    def restore(self, snapshot):
        if (snapshot.get("schema") != "R8_FROZEN_R7_SNAPSHOT_V1" or
                snapshot.get("context_sha256") != self.context["sha256"] or
                snapshot.get("visits") != snapshot.get("native", {}).get("visits")):
            raise ValueError("R8 frozen control snapshot identity")
        self.native.load_state(snapshot["native"])
        restore_rng(snapshot["rng"])
        COUNTS.clear()
        COUNTS.update(snapshot["physical_counts"])
