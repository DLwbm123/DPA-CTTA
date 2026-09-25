import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dpa_ctta.r8_ba.calibration_journal import CalibrationJournal


FAILURE = {"class": "INFRASTRUCTURE", "reason": "synthetic interruption",
           "evidence": {"exit_code": 137}}


class FakeMethod(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("weight", torch.ones(1))
        self.stage = "cal"

    def freeze(self):
        self.stage = "online"

    def digest(self):
        return hashlib.sha256(self.weight.numpy().tobytes()).hexdigest()


class FakeCalibrator:
    binding = "synthetic"

    def __init__(self):
        self.segmenter = nn.Identity()
        self.method = FakeMethod()
        self.steps = 0

    def snapshot(self):
        return dict(steps=self.steps, binding=self.binding,
                    method=self.method.state_dict())

    def restore(self, snapshot):
        self.steps = snapshot["steps"]
        self.method.load_state_dict(snapshot["method"])


class TestCalibrationJournal(unittest.TestCase):
    def test_equivalent_recovery_and_full_calibration_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            root.mkdir()
            selected = root / "selected.1000.pt"
            selected.write_bytes(b"synthetic selected point")
            digest = hashlib.sha256(selected.read_bytes()).hexdigest()
            (root / "fit_complete.json").write_text(json.dumps(dict(
                schema="R8_SOURCE_FIT_COMPLETE_V1", binding="synthetic", steps=16000,
                selected_sha256={"1000": digest})))
            journal = CalibrationJournal(root, 1000, FakeCalibrator())
            journal.create()
            journal.calibrator.steps = 500
            journal.checkpoint()
            journal.physical.write_text("".join(json.dumps(dict(step=i, counts={"forward": 4})) + "\n"
                                                for i in range(1, 521)))
            resumed = CalibrationJournal(root, 1000, FakeCalibrator())
            with self.assertRaisesRegex(ValueError, "only evidenced infrastructure"):
                resumed.recover_once(dict(FAILURE, **{"class": "NUMERICAL"}))
            self.assertEqual(resumed.recover_once(FAILURE), 500)
            with resumed.physical.open("a") as stream:
                stream.write("".join(json.dumps(dict(step=i, counts={"forward": 4})) + "\n"
                                     for i in range(501, 1025)))
            resumed.calibrator.steps = 1024
            receipt = resumed.complete()
            self.assertEqual(receipt["steps"], 1024)
            self.assertEqual(resumed.calibrator.method.stage, "online")
            self.assertTrue((resumed.root / "calibrated.pt").is_file())
            with self.assertRaisesRegex(ValueError, "already used"):
                resumed.recover_once(FAILURE)


if __name__ == "__main__":
    unittest.main()
