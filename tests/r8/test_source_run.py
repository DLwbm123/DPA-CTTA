import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.source_run import run_fit


class FailingTrainer:
    binding = "synthetic"
    source_seed = 20260924

    def __init__(self):
        self.steps = 0
        self.segmenter = nn.Identity()

    def snapshot(self):
        return dict(steps=self.steps, binding=self.binding, source_seed=self.source_seed)

    def fit_step(self):
        self.segmenter(torch.ones(1))
        COUNTS["backbone_forwards"] += 1
        raise ValueError("synthetic numerical failure")


class TestSourceRun(unittest.TestCase):
    def test_failed_fit_call_records_partial_cost_and_removes_guard_hook(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = FailingTrainer()
            checks = []
            with self.assertRaisesRegex(ValueError, "synthetic numerical"):
                run_fit(trainer, Path(directory) / "job", lambda: checks.append(True))
            record = json.loads((Path(directory) / "job" / "physical.jsonl").read_text())
            self.assertEqual((record["status"], record["step"]), ("FAILED_CALL", 1))
            self.assertEqual(record["counts"]["backbone_forwards"], 1)
            self.assertEqual(len(checks), 2)
            self.assertEqual(len(trainer.segmenter._forward_pre_hooks), 0)


if __name__ == "__main__":
    unittest.main()
