import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_shared.source import Record, SourceData
from dpa_ctta.r8_ba.scaler_journal import ScalerJournal, load_scaler, run_scaler
from dpa_ctta.r8_ba.schedule import anchors


class TinySegmenter(nn.Module):
    def forward(self, image, observe=False):
        COUNTS["backbone_forwards"] += 1
        return None, image.mean().expand(134).clone(), None


class TestScalerJournal(unittest.TestCase):
    @staticmethod
    def data():
        folds = dict(fit=[f"fit{i}" for i in range(32)],
                     cal=[f"cal{i}" for i in range(4)],
                     val=[f"val{i}" for i in range(4)])
        records = [Record(group, fold, torch.rand(1, 3, 8, 8),
                          torch.zeros(1, 2, 8, 8))
                   for fold, groups in folds.items() for group in groups]
        return SourceData(records, folds)

    def test_partial_anchor_recovery_and_seal(self):
        data, model = self.data(), TinySegmenter()
        bank = anchors("fit")[:2]
        with tempfile.TemporaryDirectory() as directory, \
                patch("dpa_ctta.r8_ba.scaler_journal.anchors", return_value=bank):
            root = Path(directory) / "scaler"
            job = ScalerJournal(root, model, data, "a" * 64)
            job.create()
            self.assertEqual(job.run_next(lambda: None), 1)
            first = job.results[0].clone()
            calls = 0

            def interrupted():
                nonlocal calls
                calls += 1
                if calls == 8:
                    raise TimeoutError("synthetic infrastructure interruption")

            with self.assertRaises(TimeoutError):
                job.run_next(interrupted)
            recovered = ScalerJournal(root, model, data, "a" * 64)
            failure = {"class": "INFRASTRUCTURE", "reason": "interruption",
                       "evidence": {"synthetic": True}}
            self.assertEqual(recovered.recover_once(failure), 1)
            self.assertTrue(torch.equal(recovered.results[0], first))
            self.assertEqual(recovered.run_next(lambda: None), 2)
            receipt = recovered.complete()
            self.assertEqual(receipt["observations"], 64)
            self.assertTrue(receipt["recovered"])
            self.assertTrue(torch.equal(load_scaler(root, data, "a" * 64)[:32], first))
            rows = [json.loads(line) for line in (root / "physical.jsonl").read_text().splitlines()]
            self.assertGreater(sum(row.get("anchor") == 1 and "status" not in row
                                   for row in rows), 32)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                ScalerJournal(root, model, data, "a" * 64).recover_once(failure)

    def test_numerical_failure_cannot_recover(self):
        data, model = self.data(), TinySegmenter()
        bank = anchors("fit")[:1]
        with tempfile.TemporaryDirectory() as directory, \
                patch("dpa_ctta.r8_ba.scaler_journal.anchors", return_value=bank):
            root = Path(directory) / "scaler"
            job = ScalerJournal(root, model, data, "b" * 64)
            job.create()
            with self.assertRaises(ValueError):
                job.run_next(lambda: (_ for _ in ()).throw(ValueError("nonfinite")))
            with self.assertRaisesRegex(ValueError, "cannot recover"):
                ScalerJournal(root, model, data, "b" * 64).recover_once(
                    {"class": "INFRASTRUCTURE", "reason": "false", "evidence": {"x": 1}})

    def test_complete_uncheckpointed_anchor_replays_once(self):
        data, model = self.data(), TinySegmenter()
        bank = anchors("fit")[:1]
        with tempfile.TemporaryDirectory() as directory, \
                patch("dpa_ctta.r8_ba.scaler_journal.anchors", return_value=bank):
            root = Path(directory) / "scaler"
            job = ScalerJournal(root, model, data, "c" * 64)
            job.create()
            with patch.object(job, "_snapshot", side_effect=OSError("synthetic checkpoint failure")):
                with self.assertRaises(OSError):
                    job.run_next(lambda: None)
            recovered = ScalerJournal(root, model, data, "c" * 64)
            self.assertEqual(recovered.recover_once(
                {"class": "INFRASTRUCTURE", "reason": "checkpoint failure",
                 "evidence": {"synthetic": True}}), 0)
            self.assertEqual(recovered.run_next(lambda: None), 1)
            self.assertEqual(recovered.complete()["observations"], 32)
            self.assertEqual(COUNTS["backbone_forwards"], 64)


if __name__ == "__main__":
    unittest.main()
