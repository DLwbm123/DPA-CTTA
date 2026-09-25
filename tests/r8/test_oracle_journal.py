import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn

from dpa_ctta.r7_shared.source import Record, SourceData
from dpa_ctta.r8_ba.oracle_journal import OracleJournal, load_oracles, run_oracles


class TinySegmenter(nn.Module):
    amplitude = 0.1

    def forward(self, image, v):
        return image[:, :2] + v[:2].view(1, 2, 1, 1)


class TestOracleJournal(unittest.TestCase):
    @staticmethod
    def data():
        folds = dict(fit=[f"fit{i}" for i in range(32)],
                     cal=[f"cal{i}" for i in range(4)],
                     val=[f"val{i}" for i in range(4)])
        records = [Record(group, fold, torch.rand(1, 3, 8, 8),
                          torch.randint(2, (1, 2, 8, 8)).float())
                   for fold, groups in folds.items() for group in groups]
        return SourceData(records, folds)

    def test_one_verified_recovery_preserves_physical_tail(self):
        data, model = self.data(), TinySegmenter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "oracle"
            job = OracleJournal(root, model, data, "a" * 64)
            job.create()
            self.assertEqual(job.run_next(lambda: None), 1)
            original = job.results[0][0].clone()

            calls = 0

            def interrupted():
                nonlocal calls
                calls += 1
                if calls == 8:
                    raise TimeoutError("synthetic infrastructure interruption")

            with self.assertRaises(TimeoutError):
                job.run_next(interrupted)
            with self.assertRaisesRegex(ValueError, "next anchor"):
                job.run_next(lambda: None)
            recovered = OracleJournal(root, model, data, "a" * 64)
            failure = {"class": "INFRASTRUCTURE", "reason": "test interruption",
                       "evidence": {"synthetic": True}}
            self.assertEqual(recovered.recover_once(failure), 1)
            self.assertTrue(torch.equal(recovered.results[0][0], original))
            self.assertEqual(recovered.run_next(lambda: None), 2)
            rows = [json.loads(line) for line in (root / "physical.jsonl").read_text().splitlines()]
            self.assertEqual(sum(r.get("ordinal") == 0 and "step" in r and
                                 r.get("status") is None for r in rows), 256)
            self.assertGreater(sum(r.get("ordinal") == 1 and r.get("status") is None
                                   for r in rows), 256)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                OracleJournal(root, model, data, "a" * 64).recover_once(failure)

    def test_sealed_source_only_payload(self):
        data, model = self.data(), TinySegmenter()
        sizes = dict(fit=1, cal=1, val=1)
        order = tuple((fold, 0) for fold in sizes)
        with tempfile.TemporaryDirectory() as directory, \
                patch("dpa_ctta.r8_ba.oracle_journal.SIZES", sizes), \
                patch("dpa_ctta.r8_ba.oracle_journal.ORDER", order), \
                patch("dpa_ctta.r8_ba.oracles.SIZES", sizes):
            root = Path(directory) / "oracle"
            receipt = run_oracles(root, model, data, "b" * 64, lambda: None)
            self.assertEqual(receipt["anchors"], 3)
            payload = load_oracles(root, data, "b" * 64, 0.1)
            self.assertEqual(set(payload), set(sizes))
            self.assertEqual(payload["fit"].values.shape, (1024, 1))

    def test_numerical_failure_cannot_be_reclassified_as_infrastructure(self):
        data, model = self.data(), TinySegmenter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "oracle"
            job = OracleJournal(root, model, data, "c" * 64)
            job.create()
            with self.assertRaises(ValueError):
                job.run_next(lambda: (_ for _ in ()).throw(ValueError("nonfinite")))
            failure = {"class": "INFRASTRUCTURE", "reason": "incorrect claim",
                       "evidence": {"synthetic": True}}
            with self.assertRaisesRegex(ValueError, "cannot recover"):
                OracleJournal(root, model, data, "c" * 64).recover_once(failure)


if __name__ == "__main__":
    unittest.main()
