import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from dpa_ctta.r8_ba.journal import TargetJournal
from dpa_ctta.r8_ba.online import run
from dpa_ctta.r8_ba.scoring import score
from dpa_ctta.r8_ba.streams import rows_sha
from dpa_ctta.r7_shared.numerics import COUNTS


class FakeHost:
    def __init__(self):
        self.context = {"sha256": "0" * 64}
        self.visits = 0

    def check_frozen(self, boundary=False):
        pass

    def snapshot(self):
        return {"visits": self.visits}

    def step(self, image):
        self.visits += 1
        return torch.zeros(1, 2, 512, 512), {"visit": self.visits, "counts": {"forward": 1}}


class FakeReader:
    seen = []

    def __init__(self, root, limit, kind):
        if kind != "image":
            raise AssertionError("mask capability entered online phase")

    def read(self, row):
        self.seen.append(set(row))
        return torch.ones(1, 3, 512, 512)

    def after_check(self):
        pass


class FakeFailHost(FakeHost):
    def step(self, image):
        COUNTS["backbone_forwards"] += 1
        raise ValueError("synthetic numerical failure")


class FakeMaskReader:
    reads = 0

    def __init__(self, root, limit, kind):
        if kind != "mask":
            raise AssertionError("wrong scoring capability")

    def read(self, row):
        type(self).reads += 1
        return torch.zeros(1, 2, 512, 512)

    def after_check(self):
        pass


class TestOnline(unittest.TestCase):
    def test_failed_model_call_remains_in_physical_log(self):
        with tempfile.TemporaryDirectory() as directory:
            host = FakeFailHost()
            rows = [dict(group_id="group", image_path="image", image_sha256="0" * 64,
                         image_size=[512, 512])]
            journal = TargetJournal(Path(directory) / "job", host, "failed", rows_sha(rows))
            journal.create()
            with patch("dpa_ctta.r8_ba.online.TargetReader", FakeReader):
                with self.assertRaisesRegex(ValueError, "numerical failure"):
                    run(host, rows, directory, journal, 1024, lambda: None)
            record = json.loads(journal.physical.read_text())
            self.assertEqual(record["counts"]["backbone_forwards"], 1)
            self.assertEqual(record["status"], "FAILED_CALL")

    def test_online_keeps_masks_out_and_commits_exact_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            host = FakeHost()
            rows = [dict(group_id=f"group{i}", image_path=f"image{i}", image_sha256="0" * 64,
                         image_size=[512, 512], mask_path=f"mask{i}", mask_sha256="1" * 64)
                    for i in range(2)]
            journal = TargetJournal(Path(directory) / "job", host, "synthetic", rows_sha(rows))
            journal.create()
            checks = []
            FakeReader.seen = []
            with patch("dpa_ctta.r8_ba.online.TargetReader", FakeReader):
                result = run(host, rows, directory, journal, 1024, lambda: checks.append(True))
            self.assertEqual(result["visits"], 2)
            self.assertEqual(result["prediction_bytes"], 2 * 65536)
            self.assertEqual(journal.verified_complete(2), result)
            self.assertEqual(len(checks), 4)
            self.assertEqual(FakeReader.seen, [{"image_path", "image_sha256", "image_size"}] * 2)

    def test_scoring_opens_masks_only_after_verified_online_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            host = FakeHost()
            rows = [dict(image_path="image", image_sha256="0" * 64, image_size=[512, 512],
                         mask_path="mask", mask_sha256="1" * 64, group_id="group",
                         domain="domain", subset="remaining_dev")]
            journal = TargetJournal(Path(directory) / "job", host, "synthetic", rows_sha(rows))
            journal.create()
            with patch("dpa_ctta.r8_ba.online.TargetReader", FakeReader):
                run(host, rows, directory, journal, 1024, lambda: None)
            with patch("dpa_ctta.r8_ba.scoring.TargetReader", FakeMaskReader), patch(
                    "dpa_ctta.r8_ba.scoring.evaluate", return_value=[{"dice": 1.0}]):
                receipt = score(rows, directory, journal.root, "synthetic", "0" * 64,
                                "B", 0, 1024, lambda: None)
            self.assertEqual(receipt["visits"], 1)
            self.assertEqual(json.loads((journal.root / "scalars.private.jsonl").read_text())["metrics"],
                             [{"dice": 1.0}])
            self.assertEqual(FakeMaskReader.reads, 1)
            changed = [dict(rows[0], mask_sha256="2" * 64)]
            with patch("dpa_ctta.r8_ba.scoring.TargetReader", side_effect=AssertionError("mask opened")):
                with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                    score(changed, directory, journal.root, "synthetic", "0" * 64,
                          "B", 0, 1024, lambda: None)
            (journal.root / "predictions.bits").write_bytes(b"x" * 65536)
            with patch("dpa_ctta.r8_ba.scoring.TargetReader", side_effect=AssertionError("mask opened")):
                with self.assertRaisesRegex(ValueError, "receipt mismatch"):
                    score(rows, directory, journal.root, "synthetic", "0" * 64,
                          "B", 0, 1024, lambda: None)


if __name__ == "__main__":
    unittest.main()
