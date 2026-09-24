import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import patch

import torch

from dpa_ctta.r8_ba.journal import TargetJournal
from dpa_ctta.r8_ba.online import run
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


class TestOnline(unittest.TestCase):
    def test_failed_model_call_remains_in_physical_log(self):
        with tempfile.TemporaryDirectory() as directory:
            host = FakeFailHost()
            journal = TargetJournal(Path(directory) / "job", host, "failed")
            journal.create()
            rows = [dict(image_path="image", image_sha256="0" * 64, image_size=[512, 512])]
            with patch("dpa_ctta.r8_ba.online.TargetReader", FakeReader):
                with self.assertRaisesRegex(ValueError, "numerical failure"):
                    run(host, rows, directory, journal, 1024, lambda: None)
            record = json.loads(journal.physical.read_text())
            self.assertEqual(record["counts"]["backbone_forwards"], 1)
            self.assertEqual(record["status"], "FAILED_CALL")

    def test_online_keeps_masks_out_and_commits_exact_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            host = FakeHost()
            journal = TargetJournal(Path(directory) / "job", host, "synthetic")
            journal.create()
            rows = [dict(image_path=f"image{i}", image_sha256="0" * 64,
                         image_size=[512, 512], mask_path=f"mask{i}", mask_sha256="1" * 64)
                    for i in range(2)]
            checks = []
            FakeReader.seen = []
            with patch("dpa_ctta.r8_ba.online.TargetReader", FakeReader):
                result = run(host, rows, directory, journal, 1024, lambda: checks.append(True))
            self.assertEqual(result["visits"], 2)
            self.assertEqual(result["prediction_bytes"], 2 * 65536)
            self.assertEqual(journal.verified_complete(2), result)
            self.assertEqual(len(checks), 4)
            self.assertEqual(FakeReader.seen, [{"image_path", "image_sha256", "image_size"}] * 2)


if __name__ == "__main__":
    unittest.main()
