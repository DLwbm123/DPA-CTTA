import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dpa_ctta.r8_ba.scoring import score


class TestScoringRecovery(unittest.TestCase):
    def test_partial_scoring_replays_only_uncommitted_rows_and_keeps_cost(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [dict(group_id=str(i), domain="synthetic", subset="remaining_dev") for i in range(53)]
            (root / "predictions.bits").write_bytes(bytes(53 * 65536))
            reader = SimpleNamespace(read=lambda _: torch.zeros(1, 2, 512, 512), after_check=lambda: None)
            checks = [0]
            def guard():
                checks[0] += 1
                if checks[0] == 105:
                    raise OSError("synthetic interruption after two uncommitted rows")
            with patch("dpa_ctta.r8_ba.scoring.verify_online_complete", return_value={"prediction_sha256": "1" * 64}), \
                    patch("dpa_ctta.r8_ba.scoring.TargetReader", return_value=reader), \
                    patch("dpa_ctta.r8_ba.scoring.evaluate", return_value=[dict(dice=1.0)]):
                args = (rows, root, root, "job", "0" * 64, "synthetic", 0, 1024)
                with self.assertRaises(OSError):
                    score(*args, guard)
                self.assertEqual(len((root / "scalars.private.jsonl").read_text().splitlines()), 52)
                receipt = score(*args, lambda: None, resume_failure=dict(
                    **{"class": "INFRASTRUCTURE"}, reason="synthetic", evidence={"exit": 1}))
                self.assertEqual(receipt["visits"], 53)
                scalars = [json.loads(line) for line in (root / "scalars.private.jsonl").read_text().splitlines()]
                self.assertEqual([row["visit"] for row in scalars], list(range(1, 54)))
                self.assertEqual(len((root / "score_physical.jsonl").read_text().splitlines()), 55)


if __name__ == "__main__":
    unittest.main()
