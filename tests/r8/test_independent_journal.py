import json
import tempfile
import unittest
from pathlib import Path

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.independent_journal import IndependentJournal


class TestIndependentJournal(unittest.TestCase):
    def test_replay_independent_probe_keeps_failed_attempt(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "probes"
            items = [dict(anchor=i) for i in range(3)]
            journal = IndependentJournal(root, {"binding": "synthetic"}, items)
            journal.create()
            fail = [True]
            def run(item):
                COUNTS["capacity_backward_calls"] += 128
                if item["anchor"] == 1 and fail[0]:
                    raise OSError("synthetic interruption")
                return {"score": item["anchor"]}
            with self.assertRaises(OSError):
                journal.run(run, lambda: None)
            recovered = IndependentJournal(root, {"binding": "synthetic"}, items)
            recovered.recover_once({"class": "INFRASTRUCTURE", "reason": "synthetic", "evidence": {"exit": 1}})
            fail[0] = False
            receipt = recovered.run(run, lambda: None)
            self.assertEqual(receipt["completed"], 3)
            work = [json.loads(row) for row in journal.physical.read_text().splitlines()]
            self.assertEqual(sum(row["counts"]["capacity_backward_calls"] for row in work), 512)
            self.assertEqual([row["item"] for row in json.loads((root / "results.json").read_text())], items)


if __name__ == "__main__":
    unittest.main()
