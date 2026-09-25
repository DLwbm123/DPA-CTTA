import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from torch import nn

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.validation_journal import ValidationJournal, load_validation


class TestValidationJournal(unittest.TestCase):
    def test_episode_boundary_recovery_keeps_physical_tail(self):
        segmenter = nn.Identity()
        fail = [True]

        def episode(_segmenter, _method, _data, _oracles, index, on_visit=None):
            for visit in range(1, 33):
                COUNTS["backbone_forwards"] += 2
                on_visit(visit)
                if index == 1 and visit == 5 and fail[0]:
                    raise TimeoutError("synthetic interruption")
            return dict(episode=index, curriculum="abrupt_16_16",
                        soft_Dice=0.5, proxy_MSE=0.1)

        with tempfile.TemporaryDirectory() as directory, \
                patch("dpa_ctta.r8_ba.validation_journal.EPISODES", 2), \
                patch("dpa_ctta.r8_ba.validation_journal.validate_episode", episode):
            COUNTS.clear()
            root = Path(directory) / "source_job"
            root.mkdir()
            job = ValidationJournal(root, 1000, "a" * 64, "binding")
            job.create()
            self.assertEqual(job.run_next(segmenter, None, None, None, lambda: None), 1)
            with self.assertRaises(TimeoutError):
                job.run_next(segmenter, None, None, None, lambda: None)
            recovered = ValidationJournal(root, 1000, "a" * 64, "binding")
            failure = {"class": "INFRASTRUCTURE", "reason": "interruption",
                       "evidence": {"synthetic": True}}
            self.assertEqual(recovered.recover_once(failure), 1)
            fail[0] = False
            self.assertEqual(recovered.run_next(segmenter, None, None, None, lambda: None), 2)
            receipt = recovered.complete()
            self.assertEqual(receipt["episodes"], 2)
            self.assertEqual(receipt["physical_counts"]["backbone_forwards"], 138)
            self.assertEqual(len(load_validation(root, 1000, "a" * 64, "binding")), 2)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                ValidationJournal(root, 1000, "a" * 64, "binding").recover_once(failure)


if __name__ == "__main__":
    unittest.main()
