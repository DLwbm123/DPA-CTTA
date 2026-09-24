import tempfile
import unittest
from pathlib import Path

from dpa_ctta.r8_ba.journal import SourceJournal, TargetJournal


class FakeHost:
    def __init__(self):
        self.context = {"sha256": "0" * 64}
        self.visits = 0

    def check_frozen(self, boundary=False):
        pass

    def snapshot(self):
        return {"visits": self.visits}

    def restore(self, state):
        self.visits = state["visits"]


class FakeTrainer:
    binding = "synthetic"
    source_seed = 20260924

    def __init__(self):
        self.steps = 0

    def snapshot(self):
        return dict(steps=self.steps, binding=self.binding, source_seed=self.source_seed)

    def restore(self, snapshot):
        self.steps = snapshot["steps"]


class TestJournal(unittest.TestCase):
    def test_selected_source_snapshot_survives_slot_rotation_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            trainer = FakeTrainer()
            journal = SourceJournal(root, trainer)
            journal.create()
            trainer.steps = 1000
            journal.checkpoint()
            (root / "selected.1000.json").unlink()
            recovered = SourceJournal(root, FakeTrainer())
            self.assertEqual(recovered.recover_once(), 1000)
            recovered.trainer.steps = 1250
            recovered.checkpoint()
            recovered.trainer.steps = 1500
            recovered.checkpoint()
            self.assertEqual(recovered.selected(1000)["steps"], 1000)
            with self.assertRaisesRegex(ValueError, "already used"):
                recovered.recover_once()

    def test_checkpoint_prefix_recovery_and_one_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "one-job"
            host = FakeHost()
            journal = TargetJournal(root, host, "job-1", "0" * 64, prediction_bytes=2)
            journal.create()
            for visit in range(1, 53):
                host.visits = visit
                journal.append(bytes([visit, visit]), {"visit": visit, "counts": {"forward": 1}})
            original = journal.predictions.read_bytes()
            spent = journal.physical.stat().st_size
            resumed = TargetJournal(root, FakeHost(), "job-1", "0" * 64, prediction_bytes=2)
            self.assertEqual(resumed.recover_once(), 50)
            self.assertEqual(resumed.predictions.stat().st_size, 100)
            self.assertEqual(resumed.physical.stat().st_size, spent)
            for visit in (51, 52):
                resumed.host.visits = visit
                resumed.append(bytes([visit, visit]), {"visit": visit, "counts": {"forward": 1}})
            self.assertEqual(resumed.predictions.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "already used"):
                resumed.recover_once()


if __name__ == "__main__":
    unittest.main()
