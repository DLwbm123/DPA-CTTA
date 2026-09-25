import tempfile
import unittest
import json
from pathlib import Path

from dpa_ctta.r8_ba.journal import SourceJournal, TargetJournal

FAILURE = {"class": "INFRASTRUCTURE", "reason": "synthetic interruption",
           "evidence": {"exit_code": 137}}


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
        self.method = self

    def digest(self):
        return "0" * 64

    def snapshot(self):
        return dict(steps=self.steps, binding=self.binding, source_seed=self.source_seed,
                    method_digest=self.digest(), method_config=("synthetic",))

    def restore(self, snapshot):
        self.steps = snapshot["steps"]


class TestJournal(unittest.TestCase):
    def test_source_completion_requires_all_points_and_physical_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            trainer = FakeTrainer()
            journal = SourceJournal(Path(directory) / "source", trainer)
            journal.create()
            for step in (1000, 4000, 8000, 12000, 16000):
                trainer.steps = step
                journal.checkpoint()
            with self.assertRaisesRegex(ValueError, "physical completion coverage"):
                journal.complete()
            journal.physical.write_text("".join(json.dumps({"step": step, "counts": {"forward": 1}}) + "\n"
                                                for step in range(1, 16001)))
            receipt = journal.complete()
            self.assertEqual((receipt["steps"], len(receipt["selected_sha256"])), (16000, 5))
            with self.assertRaisesRegex(ValueError, "already used"):
                journal.recover_once(FAILURE)

    def test_selected_source_snapshot_survives_slot_rotation_and_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            trainer = FakeTrainer()
            journal = SourceJournal(root, trainer)
            journal.create()
            journal.physical.write_text("".join(json.dumps({"step": i, "counts": {"forward": 1}}) + "\n"
                                                for i in range(1, 1001)))
            trainer.steps = 1000
            journal.checkpoint()
            (root / "selected.1000.json").unlink()
            recovered = SourceJournal(root, FakeTrainer())
            self.assertEqual(recovered.recover_once(FAILURE), 1000)
            recovered.trainer.steps = 1250
            recovered.checkpoint()
            recovered.trainer.steps = 1500
            recovered.checkpoint()
            self.assertEqual(recovered.selected(1000)["steps"], 1000)
            with self.assertRaisesRegex(ValueError, "already used"):
                recovered.recover_once(FAILURE)

    def test_numerical_failed_call_is_not_infrastructure_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "source"
            SourceJournal(root, FakeTrainer()).create()
            (root / "physical.jsonl").write_text(json.dumps(dict(
                step=1, status="FAILED_CALL", counts={"forward": 1},
                error_type="ValueError")) + "\n")
            with self.assertRaisesRegex(ValueError, "cannot recover"):
                SourceJournal(root, FakeTrainer()).recover_once(FAILURE)

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
            with self.assertRaisesRegex(ValueError, "only evidenced infrastructure"):
                resumed.recover_once(dict(FAILURE, **{"class": "NUMERICAL"}))
            self.assertEqual(resumed.recover_once(FAILURE), 50)
            self.assertEqual(resumed.predictions.stat().st_size, 100)
            self.assertEqual(resumed.physical.stat().st_size, spent)
            for visit in (51, 52):
                resumed.host.visits = visit
                resumed.append(bytes([visit, visit]), {"visit": visit, "counts": {"forward": 1}})
            self.assertEqual(resumed.predictions.read_bytes(), original)
            with self.assertRaisesRegex(ValueError, "already used"):
                resumed.recover_once(FAILURE)


if __name__ == "__main__":
    unittest.main()
