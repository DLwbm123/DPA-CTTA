import tempfile
import unittest
from pathlib import Path

from dpa_ctta.r8_ba.journal import TargetJournal


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


class TestJournal(unittest.TestCase):
    def test_checkpoint_prefix_recovery_and_one_use(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "one-job"
            host = FakeHost()
            journal = TargetJournal(root, host, "job-1", prediction_bytes=2)
            journal.create()
            for visit in range(1, 53):
                host.visits = visit
                journal.append(bytes([visit, visit]), {"visit": visit, "counts": {"forward": 1}})
            original = journal.predictions.read_bytes()
            spent = journal.physical.stat().st_size
            resumed = TargetJournal(root, FakeHost(), "job-1", prediction_bytes=2)
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
