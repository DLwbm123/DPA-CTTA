import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dpa_ctta.r8_ba.ledger import Ledger
from dpa_ctta.r8_ba.resources import CAPS


IDENTITY = dict(code_sha="1" * 40, protocol_sha256="2" * 64, graph_sha256="3" * 64)


def cost(**values):
    return dict.fromkeys(CAPS, 0) | values


class TestLedger(unittest.TestCase):
    def test_concurrent_admission_crash_retention_and_global_stop(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ledger"
            ledger = Ledger(root, IDENTITY)
            ledger.create(cost(), "profile-receipts")
            def reserve(i):
                Ledger(root, IDENTITY).reserve(str(i), 5 + i % 3,
                                               cost(gpu_seconds=10, backward_calls=100))
            with ThreadPoolExecutor(max_workers=3) as pool:
                list(pool.map(reserve, range(12)))
            ledger.settle("0", cost(gpu_seconds=3, backward_calls=50), "physical/0.jsonl")
            total = ledger.snapshot()["committed_and_reserved"]
            self.assertEqual(total["backward_calls"], 1150)
            self.assertEqual(total["gpu_seconds"], 113)
            with self.assertRaisesRegex(ValueError, "reused"):
                reserve(1)  # A crash cannot release or reuse attempt 1's reservation.
            with self.assertRaisesRegex(RuntimeError, "GLOBAL STOP"):
                ledger.reserve("cap", 7, cost(gpu_seconds=1, backward_calls=CAPS["backward_calls"]))
            with self.assertRaisesRegex(RuntimeError, "GLOBAL STOP"):
                ledger.guard("1", cost())  # Other active workers see the persistent stop.

    def test_identity_and_overrun_are_persistent_stops(self):
        for mismatch in (True, False):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp) / "ledger"
                ledger = Ledger(root, IDENTITY)
                ledger.create(cost(), "profile-receipts")
                ledger.reserve("a", 5, cost(gpu_seconds=10))
                with self.assertRaisesRegex(RuntimeError, "GLOBAL STOP"):
                    if mismatch:
                        Ledger(root, dict(IDENTITY, code_sha="4" * 40)).snapshot()
                    else:
                        ledger.settle("a", cost(gpu_seconds=11), "physical/a.jsonl")
                self.assertIsNotNone(json.loads((root / "state.json").read_text())["stop"])
                with self.assertRaisesRegex(RuntimeError, "GLOBAL STOP"):
                    ledger.snapshot()


if __name__ == "__main__":
    unittest.main()
