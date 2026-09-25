import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.ledger import Ledger
from dpa_ctta.r8_ba.resources import CAPS
from dpa_ctta.r8_ba.worker_budget import WorkerBudget


class TestWorkerBudget(unittest.TestCase):
    def test_physical_cost_survives_logical_counter_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            ledger = Ledger(root / "ledger", dict(code_sha="1" * 40, protocol_sha256="2" * 64,
                                                  graph_sha256="3" * 64))
            ledger.create(dict.fromkeys(CAPS, 0), "preflight")
            budget = dict(gpu_seconds=100, model_forwards=10, backward_calls=10,
                          optimizer_steps=10, vjp_calls=10, disk_bytes=1024**2)
            ledger.reserve("a1", 5, budget)
            model = torch.nn.Linear(1, 1)
            (root / "job").mkdir()
            (root / "job" / "prior-stage-output").write_bytes(bytes(4096))
            optimizer = torch.optim.Adam(model.parameters())
            backward, grad = torch.autograd.backward, torch.autograd.grad
            with patch("os.getpid", return_value=123), patch("os.getpgrp", return_value=123), \
                    patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "5"}):
                with WorkerBudget(ledger, "a1", root / "job", budget, 90) as guard:
                    guard.attach_model(model)
                    (root / "job" / "new-output").write_bytes(bytes(64))
                    for _ in range(2):
                        optimizer.zero_grad()
                        model(torch.ones(1, 1)).sum().backward()
                        optimizer.step()
                        COUNTS.clear()  # A scientific snapshot restore cannot refund physical work.
                    torch.autograd.grad(model(torch.ones(1, 1)).sum(), tuple(model.parameters()))
            actual = ledger.snapshot()["attempts"]["a1"]["actual"]
            self.assertEqual(actual["disk_bytes"], 64)
            self.assertEqual([actual[k] for k in ("model_forwards", "backward_calls", "optimizer_steps", "vjp_calls")],
                             [3, 2, 2, 1])
            self.assertIs(torch.autograd.backward, backward)
            self.assertIs(torch.autograd.grad, grad)


if __name__ == "__main__":
    unittest.main()
