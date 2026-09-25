import tempfile
import unittest
from pathlib import Path

import torch

from dpa_ctta.r8_ba.preparation import load_bases, save_bases


class TestBasesArtifact(unittest.TestCase):
    def test_nested_fit_bases_seal_and_identity(self):
        identity = {"oracle_receipt_sha256": "a" * 64, "amplitude": 0.1}
        a32 = torch.eye(1024, dtype=torch.float64)[:, :32]
        b64 = torch.eye(1024, dtype=torch.float64)[:, :64]
        audit = dict(covariance_groups=[f"fit{i}" for i in range(16)],
                     probe_groups=[f"fit{i}" for i in range(16, 32)],
                     covariance=torch.eye(1024, dtype=torch.float64))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "bases"
            save_bases(root, identity, {16: a32[:, :16], 32: a32},
                       {32: b64[:, :32], 64: b64}, audit,
                       {32: torch.ones(32, dtype=torch.float64),
                        64: torch.ones(64, dtype=torch.float64)},
                       {"source_VJP": 2048})
            self.assertTrue(torch.equal(load_bases(root, identity)["B_basis"][64], b64))
            with self.assertRaisesRegex(ValueError, "identity"):
                load_bases(root, dict(identity, amplitude=0.3))


if __name__ == "__main__":
    unittest.main()
