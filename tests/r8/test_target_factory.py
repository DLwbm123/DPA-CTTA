import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from dpa_ctta.r8_ba.methods import CurrentMLP
from dpa_ctta.r8_ba.target_factory import load_deployed
from dpa_ctta.r8_ba.trainer import method_config


class TestTargetFactory(unittest.TestCase):
    def test_mlp_loads_selected_journal_envelope(self):
        candidate = dict(id="B_compact_global_aux1p0", route="B", rank=32,
                         film_amplitude=0.1, observer="global", aux_multiplier=1.0)
        method = CurrentMLP(torch.eye(1024, dtype=torch.float64)[:, :32], 0.1, "global")
        method.observer.fit_scaler(torch.randn(4, 134), "fit")
        snapshot = dict(steps=1000, binding="synthetic", source_seed=20260924,
                        method=method.state_dict(), method_config=method_config(method), method_digest=method.digest())
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp).resolve() / "selected.pt"
            torch.save(dict(schema="R8_SOURCE_JOURNAL_V1", snapshot=snapshot), path)
            row = dict(config=candidate, source_seed=20260924, mode=None, source_step=1000,
                       binding="synthetic", artifact=dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            deployed = load_deployed(row, candidate, 20260924, None, mlp=True)
            self.assertEqual(deployed.stage, "online")
            self.assertEqual(deployed.digest(), method.digest())
            with self.assertRaisesRegex(ValueError, "selected source point"):
                load_deployed(dict(row, source_step=4000), candidate, 20260924, None, mlp=True)


if __name__ == "__main__":
    unittest.main()
