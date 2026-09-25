import io
import unittest
from types import SimpleNamespace

import torch

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r8_ba.native_host import NativeHost
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256


class TestNativeHost(unittest.TestCase):
    def test_source_only_snapshot_identity_and_frozen_model(self):
        identity = dict(code_sha="0" * 40, protocol_sha256=PROTOCOL_SHA256,
                        checkpoint_sha256="1" * 64, registration_sha256="2" * 64, seed=None)
        def build():
            model = torch.nn.Conv2d(3, 2, 1).eval()
            with torch.no_grad():
                model.weight.fill_(0.1)
                model.bias.zero_()
            return NativeHost(SimpleNamespace(model=model, step=model), "N_SOURCE_EVAL", identity)
        COUNTS.clear()
        host = build()
        image = torch.zeros(1, 3, 512, 512)
        _, trace = host.step(image)
        self.assertEqual(trace["counts"], {"backbone_forwards": 1})
        buffer = io.BytesIO()
        torch.save(host.snapshot(), buffer)
        restored = build()
        restored.restore(torch.load(io.BytesIO(buffer.getvalue()), weights_only=True))
        actual, _ = restored.step(image)
        self.assertEqual(restored.visits, 2)
        torch.testing.assert_close(actual, host.step(image)[0], rtol=0, atol=0)
        with torch.no_grad():
            restored.native.model.weight.add_(1)
        with self.assertRaisesRegex(ValueError, "frozen tensor changed"):
            restored.step(image)


if __name__ == "__main__":
    unittest.main()
