import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from dpa_ctta.r7_shared.context import tensor_digest
from dpa_ctta.b1_host import GRATA_COMMIT
from dpa_ctta.r8_ba.context import SOURCE_KEYS, capture
from dpa_ctta.r8_ba.gradient import GradientHost
from dpa_ctta.r8_ba.journal import TargetJournal
from dpa_ctta.r8_ba.methods import R8B, R8Segmenter
from dpa_ctta.r8_ba.preparation import gradient_scale
from test_host import Small


class FakeC:
    class Rotate_and_Flip:
        def __call__(self, image, factor):
            return image

        def inverse(self, logits, factor):
            return logits

    @staticmethod
    def augmentation_strong_style(data):
        return data["data"]

    @staticmethod
    def normalize_image_to_0_1(image):
        return image


class TestGradient(unittest.TestCase):
    def test_source_fit_rms_scale_without_centering(self):
        values = torch.zeros(1024, 512, dtype=torch.float64)
        values[0, :256] = 2
        scale = gradient_scale(SimpleNamespace(fold="fit", values=values),
                               torch.eye(1024, dtype=torch.float64)[:, :32])
        self.assertAlmostEqual(scale[0].item(), 2 ** 0.5)
        self.assertEqual(scale[1].item(), 1e-3)

    def test_three_arms_count_real_backward_and_commit(self):
        for arm, expected in (("B_G1", 8), ("B_G3", 10), ("COLD_G3", 10)):
            with self.subTest(arm=arm):
                segmenter = R8Segmenter(Small(), 0.1)
                method = R8B(torch.eye(1024, dtype=torch.float64)[:, :32], 0.1)
                method.observer.fit_scaler(torch.randn(4, 134), "fit")
                method.freeze()
                scale = torch.ones(32, dtype=torch.float64)
                config = dict(id="synthetic", rank=32, film_amplitude=0.1, observer="global",
                              aux_multiplier=1.0, gradient_arm=arm, gradient_lr=0.001,
                              scale_sha256=tensor_digest([("scale", scale)]),
                              grata_commit=GRATA_COMMIT)
                source = {key: "0" * 64 for key in SOURCE_KEYS}
                context = capture(segmenter, method, config, source)
                host = GradientHost(segmenter, method, config, source, context,
                                    arm, scale, 0.001, api=FakeC())
                image = torch.rand(1, 3, 512, 512)
                if arm == "B_G1":
                    with tempfile.TemporaryDirectory() as directory:
                        journal = TargetJournal(Path(directory) / "job", host, "synthetic", "0" * 64,
                                                prediction_bytes=2)
                        journal.create()
                        prediction, trace = host.step(image)
                        journal.append(b"\x00\x01", trace)
                        self.assertEqual(journal.recover_once({"class": "INFRASTRUCTURE",
                                                               "reason": "synthetic interruption",
                                                               "evidence": {"exit_code": 137}}), 0)
                        replay, _ = host.step(image)
                        self.assertTrue(torch.equal(prediction, replay))
                else:
                    prediction, trace = host.step(image)
                self.assertEqual(prediction.shape, (1, 2, 512, 512))
                self.assertEqual(trace["counts"]["backbone_forwards"], expected)
                self.assertEqual(trace["counts"]["target_backward_calls"], expected - 7)
                self.assertEqual(trace["counts"]["target_Adam"], expected - 7)
                self.assertEqual(host.state["counter"], 1)
                if arm == "COLD_G3":
                    self.assertTrue(torch.equal(host.state["z"], torch.zeros_like(host.state["z"])))
                segmenter.close()


if __name__ == "__main__":
    unittest.main()
