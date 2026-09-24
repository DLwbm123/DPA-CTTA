import unittest
import tempfile
from pathlib import Path

import torch
from torch import nn

from dpa_ctta.r8_ba.context import SOURCE_KEYS, capture
from dpa_ctta.r8_ba.host import OnlineHost
from dpa_ctta.r8_ba.journal import TargetJournal
from dpa_ctta.r8_ba.methods import R8B, R8Segmenter


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = nn.Module()
        self.res.conv1 = nn.Conv2d(3, 64, 1)
        self.up1 = nn.Conv2d(64, 256, 1)
        self.up3 = nn.Conv2d(256, 256, 1)
        self.seg_head = nn.Conv2d(256, 2, 1)

    def forward(self, image):
        h = torch.nn.functional.adaptive_avg_pool2d(image, (8, 8))
        h = self.up3(self.up1(self.res.conv1(h)))
        return torch.nn.functional.interpolate(self.seg_head(h), (512, 512))


class TestHost(unittest.TestCase):
    def test_image_only_commit_and_roundtrip(self):
        segmenter = R8Segmenter(Small(), 0.1)
        method = R8B(torch.eye(1024, dtype=torch.float64)[:, :32], 0.1)
        method.observer.fit_scaler(torch.randn(4, 134), "fit")
        method.freeze()
        config = dict(id="B_compact_global_aux1p0", route="B", rank=32, film_amplitude=0.1,
                      observer="global", aux_multiplier=1.0)
        source = {key: "0" * 64 for key in SOURCE_KEYS}  # synthetic identity only
        context = capture(segmenter, method, config, source)
        eta = method.frozen_eta.clone()
        with torch.no_grad():
            method.frozen_eta.add_(0.1)
        with self.assertRaisesRegex(ValueError, "ISTA step"):
            capture(segmenter, method, config, source)
        with torch.no_grad():
            method.frozen_eta.copy_(eta)
        host = OnlineHost(segmenter, method, config, source, context)
        image = torch.rand(1, 3, 512, 512)
        with tempfile.TemporaryDirectory() as directory:
            journal = TargetJournal(Path(directory) / "job", host, "synthetic", "0" * 64,
                                    prediction_bytes=2)
            journal.create()
            first, trace = host.step(image)
            journal.append(b"\x00\x01", trace)
            self.assertEqual(journal.recover_once(), 0)
            replay_first, _ = host.step(image)
            self.assertTrue(torch.equal(first, replay_first))
        self.assertEqual(trace["visit"], 1)
        self.assertEqual(trace["counts"]["backbone_forwards"], 2)
        snapshot = host.snapshot()
        second, _ = host.step(image)
        host.restore(snapshot)
        replay, _ = host.step(image)
        self.assertTrue(torch.equal(second, replay))
        self.assertFalse(torch.equal(first, second))
        method.aux_multiplier = 0.1
        with self.assertRaisesRegex(ValueError, "scientific configuration"):
            host.step(image)
        segmenter.close()


if __name__ == "__main__":
    unittest.main()
