import unittest

import torch

from dpa_ctta.r7_shared.source import Record, SourceData
from dpa_ctta.r8_ba.methods import R8B
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.trainer import SourceTrainer, lr_at


class TinySegmenter:
    amplitude = 0.1

    def __call__(self, image, v=None, observe=False):
        if v is None:
            v = torch.zeros(1024)
        logits = image[:, :2] + v[:2].view(1, 2, 1, 1)
        if observe:
            mean = image.mean()
            return logits, mean.expand(134).clone(), mean.expand(64, 64).clone()
        return logits


class TestTrainer(unittest.TestCase):
    def test_chunk_snapshot_roundtrip(self):
        torch.manual_seed(7)
        folds = {k: [f"{k}{i}" for i in range(n)] for k, n in (("fit", 32), ("cal", 4), ("val", 4))}
        data = SourceData([Record(g, k, torch.rand(1, 3, 8, 8), torch.randint(2, (1, 2, 8, 8)).float())
                           for k, groups in folds.items() for g in groups], folds)
        support, query = tuple(folds["fit"][:2]), tuple(folds["fit"][2:4])
        oracles = Oracles("fit", 0.1, torch.zeros(1024, 512), (support,) * 512,
                          (query,) * 512, ((),) * 512)
        method = R8B(torch.eye(1024, dtype=torch.float64)[:, :32], 0.1)
        method.observer.fit_scaler(torch.randn(4, 134), "fit")
        trainer = SourceTrainer(TinySegmenter(), method, data, oracles, 20260924, "synthetic")
        self.assertEqual(trainer.fit_step()["query_visits"], 8)
        snapshot = trainer.snapshot()
        expected = trainer.fit_step()
        expected_weights = {k: v.clone() for k, v in method.state_dict().items()}
        trainer.restore(snapshot)
        self.assertEqual(expected, trainer.fit_step())
        self.assertTrue(all(torch.equal(v, expected_weights[k]) for k, v in method.state_dict().items()))
        self.assertEqual(lr_at(16000), 3e-5)


if __name__ == "__main__":
    unittest.main()
