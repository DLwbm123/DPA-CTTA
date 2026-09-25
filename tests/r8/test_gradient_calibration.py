import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from dpa_ctta.r8_ba.gradient_calibration import episode, select


class TestGradientCalibration(unittest.TestCase):
    def test_disjoint_query_uses_current_code_without_gradient(self):
        queries = []
        class Segmenter(torch.nn.Module):
            def forward(self, image, v):
                queries.append((torch.is_grad_enabled(), v.clone()))
                return torch.ones(1, 2, 4, 4) * v[0]
        segmenter = Segmenter()
        host = SimpleNamespace(segmenter=segmenter, visits=0, arm="COLD_G3", lr=0.001)
        def step(image):
            segmenter(image, torch.ones(1024) * 0.75)
            host.visits += 1
        host.step = step
        groups = {str(i) for i in range(23)}
        data = SimpleNamespace(folds={"cal": groups},
                               get=lambda name, fold: SimpleNamespace(image=torch.zeros(1, 3, 4, 4),
                                                                       label=torch.ones(1, 2, 4, 4)))
        oracles = SimpleNamespace(fold="cal", support_pairs=(("0", "1"),) * 128, validate=lambda _: None)
        with patch("dpa_ctta.r8_ba.gradient_calibration.simulate", side_effect=lambda image, *_: image):
            row = episode(host, data, oracles, 0, 20260924)
        self.assertEqual(host.visits, 4)
        self.assertTrue(all(not queries[i][0] for i in (1, 3, 5, 7)))
        self.assertTrue(all(torch.equal(v, torch.ones(1024) * 0.75) for _, v in queries))
        self.assertTrue(all(a != b and b not in ("0", "1") for a, b in row["support_query_groups"]))
        rows = [dict(arm=arm, lr=lr, source_seed=seed, episode=index, soft_Dice=0.5)
                for arm in ("B_G1", "B_G3", "COLD_G3") for lr in (0.001, 0.01, 0.1)
                for seed in (20260924, 20260925) for index in range(64)]
        self.assertEqual(set(select(rows)["selected_lr"].values()), {0.001})
        with self.assertRaisesRegex(ValueError, "1152"):
            select(rows[:-1])


if __name__ == "__main__":
    unittest.main()
