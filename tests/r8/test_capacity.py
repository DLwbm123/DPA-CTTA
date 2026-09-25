import unittest

import torch

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_shared.source import Record, SourceData
from dpa_ctta.r8_ba.capacity import ANCHORS, capacity_one
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.protocol import CAPACITY_OPTIMIZER, PROTOCOL_SHA256, SELECTION_METRIC
from test_oracles import TinySegmenter


class TestCapacity(unittest.TestCase):
    def test_frozen_source_val_probe(self):
        self.assertEqual((CAPACITY_OPTIMIZER["lr"], SELECTION_METRIC), (0.03, "soft_Dice"))
        self.assertEqual(len(ANCHORS), 64)
        folds = dict(fit=[f"fit{i}" for i in range(32)], cal=[f"cal{i}" for i in range(4)],
                     val=[f"val{i}" for i in range(4)])
        records = [Record(name, fold, torch.rand(1, 3, 8, 8),
                          torch.randint(2, (1, 2, 8, 8)).float())
                   for fold, names in folds.items() for name in names]
        data = SourceData(records, folds)
        support, query = tuple(folds["val"][:2]), tuple(folds["val"][2:])
        oracles = Oracles("val", 0.1, torch.zeros(1024, 128),
                          (support,) * 128, (query,) * 128, ((),) * 128)
        segmenter = TinySegmenter()
        before = COUNTS.copy()
        result = capacity_one(segmenter, data, oracles,
                              torch.eye(1024, dtype=torch.float64)[:, :16], "A16", 1)
        self.assertEqual(result["protocol_sha256"], PROTOCOL_SHA256)
        self.assertEqual(result["support_groups"], list(support))
        self.assertEqual(result["query_groups"], list(query))
        self.assertEqual(set(result["query_soft_Dice"]),
                         {"zero", "ambient_oracle", "oracle_projection", "direct_latent"})
        self.assertEqual(segmenter.forwards, 2 * 128 + 2 * 4)
        self.assertEqual(COUNTS["capacity_backward_calls"] - before["capacity_backward_calls"], 128)
        self.assertEqual(COUNTS["capacity_Adam"] - before["capacity_Adam"], 128)
        with self.assertRaises(ValueError):
            capacity_one(segmenter, data, oracles,
                         torch.eye(1024, dtype=torch.float64)[:, :16], "A16", 33)


if __name__ == "__main__":
    unittest.main()
