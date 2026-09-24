import unittest

import torch

from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.r7_shared.source import Record, SourceData
from dpa_ctta.r8_ba.oracles import oracle_one


class TinySegmenter:
    amplitude = 0.1

    def __init__(self):
        self.forwards = 0

    def __call__(self, image, v):
        self.forwards += 1
        return image[:, :2] + v[:2].view(1, 2, 1, 1)


class TestOracle(unittest.TestCase):
    def test_source_only_roles_and_physical_steps(self):
        folds = dict(fit=[f"fit{i}" for i in range(32)], cal=[f"cal{i}" for i in range(4)],
                     val=[f"val{i}" for i in range(4)])
        records = [Record(group, fold, torch.rand(1, 3, 8, 8), torch.randint(2, (1, 2, 8, 8)).float())
                   for fold, groups in folds.items() for group in groups]
        data, model = SourceData(records, folds), TinySegmenter()
        before = COUNTS.copy()
        value, support, query, diagnostics = oracle_one(model, data, "fit", 0)
        self.assertEqual(value.shape, (1024,))
        self.assertEqual(len(set((*support, *query))), 4)
        self.assertEqual([d["step"] for d in diagnostics], [16, 64, 256])
        self.assertEqual(model.forwards, 2 * 256 + 2 * 3)
        self.assertEqual(COUNTS["source_backward_calls"] - before["source_backward_calls"], 256)
        self.assertEqual(COUNTS["source_Adam"] - before["source_Adam"], 256)


if __name__ == "__main__":
    unittest.main()
