import unittest

import torch

from dpa_ctta.r7_shared.source import Record, SourceData
from dpa_ctta.r8_ba.calibration import Calibrator, score_validation, select_checkpoint, select_family, validate_episode
from dpa_ctta.r8_ba.methods import R8B
from dpa_ctta.r8_ba.oracles import Oracles
from dpa_ctta.r8_ba.schedule import CURRICULA
from dpa_ctta.r8_ba.trainer import SAVE_STEPS
from test_trainer import TinySegmenter


class TestCalibration(unittest.TestCase):
    def test_calibration_step_and_source_only_selection(self):
        torch.manual_seed(11)
        folds = {k: [f"{k}{i}" for i in range(n)] for k, n in (("fit", 32), ("cal", 4), ("val", 4))}
        data = SourceData([Record(g, k, torch.rand(1, 3, 8, 8), torch.randint(2, (1, 2, 8, 8)).float())
                           for k, groups in folds.items() for g in groups], folds)

        def oracles(fold, count):
            pair, query = tuple(folds[fold][:2]), tuple(folds[fold][2:4])
            return Oracles(fold, 0.1, torch.zeros(1024, count), (pair,) * count,
                           (query,) * count, ((),) * count)

        method = R8B(torch.eye(1024, dtype=torch.float64)[:, :32], 0.1)
        method.observer.fit_scaler(torch.randn(4, 134), "fit")
        calibrator = Calibrator(TinySegmenter(), method, data, oracles("cal", 128), "synthetic")
        self.assertEqual(calibrator.step()["step"], 1)
        snapshot = calibrator.snapshot()
        expected = calibrator.step()
        expected_raw = calibrator.method.cal_raw.detach().clone()
        calibrator.restore(snapshot)
        self.assertEqual(expected, calibrator.step())
        self.assertTrue(torch.equal(expected_raw, calibrator.method.cal_raw.detach()))
        calibrator.method.freeze()
        result = validate_episode(TinySegmenter(), calibrator.method, data, oracles("val", 128), 0)
        self.assertEqual(result["severity"], "clean")
        rows = [dict(episode=index, curriculum=mode, soft_Dice=0.5)
                for index, mode in enumerate(mode for mode in CURRICULA for _ in range(16))]
        self.assertEqual(score_validation((rows, rows)), 0.5)
        self.assertEqual(select_checkpoint({step: (rows, rows) for step in SAVE_STEPS})[0], 1000)
        self.assertEqual(select_family({"B2": 0.5, "B1": 0.5}, {"B2": 2., "B1": 1.}), "B1")


if __name__ == "__main__":
    unittest.main()
