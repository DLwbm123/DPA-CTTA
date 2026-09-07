import inspect
import unittest

import torch

from dpa_ctta.losses import (
    classwise_dense_feature_loss,
    distillation_loss,
    state_matching_loss,
    trajectory_increment_loss,
    worst_class_segmentation_loss,
)
from dpa_ctta.manifold import MultiScaleLatentFiLM
from dpa_ctta.offline.oracle import optimize_oracle_state
from dpa_ctta.offline.trajectories import trajectory_increments
from common import TinySource


class LossTests(unittest.TestCase):
    def test_worst_class_loss_is_scalar(self):
        loss = worst_class_segmentation_loss(torch.randn(2, 2, 8, 8), torch.rand(2, 2, 8, 8))
        self.assertEqual(loss.ndim, 0)

    def test_good_segmentation_has_lower_loss(self):
        label = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
        good = torch.where(label.bool(), torch.tensor(8.0), torch.tensor(-8.0))
        bad = -good
        self.assertLess(float(worst_class_segmentation_loss(good, label)), float(worst_class_segmentation_loss(bad, label)))

    def test_segmentation_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            worst_class_segmentation_loss(torch.zeros(1, 1, 8, 8), torch.zeros(1, 2, 8, 8))

    def test_state_matching_zero_for_equal_states(self):
        state = torch.randn(2, 16)
        self.assertEqual(float(state_matching_loss(state, state)), 0.0)

    def test_state_matching_positive_for_different_states(self):
        self.assertGreater(float(state_matching_loss(torch.zeros(1, 16), torch.ones(1, 16))), 0)

    def test_state_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            state_matching_loss(torch.zeros(1, 16), torch.zeros(2, 16))

    def test_increment_loss_zero_for_matching_trajectories(self):
        states = torch.randn(4, 16)
        self.assertEqual(float(trajectory_increment_loss(states, states)), 0.0)

    def test_increment_loss_ignores_common_offset(self):
        states = torch.randn(4, 16)
        self.assertLess(float(trajectory_increment_loss(states, states + 3)), 1e-12)

    def test_single_state_trajectory_rejected(self):
        with self.assertRaises(ValueError):
            trajectory_increment_loss(torch.zeros(1, 16), torch.zeros(1, 16))

    def test_dense_feature_loss_is_scalar(self):
        feature = torch.rand(1, 4, 8, 8)
        mask = torch.rand(1, 1, 16, 16)
        self.assertEqual(classwise_dense_feature_loss([feature], [feature.clone()], mask).ndim, 0)

    def test_dense_feature_loss_zero_for_equal_features(self):
        feature = torch.rand(1, 4, 8, 8)
        mask = torch.rand(1, 1, 16, 16)
        self.assertLess(float(classwise_dense_feature_loss([feature], [feature.clone()], mask)), 1e-12)

    def test_dense_feature_gradient_connected(self):
        synthetic = torch.rand(1, 4, 8, 8, requires_grad=True)
        mask = torch.rand(1, 1, 16, 16, requires_grad=True)
        loss = classwise_dense_feature_loss([synthetic], [torch.zeros_like(synthetic)], mask)
        loss.backward()
        for tensor in (synthetic, mask):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(int(torch.count_nonzero(tensor.grad)), 0)

    def test_distillation_loss_combines_four_terms(self):
        predicted = torch.ones(3, 16)
        oracle = torch.zeros(3, 16)
        synthetic = torch.ones(1, 2, 4, 4)
        reference = torch.zeros_like(synthetic)
        mask = torch.full((1, 1, 4, 4), 0.5)
        result = distillation_loss(
            torch.tensor(2.0), predicted[-1:], oracle[-1:], predicted, oracle,
            [synthetic], [reference], mask, alpha=2, beta=3, gamma=4
        )
        expected = (
            2
            + 2 * state_matching_loss(predicted[-1:], oracle[-1:])
            + 3 * trajectory_increment_loss(predicted, oracle)
            + 4 * classwise_dense_feature_loss([synthetic], [reference], mask)
        )
        self.assertTrue(torch.equal(result, expected))


class OfflineTests(unittest.TestCase):
    def test_oracle_requires_named_source_label(self):
        self.assertIn("source_label", inspect.signature(optimize_oracle_state).parameters)

    def test_oracle_rejects_missing_source_label(self):
        model = MultiScaleLatentFiLM(TinySource(), {"low": 4, "mid": 4, "tail": 1})
        try:
            with self.assertRaises(ValueError):
                optimize_oracle_state(model, torch.rand(1, 3, 16, 16), None)
        finally:
            model.close()

    def test_oracle_returns_detached_state(self):
        model = MultiScaleLatentFiLM(TinySource(), {"low": 4, "mid": 4, "tail": 1})
        try:
            result = optimize_oracle_state(
                model, torch.rand(1, 3, 16, 16), torch.rand(1, 1, 16, 16), steps=1
            )
            self.assertEqual(result.shape, (1, 16))
            self.assertFalse(result.requires_grad)
        finally:
            model.close()

    def test_oracle_source_base_gradients_remain_none(self):
        source = TinySource()
        model = MultiScaleLatentFiLM(source, {"low": 4, "mid": 4, "tail": 1})
        try:
            optimize_oracle_state(
                model, torch.rand(1, 3, 16, 16), torch.rand(1, 1, 16, 16), steps=1
            )
            self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))
        finally:
            model.close()

    def test_trajectory_increment_values(self):
        states = torch.tensor([[0.0, 1.0], [2.0, 4.0], [5.0, 9.0]])
        expected = torch.tensor([[2.0, 3.0], [3.0, 5.0]])
        self.assertTrue(torch.equal(trajectory_increments(states), expected))

    def test_trajectory_increment_rejects_short_input(self):
        with self.assertRaises(ValueError):
            trajectory_increments(torch.zeros(1, 16))


if __name__ == "__main__":
    unittest.main()
