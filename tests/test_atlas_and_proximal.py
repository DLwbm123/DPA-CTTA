import unittest

import torch

from dpa_ctta.anchors import DistilledPotentialAnchor
from dpa_ctta.atlas import DistilledPotentialAtlas
from dpa_ctta.proximal import closed_form_update
from dpa_ctta.types import AtlasResult
from common import frozen_feature, make_anchor, make_atlas


class AnchorTests(unittest.TestCase):
    def setUp(self):
        self.anchor, self.descriptor = make_anchor("polyp")

    def test_synthetic_image_range(self):
        self.assertTrue(((self.anchor.image > 0) & (self.anchor.image < 1)).all())

    def test_synthetic_mask_range(self):
        self.assertTrue(((self.anchor.soft_mask > 0) & (self.anchor.soft_mask < 1)).all())

    def test_precision_strictly_exceeds_floor(self):
        self.assertTrue((self.anchor.precision > self.anchor.precision_floor).all())

    def test_descriptor_is_derived_not_parameter(self):
        self.assertNotIn("descriptor", dict(self.anchor.named_parameters()))

    def test_descriptor_uses_synthetic_tensors(self):
        first = self.anchor.descriptor(self.descriptor, frozen_feature(self.anchor.image))
        with torch.no_grad():
            self.anchor.image_logits.add_(0.5)
        second = self.anchor.descriptor(self.descriptor, frozen_feature(self.anchor.image))
        self.assertFalse(torch.equal(first, second))

    def test_invalid_spatial_shapes_rejected(self):
        with self.assertRaises(ValueError):
            DistilledPotentialAnchor(
                torch.zeros(1, 3, 8, 8),
                torch.zeros(1, 1, 4, 4),
                torch.zeros(16),
                torch.zeros(16, 25),
                torch.zeros(16),
            )


class AtlasTests(unittest.TestCase):
    def setUp(self):
        self.atlas = make_atlas("polyp")
        self.query = torch.randn(1, self.atlas.descriptor.output_dim)

    def test_weights_nonnegative(self):
        result = self.atlas(self.query, frozen_feature)
        self.assertTrue((result.weights >= 0).all())

    def test_weights_sum_to_one(self):
        result = self.atlas(self.query, frozen_feature)
        self.assertTrue(torch.allclose(result.weights.sum(1), torch.ones(1)))

    def test_weights_are_finite(self):
        self.assertTrue(torch.isfinite(self.atlas(self.query, frozen_feature).weights).all())

    def test_precision_and_rhs_shapes(self):
        result = self.atlas(self.query, frozen_feature)
        self.assertEqual(result.precision.shape, (1, 16))
        self.assertEqual(result.rhs.shape, (1, 16))

    def test_nearest_anchor_gets_larger_weight(self):
        anchors = self.atlas.anchor_descriptors(frozen_feature)
        result = self.atlas(anchors[:1], frozen_feature)
        self.assertGreater(float(result.weights[0, 0]), float(result.weights[0, 1]))

    def test_local_map_affects_rhs(self):
        before = self.atlas(self.query, frozen_feature).rhs.clone()
        with torch.no_grad():
            self.atlas.anchors[0].local_map.add_(0.1)
        after = self.atlas(self.query, frozen_feature).rhs
        self.assertFalse(torch.equal(before, after))

    def test_local_chart_uses_unscaled_descriptor_delta(self):
        anchor = self.atlas.anchors[0]
        single = DistilledPotentialAtlas(
            [anchor], self.atlas.descriptor, self.atlas.scaler, self.atlas.temperature
        )
        anchor_descriptor = single.anchor_descriptors(frozen_feature)
        query = anchor_descriptor + 0.2
        single.scaler.set_statistics(
            torch.zeros_like(single.scaler.center), torch.full_like(single.scaler.scale, 2)
        )
        result = single(query, frozen_feature)
        expected_center = anchor.state_center + anchor.local_map @ torch.full_like(query[0], 0.2)
        self.assertTrue(torch.allclose(result.rhs / result.precision, expected_center[None], atol=1e-6))

    def test_anchor_permutation_invariance(self):
        result = self.atlas(self.query, frozen_feature)
        permuted = DistilledPotentialAtlas(
            list(reversed(self.atlas.anchors)),
            self.atlas.descriptor,
            self.atlas.scaler,
            self.atlas.temperature,
        )(self.query, frozen_feature)
        self.assertTrue(torch.allclose(result.precision, permuted.precision, atol=1e-7))
        self.assertTrue(torch.allclose(result.rhs, permuted.rhs, atol=1e-7))

    def test_all_anchor_fields_receive_gradient(self):
        result = self.atlas(self.query, frozen_feature)
        (result.rhs.square().mean() + result.precision.mean()).backward()
        for anchor in self.atlas.anchors:
            for name in ("image_logits", "mask_logits", "state_center", "local_map", "precision_raw"):
                gradient = getattr(anchor, name).grad
                self.assertIsNotNone(gradient, name)
                self.assertTrue(torch.isfinite(gradient).all(), name)
                self.assertGreater(int(torch.count_nonzero(gradient)), 0, name)

    def test_empty_atlas_rejected(self):
        with self.assertRaises(ValueError):
            DistilledPotentialAtlas([], self.atlas.descriptor)

    def test_nonpositive_temperature_rejected(self):
        with self.assertRaises(ValueError):
            DistilledPotentialAtlas(list(self.atlas.anchors), self.atlas.descriptor, temperature=0)


class ProximalTests(unittest.TestCase):
    def setUp(self):
        self.previous = torch.tensor([[0.3, -0.2]], dtype=torch.float64)
        self.atlas = AtlasResult(
            weights=torch.tensor([[0.25, 0.75]], dtype=torch.float64),
            precision=torch.tensor([[2.0, 4.0]], dtype=torch.float64),
            rhs=torch.tensor([[1.0, -2.0]], dtype=torch.float64),
        )

    def test_closed_form_value(self):
        result = closed_form_update(self.previous, self.atlas, 1.5, 0.1)
        expected = (self.atlas.rhs + 1.5 * self.previous) / (self.atlas.precision + 1.5)
        self.assertTrue(torch.equal(result.state, expected))

    def test_first_order_condition(self):
        result = closed_form_update(self.previous, self.atlas, 1.5, 0.1)
        gradient = self.atlas.precision * result.state - self.atlas.rhs + 1.5 * (result.state - self.previous)
        self.assertLess(float(gradient.abs().max()), 1e-12)

    def test_matches_direct_diagonal_solve(self):
        result = closed_form_update(self.previous, self.atlas, 1.5, 0.1)
        matrix = torch.diag(self.atlas.precision[0] + 1.5)
        direct = torch.linalg.solve(matrix, self.atlas.rhs[0] + 1.5 * self.previous[0])
        self.assertLess(float((result.state[0] - direct).abs().max()), 1e-7)

    def test_contraction_bound_formula(self):
        result = closed_form_update(self.previous, self.atlas, 1.5, 0.1)
        self.assertAlmostEqual(result.contraction_bound, 1.5 / 1.6)

    def test_contraction_inequality(self):
        other = self.previous + torch.tensor([[0.4, -0.6]], dtype=torch.float64)
        first = closed_form_update(self.previous, self.atlas, 1.5, 0.1).state
        second = closed_form_update(other, self.atlas, 1.5, 0.1).state
        bound = 1.5 / 1.6 * torch.linalg.vector_norm(self.previous - other)
        self.assertLessEqual(float(torch.linalg.vector_norm(first - second)), float(bound) + 1e-12)

    def test_larger_lambda_stays_closer_to_history(self):
        small = closed_form_update(self.previous, self.atlas, 0.2, 0.1).state
        large = closed_form_update(self.previous, self.atlas, 20.0, 0.1).state
        self.assertLess(float((large - self.previous).norm()), float((small - self.previous).norm()))

    def test_result_preserves_atlas_fields(self):
        result = closed_form_update(self.previous, self.atlas, 1.5, 0.1)
        self.assertIs(result.weights, self.atlas.weights)
        self.assertIs(result.precision, self.atlas.precision)
        self.assertIs(result.rhs, self.atlas.rhs)

    def test_nonpositive_lambda_rejected(self):
        with self.assertRaises(ValueError):
            closed_form_update(self.previous, self.atlas, 0, 0.1)

    def test_shape_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            closed_form_update(torch.zeros(1, 3), self.atlas, 1.0, 0.1)

    def test_precision_below_claimed_floor_rejected(self):
        too_small = AtlasResult(self.atlas.weights, torch.full((1, 2), 0.01), self.atlas.rhs)
        with self.assertRaises(ValueError):
            closed_form_update(self.previous, too_small, 1.0, 0.1)


if __name__ == "__main__":
    unittest.main()
