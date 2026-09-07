import inspect
import unittest

import torch

from dpa_ctta.descriptors import DescriptorScaler, FrozenDescriptor
from common import frozen_feature


class DescriptorTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(8)
        self.image = torch.rand(2, 3, 16, 16)

    def test_fundus_output_dimension(self):
        descriptor = FrozenDescriptor("fundus", 4)
        mask = torch.rand(2, 2, 16, 16)
        self.assertEqual(descriptor(self.image, mask, frozen_feature(self.image)).shape, (2, 33))

    def test_polyp_output_dimension(self):
        descriptor = FrozenDescriptor("polyp", 4)
        mask = torch.rand(2, 1, 16, 16)
        self.assertEqual(descriptor(self.image, mask, frozen_feature(self.image)).shape, (2, 25))

    def test_descriptor_is_finite(self):
        descriptor = FrozenDescriptor("polyp", 4)
        result = descriptor(self.image, torch.rand(2, 1, 16, 16), frozen_feature(self.image))
        self.assertTrue(torch.isfinite(result).all())

    def test_descriptor_signature_has_no_history_state(self):
        names = set(inspect.signature(FrozenDescriptor.forward).parameters)
        self.assertFalse(names & {"z", "z_prev", "state", "previous_state"})

    def test_area_increases_with_soft_mass(self):
        descriptor = FrozenDescriptor("polyp", 4)
        low = descriptor._anatomy(torch.full((1, 1, 8, 8), 0.1))[0, 0]
        high = descriptor._anatomy(torch.full((1, 1, 8, 8), 0.9))[0, 0]
        self.assertGreater(float(high), float(low))

    def test_centroid_tracks_horizontal_shift(self):
        descriptor = FrozenDescriptor("polyp", 4)
        left = torch.zeros(1, 1, 8, 8)
        right = torch.zeros_like(left)
        left[..., :2] = 1
        right[..., -2:] = 1
        self.assertLess(float(descriptor._anatomy(left)[0, 1]), float(descriptor._anatomy(right)[0, 1]))

    def test_constant_mask_has_zero_boundary(self):
        descriptor = FrozenDescriptor("polyp", 4)
        anatomy = descriptor._anatomy(torch.full((1, 1, 8, 8), 0.5))
        self.assertEqual(float(anatomy[0, 6]), 0.0)

    def test_fundus_ratio_uses_cup_over_disc(self):
        descriptor = FrozenDescriptor("fundus", 4)
        mask = torch.stack((torch.full((8, 8), 0.8), torch.full((8, 8), 0.4)))[None]
        ratio = descriptor._anatomy(mask)[0, -2]
        self.assertAlmostEqual(float(ratio), 0.5, places=6)

    def test_fundus_nesting_penalizes_cup_outside_disc(self):
        descriptor = FrozenDescriptor("fundus", 4)
        valid = torch.stack((torch.full((8, 8), 0.8), torch.full((8, 8), 0.4)))[None]
        invalid = valid.flip(1)
        self.assertEqual(float(descriptor._anatomy(valid)[0, -1]), 0.0)
        self.assertGreater(float(descriptor._anatomy(invalid)[0, -1]), 0)

    def test_polyp_compactness_is_nonnegative(self):
        descriptor = FrozenDescriptor("polyp", 4)
        value = descriptor._anatomy(torch.rand(1, 1, 8, 8))[0, -1]
        self.assertGreaterEqual(float(value), 0)

    def test_fixed_projection_is_not_parameter(self):
        descriptor = FrozenDescriptor("polyp", 4)
        self.assertNotIn("feature_projection", dict(descriptor.named_parameters()))
        self.assertIn("feature_projection", dict(descriptor.named_buffers()))

    def test_image_mask_feature_gradients_are_connected(self):
        descriptor = FrozenDescriptor("polyp", 4)
        image = torch.rand(1, 3, 16, 16, requires_grad=True)
        mask = torch.rand(1, 1, 16, 16, requires_grad=True)
        feature = frozen_feature(image)
        feature.retain_grad()
        descriptor(image, mask, feature).square().mean().backward()
        for tensor in (image, mask, feature):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(int(torch.count_nonzero(tensor.grad)), 0)

    def test_wrong_image_channels_rejected(self):
        descriptor = FrozenDescriptor("polyp", 4)
        with self.assertRaises(ValueError):
            descriptor(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), torch.rand(1, 4, 8, 8))

    def test_wrong_mask_classes_rejected(self):
        descriptor = FrozenDescriptor("fundus", 4)
        with self.assertRaises(ValueError):
            descriptor(torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8), torch.rand(1, 4, 8, 8))

    def test_wrong_feature_channels_rejected(self):
        descriptor = FrozenDescriptor("polyp", 4)
        with self.assertRaises(ValueError):
            descriptor(torch.rand(1, 3, 8, 8), torch.rand(1, 1, 8, 8), torch.rand(1, 3, 8, 8))


class ScalerTests(unittest.TestCase):
    def test_default_scaler_is_identity(self):
        scaler = DescriptorScaler(3)
        value = torch.tensor([[1.0, 2.0, 3.0]])
        self.assertTrue(torch.equal(scaler(value), value))

    def test_statistics_transform(self):
        scaler = DescriptorScaler(2).set_statistics(torch.tensor([1.0, 2.0]), torch.tensor([2.0, 4.0]))
        self.assertTrue(torch.equal(scaler(torch.tensor([[3.0, 6.0]])), torch.ones(1, 2)))

    def test_statistics_mark_fitted(self):
        scaler = DescriptorScaler(2).set_statistics(torch.zeros(2), torch.ones(2))
        self.assertTrue(bool(scaler.fitted))

    def test_nonpositive_scale_rejected(self):
        with self.assertRaises(ValueError):
            DescriptorScaler(2).set_statistics(torch.zeros(2), torch.tensor([1.0, 0.0]))

    def test_wrong_descriptor_dimension_rejected(self):
        with self.assertRaises(ValueError):
            DescriptorScaler(2)(torch.zeros(1, 3))


if __name__ == "__main__":
    unittest.main()
