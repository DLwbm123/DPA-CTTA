import math
import unittest
import torch
from dpa_ctta.medical_losses import medical_loss
from dpa_ctta.proxy_loss import FixedProxy, ProxyProvenance


def half_plane(size, classes=1):
    """Exact pixel-center distance to opposite class, divided by image diagonal."""
    mask = torch.zeros(1, classes, size, size)
    mask[..., :size // 2] = 1
    columns = torch.arange(size)
    distance = torch.where(columns < size // 2, columns - size // 2, columns - size // 2 + 1)
    distance = distance.float().view(1, 1, 1, size).expand_as(mask) / math.hypot(size, size)
    return mask, distance.clone()


class MedicalLossTests(unittest.TestCase):
    def test_hand_reference_per_image_independent_channel(self):
        mask = torch.tensor([[[[1., 0.], [0., 0.]], [[1., 1.], [0., 0.]]],
                             [[[0., 0.], [0., 0.]], [[1., 1.], [1., 1.]]]], dtype=torch.float64)
        logits = torch.linspace(-2, 2, 16, dtype=torch.float64).reshape_as(mask).requires_grad_()
        distance = (1 - 2 * mask) / math.sqrt(8)
        parts = medical_loss(logits, mask, distance, .1)
        expected, boundaries = [], []
        for image in range(2):
            for channel in range(2):
                z, y = logits[image, channel], mask[image, channel]
                p = z.sigmoid()
                bce = torch.logaddexp(torch.zeros_like(z), z) - z * y
                groups = [bce[y == label].mean() for label in (0, 1) if (y == label).any()]
                expected.append(torch.stack(groups).mean() + 1 - (2 * (p * y).sum() + 1e-6) / (p.sum() + y.sum() + 1e-6))
                if y.min() != y.max():
                    boundaries.append((distance[image, channel] * (p - y)).mean())
        self.assertTrue(torch.allclose(parts.region, torch.stack(expected).mean(), atol=1e-12))
        self.assertTrue(torch.allclose(parts.boundary, torch.stack(boundaries).mean(), atol=1e-12))
        self.assertTrue(torch.allclose(parts.total, parts.region + .1 * parts.boundary))
        self.assertEqual(parts.defined_boundary_pairs, 2)

    def test_boundary_gradient_corrects_missed_inside_and_false_outside(self):
        mask, d = half_plane(4)
        logits = ((1 - mask) * 4 - mask * 4).requires_grad_()
        parts = medical_loss(logits, mask, d)
        grad, = torch.autograd.grad(parts.boundary, logits)
        self.assertTrue((grad[mask == 1] < 0).all())
        self.assertTrue((grad[mask == 0] > 0).all())
        with self.assertRaisesRegex(ValueError, 'sign mismatch'):
            medical_loss(logits, mask, -d)

    def test_empty_full_retain_region_gradients_and_zero_boundary(self):
        for value in (0., 1.):
            logits = torch.zeros(2, 2, 4, 4, requires_grad=True)
            mask = torch.full_like(logits, value)
            parts = medical_loss(logits, mask)
            self.assertEqual(parts.defined_boundary_pairs, 0)
            self.assertEqual(float(parts.boundary), 0.)
            parts.total.backward()
            self.assertTrue((logits.grad > 0).all() if value == 0 else (logits.grad < 0).all())

    def test_known_mask_and_distance_validation(self):
        mask, d = half_plane(4)
        for bad_mask, bad_d in ((mask.requires_grad_(), d), (mask.detach() + .1, d),
                                (mask.detach(), d * float('nan')), (mask.detach(), None)):
            with self.assertRaises(ValueError):
                medical_loss(torch.zeros_like(mask), bad_mask, bad_d)
        with self.assertRaises(ValueError):
            FixedProxy(torch.zeros(1, 3, 4, 4), mask.detach(), d, 'target_evaluator')
        proxy = FixedProxy(torch.zeros(1, 3, 4, 4), mask.detach(), d, ProxyProvenance.FIXTURE)
        mask.detach().zero_()
        self.assertGreater(float(proxy.mask.sum()), 0)
