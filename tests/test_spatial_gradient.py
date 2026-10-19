import copy
import json
from pathlib import Path
import unittest
import torch
from torch import nn
from dpa_ctta.srg_ctta.spatial_gradient import Host, KINDS, spatial_gradient
from dpa_ctta.srg_ctta.spatial_run import validate, JOBS


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm2d(3)
        self.head = nn.Conv2d(3, 2, 1)
    def forward(self, x):
        return self.head(self.bn(x))


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_projection_and_magnitude_controls(self):
        z = torch.randn(1, 2, 12, 12)
        g = -z*z.sigmoid()*(-z).sigmoid()/z.numel()
        results = {k: spatial_gradient(z, g, k, 53) for k in KINDS}
        target = results['TANGENT'][0]
        for kind in KINDS[:-1]:
            self.assertAlmostEqual(float(results[kind][0].double().norm()), float(target.double().norm()), delta=1e-9)
        self.assertLess(results['TANGENT'][1]['output_radial_cosine_max'], 5e-6)
        self.assertGreater(results['NORM_ONLY'][1]['output_gradient_cosine'], .999999)
        self.assertGreater(results['TANGENT'][1]['tangent_fraction'], 1e-6)
        self.assertTrue(torch.equal(results['ENT_NORM'][0], g))
        replay = spatial_gradient(z, g, 'RANDOM_ORTH', 53)[0]
        self.assertTrue(torch.equal(results['RANDOM_ORTH'][0], replay))
        self.assertFalse(torch.equal(results['RANDOM_ORTH'][0], target))

    def test_degenerate_has_no_invented_direction(self):
        z = torch.ones(1, 2, 8, 8)
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            spatial_gradient(z, -z*.1, 'TANGENT', 53)
        with self.assertRaisesRegex(ValueError, 'zero'):
            spatial_gradient(z, z*0, 'ENT_NORM', 53)
        with self.assertRaises(ValueError):
            spatial_gradient(z[:, :1], z[:, :1], 'TANGENT', 53)

    def test_bn_updates_reset_and_fixed_buffers(self):
        base = Model().eval()
        x = torch.rand(1, 3, 16, 16)
        outputs = []
        for kind in KINDS:
            h = Host(base, kind)
            initial = copy.deepcopy(h.model.state_dict())
            pre, post, _, _, d = h.step(x, None, None, None, 53)
            self.assertFalse(torch.equal(pre, post))
            self.assertAlmostEqual(d['steps'][0]['actual_step_norm'], .01, delta=5e-7)
            self.assertGreaterEqual(d['steps'][0]['direction_cosine'], .999999)
            self.assertEqual(d['forwards'], 2)
            self.assertEqual(d['loss_backward_calls'], 1)
            self.assertTrue(torch.equal(initial['bn.running_mean'], h.model.bn.running_mean))
            self.assertTrue(torch.equal(initial['head.weight'], h.model.head.weight))
            affine = [p.clone() for p in h.params]
            h.reset_optimizer()
            self.assertTrue(all(torch.equal(p, v) for p, v in zip(h.params, affine)))
            self.assertFalse(h.opt.state)
            h.reset()
            self.assertTrue(all(torch.equal(v, h.model.state_dict()[n]) for n, v in initial.items()))
            _, replay, _, _, _ = h.step(x, None, None, None, 53)
            self.assertTrue(torch.equal(post, replay))
            outputs.append(post)
        self.assertFalse(torch.equal(outputs[0], outputs[-1]))

    def test_frozen_matrix(self):
        root = Path(__file__).resolve().parents[1]
        c = json.loads((root/'configs/spatial_logit_gradient_v1.json').read_text())
        validate(c)
        self.assertEqual(len(JOBS), 12)
        self.assertEqual(160*len(JOBS)+400, 2320)
        c['policies_by_arm']['TANGENT'] = ['FRESH']
        with self.assertRaises(AssertionError):
            validate(c)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
