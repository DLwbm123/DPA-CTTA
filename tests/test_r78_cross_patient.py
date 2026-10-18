"""Synthetic engineering checks; no checkpoint, clinical images, or GPU."""
import unittest

import torch
from torch import nn

from dpa_ctta.r78_cross_patient.objective import (
    BNState, Field, ParameterPredictor, direct_logits, energy, project,
    reference, ribbon, sample, update,
)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261018)
        torch.set_num_threads(1)
        self.image = torch.rand(1, 3, 32, 32, dtype=torch.double)
        yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing='ij')
        self.logits = torch.stack((15.5-xx, 8.5-xx)).double().unsqueeze(0)
        self.ref = reference(self.image, self.logits, size=32)

    def test_content_split_and_label_separation(self):
        from dpa_ctta.r78_cross_patient.run import prepare_metadata
        rows = [dict(sample_id=f'{i:03d}', group_id=str(i), image_relative=f'image/{i}.png',
                     mask_relative=f'mask/{i}.png', image_size=[800, 800]) for i in range(40)]
        panel, final = prepare_metadata(rows, '/synthetic')
        self.assertEqual([len(panel[k]) for k in ('train', 'validation', 'support', 'query')], [16, 8, 8, 8])
        self.assertEqual(panel, prepare_metadata(rows[::-1], '/synthetic')[0])
        self.assertTrue(all('mask_path' not in r for r in panel['query']+panel['support']))
        self.assertEqual([r['group_id'] for r in final], [r['group_id'] for r in panel['query']])
        with self.assertRaises(ValueError):
            prepare_metadata(rows+[rows[0]], '/synthetic')

    def test_fixed_sampler_and_trust_ball(self):
        field = torch.arange(16, dtype=torch.double).reshape(1, 4, 4).requires_grad_()
        points = torch.tensor([[1.5, 2.5], [3., 3.]], dtype=torch.double)
        self.assertTrue(torch.allclose(sample(field, points), torch.tensor([[11.5, 15.]], dtype=torch.double)))
        self.assertTrue(torch.isfinite(torch.autograd.grad(sample(field, points).sum(), field)[0]).all())
        with self.assertRaises(ValueError):
            sample(field, torch.tensor([[-1., 2.]]))
        self.assertTrue(torch.equal(project(torch.zeros(8)), torch.zeros(8)))
        self.assertAlmostEqual(float(project(torch.ones(8)).norm()), .05, places=6)

    def test_zero_energy_and_empty_support(self):
        state = torch.zeros((), dtype=torch.double, requires_grad=True)
        delta = torch.zeros(2, 1, 32, 32, dtype=torch.double)
        loss = energy(self.logits+state, self.ref, delta)
        self.assertEqual(float(loss.detach()), 0.)
        self.assertEqual(float(torch.autograd.grad(loss, state)[0]), 0.)
        blank = reference(self.image, torch.full_like(self.logits, -10), size=32)
        self.assertTrue(blank.empty)
        state = torch.zeros((), dtype=torch.double, requires_grad=True)
        loss = energy(self.logits+state, blank, delta)
        self.assertEqual(float(update(state, loss).detach()), 0.)
        self.assertTrue(torch.equal(direct_logits(self.logits, self.ref, delta), self.logits))

    def test_signed_update_and_mixed_derivative(self):
        def outer(phi, graph):
            state = torch.zeros((), dtype=torch.double, requires_grad=True)
            delta = phi.tanh().expand(2, 1, 32, 32)
            adapted = update(state, energy(self.logits+state, self.ref, delta), create_graph=graph)
            return (adapted-.2).square(), adapted
        phi = torch.tensor(.1, dtype=torch.double, requires_grad=True)
        loss, adapted = outer(phi, True)
        self.assertGreater(float(adapted.detach()), 0.)  # Positive delta moves foreground outwards.
        analytic = torch.autograd.grad(loss, phi)[0]
        eps = 1e-5
        plus, _ = outer(torch.tensor(.1+eps, dtype=torch.double), False)
        minus, _ = outer(torch.tensor(.1-eps, dtype=torch.double), False)
        numerical = (plus-minus)/(2*eps)
        self.assertNotEqual(float(analytic), 0.)
        self.assertTrue(torch.allclose(analytic, numerical, atol=1e-8, rtol=1e-4))

    def test_geometry_control_and_predictor(self):
        head = Field().double()
        nn.init.normal_(head.layers[-1].weight, std=.01)
        a = self.ref.inputs.clone()
        b = a.clone()
        b[:, :4] = torch.rand_like(b[:, :4])
        self.assertTrue(torch.equal(head(a, True), head(b, True)))
        self.assertFalse(torch.equal(head(a), head(b)))
        self.assertLessEqual(float(head(a).abs().max().detach()), 1.)
        predictor = ParameterPredictor(8).double()
        self.assertTrue(torch.equal(predictor(a), torch.zeros(8, dtype=torch.double)))
        predictor.readout.bias.data.fill_(1)
        self.assertLessEqual(float(predictor(a).norm().detach()), .0500001)
        self.assertTrue(torch.equal(predictor(a, empty=True), torch.zeros(8, dtype=torch.double)))

    def test_ribbon_ties_and_scorer(self):
        ref = reference(torch.zeros_like(self.image), self.logits, size=32)
        self.assertTrue(torch.equal(ribbon(ref), torch.zeros(2, 1, 32, 32, dtype=torch.double)))
        from dpa_ctta.r78_cross_patient.score import interval, metric
        import numpy as np
        mask = np.zeros((2, 32, 32), dtype=bool)
        mask[0, 4:28, 4:28] = True;mask[1, 10:22, 10:22] = True
        mm, rr = metric(mask, mask, mask)
        self.assertEqual(mm[0]['dice_percent'], 100.)
        self.assertEqual(mm[1]['assd_pixel'], 0.)
        self.assertEqual(rr['predicted_vcdr'], .5)
        self.assertEqual(interval(np.ones((8, 4, 2))), [1., 1.])
        mm, rr = metric(np.zeros_like(mask), mask, mask)
        self.assertIsNone(mm[0]['assd_pixel'])
        self.assertIsNone(rr['predicted_vcdr'])

    def test_functional_bn_is_readonly_and_differentiable(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.up3 = nn.Module()
                self.up3.bn = nn.BatchNorm2d(2)
                self.conv = nn.Conv2d(3, 2, 1)

            def forward(self, x):
                return self.up3.bn(self.conv(x)), None

        model = Toy().double()
        host = BNState(model)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        a = host.zero()
        z = host(self.image, a)
        self.assertTrue(torch.equal(z, host(self.image, a)))
        grad, = torch.autograd.grad(z.square().mean(), a, create_graph=True)
        self.assertTrue(torch.isfinite(torch.autograd.grad(grad.sum(), a)[0]).all())
        self.assertEqual(host.dimensions, 4)
        self.assertTrue(all(torch.equal(v, before[k]) for k, v in model.state_dict().items()))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))


if __name__ == '__main__':
    unittest.main()
