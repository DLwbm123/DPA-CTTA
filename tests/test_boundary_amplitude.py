import unittest
import torch

from dpa_ctta.boundary_amplitude.method import match, random_direction


class AmplitudeTests(unittest.TestCase):
    def test_exact_radius_and_direction(self):
        x = torch.tensor([3e-5, -4e-5])
        f = torch.tensor([0., .05])
        a, d = match(x, f)
        self.assertTrue(torch.allclose(a, torch.tensor([.03, -.04]), atol=1e-8))
        self.assertAlmostEqual(d['multiplier'], 1000., places=3)
        self.assertGreater(d['direction_cosine'], .999999)
        reverse, _ = match(-x, f)
        self.assertTrue(torch.equal(reverse, -a))

    def test_zero_and_nonfinite_are_not_hidden(self):
        self.assertTrue(torch.equal(match(torch.ones(4), torch.zeros(4))[0], torch.zeros(4)))
        with self.assertRaises(ValueError):
            match(torch.zeros(4), torch.ones(4)*.01)
        with self.assertRaises(ValueError):
            match(torch.tensor([float('nan')]), torch.tensor([.01]))

    def test_fixed_null_direction(self):
        a = random_direction(512, 'cpu');b = random_direction(512, 'cpu')
        self.assertTrue(torch.equal(a, b))
        f = torch.zeros(512);f[0] = .05
        self.assertAlmostEqual(float(match(a, f)[0].norm()), .05, places=7)

    def test_functional_bn_gets_real_matching_offsets(self):
        from torch import nn
        from dpa_ctta.r78_cross_patient.objective import BNState
        class Toy(nn.Module):
            def __init__(self):
                super().__init__();self.up3 = nn.Module();self.up3.bn = nn.BatchNorm2d(2)
                self.conv = nn.Conv2d(3, 2, 1)
            def forward(self, image):
                return self.up3.bn(self.conv(image))
        torch.manual_seed(12);model = Toy();host = BNState(model)
        x = torch.rand(1, 3, 16, 16)
        before = {n: p.clone() for n, p in model.named_parameters()}
        a, _ = match(torch.tensor([3e-5, -4e-5, 0., 0.]), torch.tensor([.05, 0., 0., 0.]))
        z = host(x, a);z0 = host(x, host.zero())
        self.assertGreater(float((z-z0).square().mean().detach()), 0)
        self.assertTrue(all(torch.equal(p, before[n]) for n, p in model.named_parameters()))
        from dpa_ctta.boundary_amplitude.run import ARMS
        self.assertEqual(len(ARMS), 12)
        self.assertEqual((len(ARMS)-2)*8*4*3+2*8*4, 1024)


if __name__ == '__main__':
    unittest.main()
