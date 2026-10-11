"""Synthetic CPU tests only: no private data, source checkpoint or GPU admission."""
import copy
import math
import unittest
from unittest.mock import patch
import torch
from .method import Host, Sensor, candidates, photo_probe, filter_update, flatten
from ..r37_cw_orientation.method import Host as BaseHost, candidates as base_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal
from ..host_diagnostic import rng


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(42)
        cls.model = Tiny()
        cls.x = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)

    def host(self, config=None, base=False):
        return (BaseHost if base else Host)(None, config or candidates()[-1], 17011,
                                           'synthetic', device='cpu', model=copy.deepcopy(self.model))

    def test_disabled_full_snapshot_bitwise_and_controls(self):
        self.assertEqual(candidates()[:3], [*base_candidates()[:2], base_candidates()[3]])
        a = self.host(dict(candidates()[-1], joint_strength=0.))
        b = self.host(base_candidates()[3], base=True)
        try:
            for x in (self.x, self.x.roll(9, -1)):
                za, da = a.step(x); sa = a.snapshot()
                zb, db = b.step(x); sb = b.snapshot()
                self.assertTrue(torch.equal(za, zb))
                self.assertTrue(equal(da, db))
                self.assertTrue(equal(sa, sb))
                self.assertEqual(da['native'][0]['counts'], dict(forwards=8, backwards=1, base_adam=1, perturb=0, restore=0))
            a.check_frozen(True); b.check_frozen(True)
        finally:
            a.close(); b.close()

    def test_photo_and_sensor_reference_rng_and_channel_independence(self):
        x = self.x.clone(); x[..., 0, 0] = 0; x[..., -1, -1] = 1
        for index in range(2):
            y = photo_probe(x, index)
            self.assertTrue(torch.isfinite(y).all())
            self.assertTrue(torch.equal(y[..., 0, 0], x[..., 0, 0]))
            self.assertTrue(torch.equal(y[..., -1, -1], x[..., -1, -1]))
        self.assertTrue(torch.equal(photo_probe(x, 1)[:, 1], x[:, 1]) or torch.allclose(photo_probe(x, 1)[:, 1], x[:, 1], atol=1e-7))
        z = torch.randn(1, 2, 32, 32, requires_grad=True)
        predictions = torch.stack([z.detach().sigmoid()]*6)
        saved = rng(); s = Sensor(z, predictions)
        self.assertTrue(equal(saved, rng()))
        self.assertTrue(torch.equal(s(z), torch.zeros(2)))
        g = torch.autograd.grad(s(z)[0], z)[0]
        self.assertEqual(float(g[:, 1].abs().sum()), 0.)
        self.assertGreater(float(g[:, 0].norm()), 0.)
        flat = torch.zeros_like(z, requires_grad=True)
        empty = Sensor(flat, torch.full_like(predictions, .5))
        self.assertEqual(float(empty.weight.sum()), 0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(empty(flat).sum(), flat)[0]).all())

    def test_filter_closed_form_and_rank_zero(self):
        d = torch.tensor([.2, -.3, .4], dtype=torch.float64)
        h = torch.tensor([4., 9., 16.], dtype=torch.float64)
        a = torch.tensor([[2., 0.], [0., 0.], [0., 0.]], dtype=torch.float64)
        b = a.clone()
        out, info = filter_update(d, h, a, b)
        expected = d.clone(); expected[0] /= 1+1/1.01
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
        self.assertEqual(info['rank'], 1)
        self.assertTrue(torch.equal(filter_update(d, h, a*0, b)[0], d))
        self.assertTrue(torch.equal(filter_update(d, h, a, b, 0)[0], d))
        torch.testing.assert_close(filter_update(d, h, a, b*0)[0], d)
        for strength in (-1., float('nan')):
            with self.assertRaises(ValueError): filter_update(d, h, a, b, strength)
        with self.assertRaises(ValueError): filter_update(d, h*0, a, b)

    def test_filter_noncommuting_dense_reference_complement_contraction(self):
        gen = torch.Generator().manual_seed(7)
        d = torch.randn(9, dtype=torch.float64, generator=gen)
        h = torch.rand(9, dtype=torch.float64, generator=gen)+.1
        a = torch.randn(9, 2, dtype=torch.float64, generator=gen)
        b = torch.randn(9, 2, dtype=torch.float64, generator=gen)
        out, info = filter_update(d, h, a, b)
        x, y = a/h.sqrt()[:, None], b/h.sqrt()[:, None]
        q, _ = torch.linalg.qr(x)
        aa = (q.T@x)@(x.T@q)/x.square().sum()
        ss = (q.T@y)@(y.T@q)/y.square().sum()
        self.assertGreater(float((aa@ss-ss@aa).norm()), 1e-3)
        e, v = torch.linalg.eigh(aa+.01*torch.eye(2, dtype=torch.float64))
        r = (v/e.sqrt())@v.T
        k = torch.linalg.inv(torch.eye(2, dtype=torch.float64)+r@ss@r)
        dense = torch.eye(9, dtype=torch.float64)+q@(k-torch.eye(2, dtype=torch.float64))@q.T
        expected = dense@(h.sqrt()*d)/h.sqrt()
        torch.testing.assert_close(out, expected, atol=1e-12, rtol=1e-12)
        change = h.sqrt()*(out-d)
        torch.testing.assert_close(change-q@(q.T@change), torch.zeros_like(change), atol=1e-12, rtol=0)
        self.assertLessEqual(float((h.sqrt()*out).norm()), float((h.sqrt()*d).norm())+1e-12)
        self.assertTrue(all(0 < e <= 1+1e-12 for e in info['filter_eigenvalues']))

    def test_enabled_calls_adam_formula_aux_no_grad_and_snapshot(self):
        a = self.host(); b = self.host(base_candidates()[3], base=True)
        trace = []; proposals = []
        handle = a.native.model.register_forward_pre_hook(lambda *args: trace.append(torch.is_grad_enabled()))
        def post(opt, args, kwargs):
            proposals.append((flatten(a.native.params).clone(), copy.deepcopy(opt.state_dict())))
        h2 = a.native.base.register_step_post_hook(post)
        recorded = []
        vjps = []
        native_grad = torch.autograd.grad
        def observed_grad(*args, **kwargs):
            values = native_grad(*args, **kwargs)
            vjps.append(flatten(values))
            return values
        def observed(delta, metric, acquisition, anatomy, strength):
            out, detail = filter_update(delta, metric, acquisition, anatomy, strength)
            recorded.append((delta.clone(), metric.clone(), acquisition.clone(), anatomy.clone(), out.clone()))
            return out, detail
        try:
            initial = flatten(a.native.params).clone()
            with patch('dpa_ctta.joint_sensitivity.method.filter_update', observed), patch('torch.autograd.grad', observed_grad):
                za, detail = a.step(self.x)
            self.assertEqual(detail['native'][0]['counts'], dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0))
            self.assertEqual(trace, [True, False, False, False, False, False, True, True, True, False])
            self.assertEqual(len(proposals), 1)
            self.assertTrue(equal(proposals[0][1], a.native.base.state_dict()))
            delta, metric, acquisition, anatomy, out = recorded[0]
            self.assertEqual(len(vjps), 4)
            self.assertTrue(torch.equal(anatomy, torch.stack(vjps[:2], 1)))
            shared = (vjps[0]+vjps[1])/math.sqrt(2)
            self.assertTrue(torch.equal(acquisition, torch.stack([g-shared for g in vjps[2:]], 1)))
            self.assertGreater(float(acquisition.norm()), 0.)
            self.assertGreater(float(anatomy.norm()), 0.)
            torch.testing.assert_close(proposals[0][0]-initial, delta, rtol=0, atol=0)
            torch.testing.assert_close(flatten(a.native.params), initial+out, rtol=0, atol=0)
            b.step(self.x)
            # Auxiliary VJPs must neither contaminate base loss gradients nor Adam moments/RNG.
            self.assertTrue(equal(a.native.base.state_dict(), b.native.base.state_dict()))
            self.assertTrue(equal(a.snapshot()['gradients'], b.snapshot()['gradients']))
            self.assertTrue(equal(a.snapshot()['native_rng'], b.snapshot()['native_rng']))
            saved = a.snapshot(); za, _ = a.step(self.x.roll(3, -1)); end = a.snapshot()
            a.restore(saved); zb, _ = a.step(self.x.roll(3, -1))
            self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(end, a.snapshot()))
            a.check_frozen(True)
        finally:
            handle.remove(); h2.remove(); a.close(); b.close()

    def test_zero_evidence_keeps_proposal_and_physical_calls(self):
        a = self.host(); b = self.host(base_candidates()[3], base=True)
        def empty_sensor(*args):
            sensor = Sensor(*args); sensor.weight.zero_(); return sensor
        try:
            with patch('dpa_ctta.joint_sensitivity.method.Sensor', empty_sensor):
                za, detail = a.step(self.x)
            sa = a.snapshot(); zb, _ = b.step(self.x); sb = b.snapshot()
            self.assertTrue(torch.equal(za, zb))
            self.assertEqual(detail['diagnostics']['joint']['rank'], 0)
            self.assertEqual(detail['native'][0]['counts']['backwards'], 5)
            for key in ('parameters', 'gradients', 'adam', 'grata', 'native_rng'):
                self.assertTrue(equal(sa[key], sb[key]), key)
        finally:
            a.close(); b.close()


if __name__ == '__main__': unittest.main()
