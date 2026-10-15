"""Generated CPU fixtures, independent dense oracle and physical-call checks."""
import copy
import unittest
from unittest.mock import patch

import torch

from .acquisition_weighted import Host, candidates
from .factorial import Host as FactorialHost, candidates as factorial_candidates
from .method import Sensor, filter_update, flatten
from ..host_diagnostic import rng
from ..r19_model_only.method import equal
from ..r20_model_only_search.tests import Tiny


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = Tiny()
        cls.x = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)

    def host(self, config=None, factory=Host):
        return factory(None, config or candidates()[-1], 17011, 'synthetic', device='cpu',
                       model=copy.deepcopy(self.model))

    def test_noncommuting_full_space_objective_and_complement(self):
        gen = torch.Generator().manual_seed(19)
        for dtype in (torch.float32, torch.float64):
            d = torch.randn(9, generator=gen, dtype=dtype)
            h = torch.exp(torch.randn(9, generator=gen, dtype=dtype))
            a = torch.randn(9, 2, generator=gen, dtype=dtype)
            b = torch.randn(9, 2, generator=gen, dtype=dtype)
            out, info = filter_update(d, h, a, b, shared_span=True, acquisition_weighted=True)
            root = h.double().sqrt()
            x, y = a.double()/root[:, None], b.double()/root[:, None]
            aa, ss = x@x.T/x.square().sum(), y@y.T/y.square().sum()
            self.assertGreater(float((aa@ss-ss@aa).norm()), 1e-3)
            eye = torch.eye(9, dtype=torch.float64)
            e, v = torch.linalg.eigh(aa+.01*eye)
            r = (v*e.sqrt())@v.T
            penalty = r@ss@r
            expected = torch.linalg.solve(eye+penalty, root*d.double())
            tolerance = 2e-7 if dtype == torch.float32 else 1e-11
            torch.testing.assert_close(root*out.double(), expected, rtol=tolerance, atol=tolerance)
            # Stationarity of 1/2||u-w||^2 + 1/2 u^T penalty u.
            torch.testing.assert_close((eye+penalty)@(root*out.double()), root*d.double(),
                                       rtol=tolerance, atol=tolerance)
            q, _ = torch.linalg.qr(torch.cat((x, y), 1))
            change = root*(out.double()-d.double())
            torch.testing.assert_close(change-q@(q.T@change), torch.zeros_like(change),
                                       rtol=0, atol=tolerance)
            self.assertEqual(info['rank'], 4)
            self.assertLessEqual(float((root*out.double()).norm()), float((root*d.double()).norm())+tolerance)
            self.assertTrue(all(1/2.01-1e-10 <= value <= 1+1e-10 for value in info['filter_eigenvalues']))

    def test_acquisition_weak_structural_direction_is_retained(self):
        d = torch.tensor([0., 1., 0.], dtype=torch.float64)
        a = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64)
        b = a.roll(1, 0)
        old, _ = filter_update(d, torch.ones_like(d), a, b, shared_span=True)
        new, _ = filter_update(d, torch.ones_like(d), a, b, shared_span=True, acquisition_weighted=True)
        torch.testing.assert_close(old, d/101, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(new, d/1.01, rtol=1e-12, atol=1e-12)

    def test_disabled_missing_rank_deficient_and_scale_limits(self):
        d = torch.tensor([1., 2., 3.], dtype=torch.float64)
        h = torch.tensor([1., 2., 4.], dtype=torch.float64)
        a = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64)
        b = a.clone()
        for x, y, strength in ((a, b, 0.), (a*0, b, 1.), (a, b*0, 1.)):
            out, info = filter_update(d, h, x, y, strength, shared_span=True, acquisition_weighted=True)
            self.assertTrue(torch.equal(out, d))
            self.assertEqual(info['rank'], 0)
        out, info = filter_update(d, h, a, b, shared_span=True, acquisition_weighted=True)
        self.assertEqual(info['rank'], 1)
        torch.testing.assert_close(out, torch.tensor([1/2.01, 2., 3.], dtype=d.dtype))
        scaled, _ = filter_update(d, h, a*1e-5, b*1e5, shared_span=True, acquisition_weighted=True)
        torch.testing.assert_close(out, scaled, rtol=1e-12, atol=1e-12)

    def test_invalid_modes_and_config_do_not_consume_rng(self):
        d = torch.ones(3); a = torch.ones(3, 2)
        for options in ({}, {'shared_span': True, 'descent_gradient': -d}):
            with self.assertRaises(ValueError):
                filter_update(d, d, a, a, acquisition_weighted=True, **options)
        saved = rng()
        for change in ({'joint_strength': .5}, {'joint_strength': True}, {'joint_basis': 'acquisition'},
                       {'joint_filter': 'other'}, {'joint_descent_constraint': True}):
            with self.assertRaises(ValueError): self.host(dict(candidates()[-1], **change))
        self.assertTrue(equal(saved, rng()))

    def test_disabled_full_c_parity_two_arrivals(self):
        weighted = self.host(dict(candidates()[-1], joint_strength=0.))
        plain = self.host(factorial_candidates()[0], FactorialHost)
        try:
            for image in (self.x, self.x.roll(7, -1)):
                za, da = weighted.step(image); sa = weighted.snapshot()
                zb, db = plain.step(image); sb = plain.snapshot()
                self.assertTrue(torch.equal(za, zb))
                self.assertTrue(equal(da, db))
                self.assertTrue(equal(sa, sb))
                self.assertEqual(da['native'][0]['counts'],
                                 dict(forwards=8, backwards=1, base_adam=1, perturb=0, restore=0))
        finally:
            weighted.close(); plain.close()

    def test_enabled_calls_rng_actual_write_and_adam(self):
        host, plain = self.host(), self.host(factorial_candidates()[0], FactorialHost)
        initial = flatten(host.native.params)
        forwards, backwards, steps, vjps, captured = [], [], [], [], []
        hooks = [host.native.model.register_forward_pre_hook(lambda *a: forwards.append(torch.is_grad_enabled())),
                 host.native.model.head.register_full_backward_hook(lambda *a: backwards.append(True)),
                 host.native.base.register_step_post_hook(lambda *a: steps.append(flatten(host.native.params)))]
        original = torch.autograd.grad
        def grad(*args, **kwargs):
            vjps.append(True); return original(*args, **kwargs)
        def observed(*args, **kwargs):
            result = filter_update(*args, **kwargs)
            captured.append((args, kwargs, result[0])); return result
        try:
            with patch('torch.autograd.grad', grad), patch('dpa_ctta.joint_sensitivity.method.filter_update', observed):
                _, detail = host.step(self.x)
            self.assertEqual(forwards, [True]+[False]*5+[True]*3+[False])
            self.assertEqual((len(backwards), len(steps), len(vjps), len(captured)), (5, 1, 4, 1))
            self.assertEqual(detail['native'][0]['counts'],
                             dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0))
            args, kwargs, filtered = captured[0]
            self.assertEqual(kwargs, dict(shared_span=True, acquisition_weighted=True))
            self.assertTrue(torch.equal(args[0], steps[0]-initial))
            self.assertTrue(torch.equal(flatten(host.native.params), initial+filtered))
            gradient = flatten([p.grad for p in host.native.params]).double()
            diag = detail['diagnostics']['joint']
            self.assertEqual(diag['loss_proposal_dot'], float(gradient@args[0].double()))
            self.assertEqual(diag['loss_actual_dot'], float(gradient@(flatten(host.native.params)-initial).double()))
            self.assertEqual(diag['loss_dot_change'], diag['loss_actual_dot']-diag['loss_proposal_dot'])
            plain.step(self.x)
            self.assertTrue(equal(host.native.base.state_dict(), plain.native.base.state_dict()))
            for key in ('gradients', 'buffers', 'native_rng', 'rng'):
                self.assertTrue(equal(host.snapshot()[key], plain.snapshot()[key]), key)
            host.check_frozen(True)
        finally:
            for hook in hooks: hook.remove()
            host.close(); plain.close()

    def test_snapshot_replay(self):
        host = self.host()
        try:
            host.step(self.x); saved = host.snapshot()
            za, da = host.step(self.x.roll(3, -1)); end = host.snapshot()
            host.restore(saved)
            zb, db = host.step(self.x.roll(3, -1))
            self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(da, db))
            self.assertTrue(equal(end, host.snapshot()))
        finally: host.close()

    def test_zero_evidence_exact_native_fallback(self):
        host, plain = self.host(), self.host(factorial_candidates()[0], FactorialHost)
        def empty(*args):
            sensor = Sensor(*args); sensor.weight.zero_(); return sensor
        try:
            with patch('dpa_ctta.joint_sensitivity.method.Sensor', empty): za, da = host.step(self.x)
            a = host.snapshot(); zb, _ = plain.step(self.x); b = plain.snapshot()
            self.assertTrue(torch.equal(za, zb)); self.assertEqual(da['diagnostics']['joint']['rank'], 0)
            for key in ('parameters', 'buffers', 'gradients', 'adam', 'native_rng'):
                self.assertTrue(equal(a[key], b[key]), key)
        finally: host.close(); plain.close()


if __name__ == '__main__': unittest.main()
