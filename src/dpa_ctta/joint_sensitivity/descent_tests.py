"""CPU-only KKT, frozen-disabled parity and native-call checks."""
import copy
import os
import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from .descent import candidates
from .method import Host, filter_update, flatten
from .shared_span import candidates as shared_candidates
from .profile import check_joint_update
from ..r19_model_only.method import equal
from ..r20_model_only_search.tests import Tiny
from ..r37_cw_orientation.method import Host as BaseHost, candidates as base_candidates


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = Tiny()
        cls.x = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)

    def host(self, config=None, base=False):
        return (BaseHost if base else Host)(None, config or candidates()[-1], 17011,
            'synthetic', device='cpu', model=copy.deepcopy(self.model))

    def test_active_closed_form_and_inactive_exact_shared(self):
        h = torch.ones(3, dtype=torch.float64)
        a = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64)
        b = torch.tensor([[0., 0.], [1., 0.], [0., 0.]], dtype=torch.float64)
        d = torch.tensor([0., -1., .3], dtype=torch.float64)
        g = torch.tensor([0., 1., 0.], dtype=torch.float64)
        old, _ = filter_update(d, h, a, b, shared_span=True)
        out, info = filter_update(d, h, a, b, shared_span=True, descent_gradient=g)
        self.assertGreater(float(g@old), float(g@d))
        torch.testing.assert_close(out, d, atol=1e-12, rtol=0)
        self.assertTrue(info['descent_constraint_active'])
        self.assertAlmostEqual(info['descent_filtered_dot'], info['descent_proposal_dot'])
        d = torch.tensor([-1., 1., .3], dtype=torch.float64)
        g = torch.tensor([1., 0., 0.], dtype=torch.float64)
        old, _ = filter_update(d, h, a, b, shared_span=True)
        out, info = filter_update(d, h, a, b, shared_span=True, descent_gradient=g)
        self.assertFalse(info['descent_constraint_active'])
        self.assertTrue(torch.equal(old, out))

    def test_noncommuting_dense_kkt_complement_and_gradient_scale(self):
        gen = torch.Generator().manual_seed(31)
        for i in range(6):
            d = torch.randn(9, dtype=torch.float64, generator=gen)
            h = torch.exp(torch.randn(9, dtype=torch.float64, generator=gen))
            a = torch.randn(9, 2, dtype=torch.float64, generator=gen)
            b = torch.randn(9, 2, dtype=torch.float64, generator=gen)
            g = -d
            x, y = a/h.sqrt()[:, None], b/h.sqrt()[:, None]
            q, _ = torch.linalg.qr(torch.cat((x/x.norm(), y/y.norm()), dim=1))
            aa, ss = (q.T@x)@(x.T@q)/x.square().sum(), (q.T@y)@(y.T@q)/y.square().sum()
            self.assertGreater(float((aa@ss-ss@aa).norm()), 1e-3)
            ev, basis = torch.linalg.eigh(aa+.01*torch.eye(4, dtype=torch.float64))
            r = (basis/ev.sqrt())@basis.T
            m = torch.eye(4, dtype=torch.float64)+r@ss@r
            va, c = q.T@(h.sqrt()*d), q.T@(g/h.sqrt())
            v0 = torch.linalg.solve(m, va)
            active = bool(c@(v0-va) > 0)
            if active:
                system = torch.zeros(5, 5, dtype=torch.float64)
                system[:4, :4] = m
                system[:4, 4] = c
                system[4, :4] = c
                solution = torch.linalg.solve(system, torch.cat((va, (c@va).reshape(1))))
                expected_v = solution[:4]
                self.assertGreater(float(solution[4]), 0)
            else:
                expected_v = v0
            expected = (h.sqrt()*d+q@(expected_v-va))/h.sqrt()
            out, info = filter_update(d, h, a, b, shared_span=True, descent_gradient=g)
            torch.testing.assert_close(out, expected, atol=1e-10, rtol=1e-10)
            self.assertEqual(info['descent_constraint_active'], active)
            self.assertLessEqual(float(g@(out-d)), 1e-10)
            change = h.sqrt()*(out-d)
            torch.testing.assert_close(change-q@(q.T@change), torch.zeros_like(change), atol=1e-10, rtol=0)
            scaled, _ = filter_update(d, h, a, b, shared_span=True, descent_gradient=g*1e-5)
            torch.testing.assert_close(scaled, out, atol=1e-10, rtol=1e-10)

    def test_non_descent_zero_missing_sensitivity_and_invalid_inputs(self):
        d = torch.tensor([1., 2., 3.], dtype=torch.float64)
        h = torch.ones_like(d)
        a = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
        b = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        for g in (d, d*0):
            out, info = filter_update(d, h, a, b, shared_span=True, descent_gradient=g)
            self.assertTrue(torch.equal(out, d))
            self.assertEqual(info['descent_fallback'], 'non_descent_proposal')
        for x, y, strength in ((a*0, b, 1.), (a, b*0, 1.), (a, b, 0.)):
            out, info = filter_update(d, h, x, y, strength, shared_span=True, descent_gradient=-d)
            self.assertTrue(torch.equal(out, d))
            self.assertEqual(info['rank'], 0)
        for g, shared in ((d[:2], True), (d*float('nan'), True), (d, False)):
            with self.assertRaises(ValueError): filter_update(d, h, a, b, shared_span=shared, descent_gradient=g)
        for config in (dict(candidates()[-1], joint_descent_constraint='yes'),
                       dict(candidates()[-1], joint_basis='acquisition')):
            with self.assertRaises(ValueError): self.host(config)
        self.assertEqual(candidates()[:-1], shared_candidates())

    def test_disabled_bitwise_frozen_r46_output_diagnostics_full_snapshot(self):
        repo = Path(__file__).resolve().parents[3]
        fixture = os.environ.get('FROZEN_SHARED_SOURCE')
        source = Path(fixture).read_text() if fixture else subprocess.check_output(['git', 'show',
            '66097c9f6ed3c37645ca315c009fbd3b9aaa836c:src/dpa_ctta/joint_sensitivity/method.py'], cwd=repo, text=True)
        frozen = types.ModuleType('dpa_ctta.joint_sensitivity._r46_reference')
        exec(compile(source, '<frozen-r46-source>', 'exec'), frozen.__dict__)
        config = dict(shared_candidates()[-1], joint_descent_constraint=False)
        current = self.host(config)
        old = frozen.Host(None, config, 17011, 'synthetic', device='cpu', model=copy.deepcopy(self.model))
        try:
            for x in (self.x, self.x.roll(7, -1)):
                za, da = current.step(x); sa = current.snapshot()
                zb, db = old.step(x); sb = old.snapshot()
                self.assertTrue(torch.equal(za, zb))
                self.assertTrue(equal(da, db))
                self.assertTrue(equal(sa, sb))
        finally:
            current.close(); old.close()

    def test_joint_disabled_bitwise_cw_lso(self):
        current = self.host(dict(candidates()[-1], joint_strength=0.))
        base = self.host(base_candidates()[3], base=True)
        try:
            for x in (self.x, self.x.roll(9, -1)):
                za, da = current.step(x); sa = current.snapshot()
                zb, db = base.step(x); sb = base.snapshot()
                self.assertTrue(torch.equal(za, zb))
                self.assertTrue(equal(da, db))
                self.assertTrue(equal(sa, sb))
                self.assertEqual(da['native'][0]['counts']['forwards'], 8)
                self.assertEqual(da['native'][0]['counts']['backwards'], 1)
        finally:
            current.close(); base.close()

    def test_profile_validates_loss_descent_instead_of_norm_contraction(self):
        diag = dict(rank=4, proposal_metric_norm=1., filtered_metric_norm=1.2,
                    descent_proposal_dot=-1., descent_actual_dot=-1.)
        check_joint_update(candidates()[-1], diag)
        with self.assertRaises(ValueError): check_joint_update(shared_candidates()[-1], diag)
        for actual in (-.9, float('nan')):
            with self.assertRaises(ValueError): check_joint_update(candidates()[-1], dict(diag, descent_actual_dot=actual))
        with self.assertRaises(KeyError): check_joint_update(candidates()[-1], dict(rank=0))

    def test_enabled_formula_counts_adam_gradients_rng_and_restore(self):
        host = self.host(); base = self.host(base_candidates()[3], base=True)
        initial = flatten(host.native.params)
        proposals, captured, vjps, forward_trace = [], [], [], []
        post = host.native.base.register_step_post_hook(lambda opt, args, kwargs:
            proposals.append((flatten(host.native.params), copy.deepcopy(opt.state_dict()),
                              flatten([p.grad for p in host.native.params]))))
        forward = host.native.model.register_forward_pre_hook(lambda *args: forward_trace.append(torch.is_grad_enabled()))
        original_grad = torch.autograd.grad
        def observed_grad(*args, **kwargs):
            vjps.append(True)
            return original_grad(*args, **kwargs)
        def observed_filter(*args, **kwargs):
            out, diag = filter_update(*args, **kwargs)
            captured.append((args, kwargs, out.clone(), diag))
            return out, diag
        try:
            with patch('dpa_ctta.joint_sensitivity.method.filter_update', observed_filter), patch('torch.autograd.grad', observed_grad):
                _, detail = host.step(self.x)
            self.assertEqual(len(proposals), 1); self.assertEqual(len(vjps), 4)
            self.assertEqual(forward_trace, [True, False, False, False, False, False, True, True, True, False])
            self.assertEqual(detail['native'][0]['counts'], dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0))
            args, kwargs, out, diag = captured[0]
            self.assertTrue(torch.equal(kwargs['descent_gradient'], proposals[0][2]))
            self.assertTrue(torch.equal(args[0], proposals[0][0]-initial))
            torch.testing.assert_close(flatten(host.native.params), initial+out, rtol=0, atol=0)
            self.assertTrue(equal(proposals[0][1], host.native.base.state_dict()))
            self.assertEqual(detail['native'][0]['adam_step'], 1)
            base.step(self.x)
            self.assertTrue(equal(base.snapshot()['gradients'], host.snapshot()['gradients']))
            self.assertTrue(equal(base.native.base.state_dict(), host.native.base.state_dict()))
            self.assertTrue(equal(base.snapshot()['native_rng'], host.snapshot()['native_rng']))
            actual = float(kwargs['descent_gradient'].double()@(flatten(host.native.params)-initial).double())
            self.assertEqual(diag['descent_actual_dot'], actual)
            scale = max(abs(diag['descent_proposal_dot']), 1e-10)
            self.assertLessEqual(actual-diag['descent_proposal_dot'], 1e-5*scale)
            saved = host.snapshot(); za, _ = host.step(self.x.roll(4, -1)); end = host.snapshot()
            host.restore(saved); zb, _ = host.step(self.x.roll(4, -1))
            self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(end, host.snapshot()))
            host.check_frozen(True)
        finally:
            forward.remove(); post.remove(); host.close(); base.close()


if __name__ == '__main__': unittest.main()
