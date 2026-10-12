"""Synthetic CPU checks for shared sensitivity span and frozen R45 behavior."""
import copy
import os
import subprocess
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .method import Host, Sensor, filter_update, flatten
from .shared_span import candidates
from ..r37_cw_orientation.method import Host as BaseHost, candidates as base_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2); torch.manual_seed(42)
        cls.model = Tiny()
        cls.x = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)

    def host(self, config=None, base=False):
        return (BaseHost if base else Host)(None, config or candidates()[-1], 17011,
                                           'synthetic', device='cpu', model=copy.deepcopy(self.model))

    def test_anatomy_outside_acquisition_closed_form(self):
        delta = torch.tensor([0., 1., 0.], dtype=torch.float64)
        h = torch.ones(3, dtype=torch.float64)
        a = torch.tensor([[1., 0.], [0., 0.], [0., 0.]], dtype=torch.float64)
        b = torch.tensor([[0., 0.], [1., 0.], [0., 0.]], dtype=torch.float64)
        old, _ = filter_update(delta, h, a, b)
        out, info = filter_update(delta, h, a, b, shared_span=True)
        self.assertTrue(torch.equal(old, delta))
        torch.testing.assert_close(out, delta/101, rtol=1e-12, atol=1e-12)
        self.assertEqual(info['rank'], 2)
        self.assertAlmostEqual(info['anatomy_capture_fraction'], 1.)

    def test_noncommuting_four_dimensional_dense_solution(self):
        gen = torch.Generator().manual_seed(19)
        d = torch.randn(8, dtype=torch.float64, generator=gen)
        h = torch.exp(torch.randn(8, dtype=torch.float64, generator=gen))
        a = torch.randn(8, 2, dtype=torch.float64, generator=gen)
        b = torch.randn(8, 2, dtype=torch.float64, generator=gen)
        out, info = filter_update(d, h, a, b, shared_span=True)
        x, y = a/h.sqrt()[:, None], b/h.sqrt()[:, None]
        q, _ = torch.linalg.qr(torch.cat((x/x.norm(), y/y.norm()), dim=1))
        aa, ss = (q.T@x)@(x.T@q)/x.square().sum(), (q.T@y)@(y.T@q)/y.square().sum()
        self.assertGreater(float((aa@ss-ss@aa).norm()), 1e-3)
        e, v = torch.linalg.eigh(aa+.01*torch.eye(4, dtype=torch.float64))
        r = (v/e.sqrt())@v.T
        k = torch.linalg.inv(torch.eye(4, dtype=torch.float64)+r@ss@r)
        expected = (torch.eye(8, dtype=torch.float64)+q@(k-torch.eye(4, dtype=torch.float64))@q.T)@(h.sqrt()*d)/h.sqrt()
        torch.testing.assert_close(out, expected, atol=1e-11, rtol=1e-11)
        change = h.sqrt()*(out-d)
        torch.testing.assert_close(change-q@(q.T@change), torch.zeros_like(change), atol=1e-11, rtol=0)
        self.assertEqual(info['rank'], 4)
        self.assertLessEqual(float((h.sqrt()*out).norm()), float((h.sqrt()*d).norm())+1e-11)
        self.assertTrue(all(0 < e <= 1+1e-11 for e in info['filter_eigenvalues']))

    def test_missing_sensitivity_disabled_and_scale_invariance(self):
        d = torch.tensor([1., 2., 3.], dtype=torch.float64); h = torch.ones_like(d)
        a = torch.tensor([[1., 0.], [0., 1.], [0., 0.]], dtype=torch.float64)
        b = torch.tensor([[0., 0.], [1., 0.], [0., 1.]], dtype=torch.float64)
        for x, y, strength in [(a, b, 0.), (a*0, b, 1.), (a, b*0, 1.)]:
            out, info = filter_update(d, h, x, y, strength, shared_span=True)
            self.assertTrue(torch.equal(out, d)); self.assertEqual(info['rank'], 0)
        out, _ = filter_update(d, h, a, b, shared_span=True)
        scaled, _ = filter_update(d, h, a*1e-5, b*1e5, shared_span=True)
        torch.testing.assert_close(out, scaled, rtol=1e-11, atol=1e-11)
        for strength in [-1., float('nan')]:
            with self.assertRaises(ValueError): filter_update(d, h, a, b, strength, shared_span=True)
        with self.assertRaises(ValueError): filter_update(d, h*0, a, b, shared_span=True)
        with self.assertRaises(ValueError): self.host(dict(candidates()[-1], joint_basis='other'))

    def test_disabled_exact_cw_lso_and_original_r45_exact(self):
        disabled = self.host(dict(candidates()[-1], joint_strength=0.))
        base = self.host(base_candidates()[3], base=True)
        repo = Path(__file__).resolve().parents[3]
        fixture = os.environ.get('FROZEN_JS_SOURCE')
        source = Path(fixture).read_text() if fixture else subprocess.check_output(['git', 'show', '15445741fe17a3fb12dc0397a745389b6c857746:src/dpa_ctta/joint_sensitivity/method.py'], cwd=repo, text=True)
        frozen = types.ModuleType('dpa_ctta.joint_sensitivity._r45_reference')
        exec(compile(source, '<frozen-r45-source>', 'exec'), frozen.__dict__)
        old = frozen.Host(None, frozen.candidates()[-1], 17011, 'synthetic', device='cpu', model=copy.deepcopy(self.model))
        current = self.host(frozen.candidates()[-1])
        try:
            for x in (self.x, self.x.roll(3, -1)):
                za, da = disabled.step(x); sa = disabled.snapshot()
                zb, db = base.step(x); sb = base.snapshot()
                self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(da, db)); self.assertTrue(equal(sa, sb))
                za, da = current.step(x); sa = current.snapshot()
                zb, db = old.step(x); sb = old.snapshot()
                self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(da, db)); self.assertTrue(equal(sa, sb))
        finally:
            for host in (disabled, base, old, current): host.close()

    def test_enabled_native_formula_calls_moments_and_restore(self):
        host = self.host(); base = self.host(base_candidates()[3], base=True)
        old = flatten(host.native.params).clone(); proposals = []; captured = []
        hook = host.native.base.register_step_post_hook(lambda opt, args, kwargs: proposals.append((flatten(host.native.params).clone(), copy.deepcopy(opt.state_dict()))))
        def observed(*args, **kwargs):
            out, diag = filter_update(*args, **kwargs); captured.append((out.clone(), kwargs)); return out, diag
        try:
            with patch('dpa_ctta.joint_sensitivity.method.filter_update', observed): _, detail = host.step(self.x)
            self.assertEqual(detail['native'][0]['counts'], dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0))
            self.assertEqual(len(proposals), 1); self.assertTrue(captured[0][1]['shared_span'])
            torch.testing.assert_close(flatten(host.native.params), old+captured[0][0], rtol=0, atol=0)
            self.assertTrue(equal(proposals[0][1], host.native.base.state_dict()))
            base.step(self.x)
            self.assertTrue(equal(base.snapshot()['gradients'], host.snapshot()['gradients']))
            self.assertTrue(equal(base.native.base.state_dict(), host.native.base.state_dict()))
            self.assertTrue(equal(base.snapshot()['native_rng'], host.snapshot()['native_rng']))
            saved = host.snapshot(); za, _ = host.step(self.x.roll(4, -1)); end = host.snapshot()
            host.restore(saved); zb, _ = host.step(self.x.roll(4, -1))
            self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(end, host.snapshot()))
            host.check_frozen(True)
        finally:
            hook.remove(); host.close(); base.close()

    def test_zero_evidence_exact_fallback_with_declared_calls(self):
        host = self.host(); base = self.host(base_candidates()[3], base=True)
        def empty_sensor(*args):
            sensor = Sensor(*args); sensor.weight.zero_(); return sensor
        try:
            with patch('dpa_ctta.joint_sensitivity.method.Sensor', empty_sensor): za, detail = host.step(self.x)
            a = host.snapshot(); zb, _ = base.step(self.x); b = base.snapshot()
            self.assertTrue(torch.equal(za, zb)); self.assertEqual(detail['diagnostics']['joint']['rank'], 0)
            self.assertEqual(detail['native'][0]['counts'], dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0))
            for key in ('parameters', 'gradients', 'adam', 'grata', 'native_rng'):
                self.assertTrue(equal(a[key], b[key]), key)
        finally:
            host.close(); base.close()


if __name__ == '__main__': unittest.main()
