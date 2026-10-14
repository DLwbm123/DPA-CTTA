"""Generated CPU fixtures; no private images, checkpoint, GPU or efficacy run."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .factorial import Host, candidates
from .method import Host as HistoricalJointHost, Sensor, filter_update, flatten
from ..host_diagnostic import rng
from ..r19_model_only.method import equal
from ..r20_model_only_search.tests import Tiny
from ..r20_model_only_search.method import weights_for
from ..r36_incremental_modules.method import local_structure_loss
from ..r37_cw_orientation.method import Host as BaseHost, candidates as base_candidates


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        cls.model = Tiny()
        cls.x = torch.linspace(.01, .99, 3 * 512 * 512).reshape(1, 3, 512, 512)

    def host(self, config, factory=Host):
        return factory(None, config, 17011, 'synthetic', device='cpu',
                       model=copy.deepcopy(self.model))

    def test_exact_five_configs_strong_control_and_factory_isolation(self):
        a = candidates()
        self.assertEqual([c['id'] for c in a],
                         ['C', 'C_LSO', 'C_JS_SHARED', 'C_LSO_JS_SHARED', 'CW_LSO'])
        self.assertEqual(a[0], base_candidates()[0])
        self.assertEqual(a[-1], base_candidates()[3])
        self.assertTrue(all(c['family'] == 'control' and c['params'] == {} for c in a[:4]))
        self.assertEqual(a[1]['module_strength'], .1)
        self.assertEqual(a[2]['joint_strength'], 1.)
        a[-1]['params']['boundary_boost'] = 99
        self.assertEqual(candidates()[-1], base_candidates()[3])

    def test_reject_unregistered_changes_before_initialization(self):
        j = candidates()[2]
        variants = [dict(j, id='other'), dict(j, joint_strength=.5),
                    dict(j, joint_strength=float('nan')), dict(j, joint_strength=True),
                    dict(j, joint_basis='acquisition'), dict(j, joint_descent_constraint=True),
                    dict(j, final_lr_multiplier=2.), dict(j, components={'W': {}}),
                    dict(candidates()[1], module_strength=.2), dict(candidates()[-1], joint_strength=1.)]
        saved = rng()
        for config in variants:
            with self.subTest(config=config), self.assertRaises(ValueError):
                self.host(config)
        self.assertTrue(equal(saved, rng()))

    def test_c_and_strong_control_full_snapshot_bitwise(self):
        for config in (candidates()[0], candidates()[-1]):
            a, b = self.host(config), self.host(config, BaseHost)
            try:
                for x in (self.x, self.x.roll(5, -1)):
                    za, da = a.step(x); sa = a.snapshot()
                    zb, db = b.step(x); sb = b.snapshot()
                    self.assertTrue(torch.equal(za, zb))
                    self.assertTrue(equal(da, db))
                    self.assertTrue(equal(sa, sb))
                a.check_frozen(True); b.check_frozen(True)
            finally:
                a.close(); b.close()

    def test_disabled_j_exact_c_and_c_lso_including_rng_and_calls(self):
        for base, joint in ((candidates()[0], candidates()[2]),
                            (candidates()[1], candidates()[3])):
            a, b = self.host(dict(joint, joint_strength=0.)), self.host(base)
            try:
                for x in (self.x, self.x.roll(7, -1)):
                    za, da = a.step(x); sa = a.snapshot()
                    zb, db = b.step(x); sb = b.snapshot()
                    self.assertTrue(torch.equal(za, zb))
                    self.assertTrue(equal(da, db))
                    self.assertTrue(equal(sa, sb))
                    self.assertEqual(da['native'][0]['counts'],
                                     dict(forwards=8, backwards=1, base_adam=1, perturb=0, restore=0))
                a.check_frozen(True); b.check_frozen(True)
            finally:
                a.close(); b.close()

    def test_c_lso_reference_loss_unweighted_base_and_total_gradient(self):
        host = self.host(candidates()[1])
        gen = torch.Generator().manual_seed(19)
        weak_logits = torch.randn(6, 1, 2, 32, 32, generator=gen)
        host.weak = list(weak_logits)
        logits = torch.randn(1, 2, 32, 32, generator=gen, requires_grad=True)
        pseudo = weak_logits.sigmoid().mean(0).detach()
        reference = logits.detach().clone().requires_grad_(True)
        try:
            loss = host._criterion(logits, pseudo)
            weight = weights_for(weak_logits.sigmoid(), .1, 2.)
            orientation, _ = local_structure_loss(reference, pseudo, weight)
            base = F.binary_cross_entropy_with_logits(reference, pseudo)
            expected = base + .1 * orientation
            self.assertTrue(torch.equal(loss, expected))
            ga = torch.autograd.grad(loss, logits)[0]
            gb = torch.autograd.grad(expected, reference)[0]
            self.assertTrue(torch.equal(ga, gb))
            weighted = (F.binary_cross_entropy_with_logits(reference, pseudo, reduction='none') * weight).mean()
            self.assertNotEqual(float(base), float(weighted))
            self.assertEqual(host.diag['pseudo_loss'], float(base.detach()))
            self.assertFalse(weight.requires_grad)
            self.assertNotIn('W', host.modules)
            self.assertEqual(host.diag['module_coefficient'], .1)
        finally:
            host.close()

    def test_all_conditions_physical_calls_and_global_rng(self):
        final_rng = []
        for config in candidates():
            host = self.host(config)
            trace, backwards, steps, vjps = [], [], [], []
            forward = host.native.model.register_forward_pre_hook(
                lambda *args: trace.append(torch.is_grad_enabled()))
            reverse = host.native.model.head.register_full_backward_hook(
                lambda *args: backwards.append(True))
            adam = host.native.base.register_step_post_hook(lambda *args: steps.append(True))
            original = torch.autograd.grad
            def observed(*args, **kwargs):
                vjps.append(True)
                return original(*args, **kwargs)
            try:
                with patch('torch.autograd.grad', observed):
                    _, detail = host.step(self.x)
                joint = bool(config.get('joint_strength'))
                self.assertEqual(trace, ([True, False, False, False, False, False, True, True, True, False]
                                         if joint else [False] * 6 + [True, False]))
                self.assertEqual(len(backwards), 5 if joint else 1)
                self.assertEqual(len(steps), 1)
                self.assertEqual(len(vjps), 4 if joint else 0)
                self.assertEqual(detail['native'][0]['counts'],
                                 dict(forwards=10 if joint else 8, backwards=5 if joint else 1,
                                      base_adam=1, perturb=0, restore=0))
                self.assertEqual(detail['native'][0]['adam_step'], 1)
                final_rng.append(rng())
                self.assertTrue(equal(final_rng[-1], host.native.rng))
                saved = host.snapshot()
                self.assertTrue(equal(saved['rng'], saved['native_rng']))
            finally:
                forward.remove(); reverse.remove(); adam.remove(); host.close()
        self.assertTrue(all(equal(final_rng[0], value) for value in final_rng[1:]))

    def test_enabled_shared_formula_base_gradients_adam_and_rng(self):
        for base, joint in ((candidates()[0], candidates()[2]),
                            (candidates()[1], candidates()[3])):
            host, plain = self.host(joint), self.host(base)
            initial = flatten(host.native.params)
            proposals, captured = [], []
            post = host.native.base.register_step_post_hook(lambda opt, *args:
                proposals.append((flatten(host.native.params), copy.deepcopy(opt.state_dict()))))
            def observed(*args, **kwargs):
                output, diagnostic = filter_update(*args, **kwargs)
                captured.append((args, kwargs, output.clone()))
                return output, diagnostic
            try:
                with patch('dpa_ctta.joint_sensitivity.method.filter_update', observed):
                    _, detail = host.step(self.x)
                self.assertEqual(len(proposals), 1)
                self.assertEqual(len(captured), 1)
                args, kwargs, out = captured[0]
                self.assertEqual(kwargs, {'shared_span': True})
                self.assertTrue(torch.equal(args[0], proposals[0][0] - initial))
                self.assertTrue(torch.equal(flatten(host.native.params), initial + out))
                self.assertGreater(detail['diagnostics']['joint']['rank'], 0)
                plain.step(self.x)
                self.assertTrue(equal(plain.native.base.state_dict(), host.native.base.state_dict()))
                self.assertTrue(equal(proposals[0][1], host.native.base.state_dict()))
                self.assertTrue(equal(plain.snapshot()['gradients'], host.snapshot()['gradients']))
                self.assertTrue(equal(plain.snapshot()['native_rng'], host.snapshot()['native_rng']))
                host.check_frozen(True)
            finally:
                post.remove(); host.close(); plain.close()

    def test_snapshot_restore_exact_replay_all_five_conditions(self):
        for config in candidates():
            host = self.host(config)
            try:
                host.step(self.x)
                saved = host.snapshot()
                za, da = host.step(self.x.roll(3, -1)); end = host.snapshot()
                host.restore(saved)
                zb, db = host.step(self.x.roll(3, -1))
                self.assertTrue(torch.equal(za, zb))
                self.assertTrue(equal(da, db))
                self.assertTrue(equal(end, host.snapshot()))
                host.check_frozen(True)
            finally:
                host.close()

    def test_zero_evidence_exact_native_update_for_both_backgrounds(self):
        def empty(*args):
            sensor = Sensor(*args); sensor.weight.zero_(); return sensor
        for base, joint in ((candidates()[0], candidates()[2]),
                            (candidates()[1], candidates()[3])):
            host, plain = self.host(joint), self.host(base)
            try:
                with patch('dpa_ctta.joint_sensitivity.method.Sensor', empty):
                    za, detail = host.step(self.x)
                a = host.snapshot(); zb, _ = plain.step(self.x); b = plain.snapshot()
                self.assertTrue(torch.equal(za, zb))
                self.assertEqual(detail['diagnostics']['joint']['rank'], 0)
                for key in ('parameters', 'buffers', 'gradients', 'adam', 'grata', 'native_rng'):
                    self.assertTrue(equal(a[key], b[key]), key)
            finally:
                host.close(); plain.close()

    def test_historical_joint_guard_remains_cw_lso_only(self):
        with self.assertRaisesRegex(ValueError, 'unchanged CW_LSO'):
            self.host(candidates()[2], HistoricalJointHost)


if __name__ == '__main__':
    unittest.main()
