"""CPU committed-rounding counterexample, fallback and durable failure evidence."""
import copy
import io
import json
import os
import tempfile
import time
import types
import unittest
from collections import Counter
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import torch

from .method import commit_descent, descent_tolerance, filter_update
from .profile import check_joint_update
from ..r20_model_only_search import runtime
from ..r8_ba.journal import TargetJournal
from ..r47_shared_descent.method import Host, candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal
from .method import flatten


class Tests(unittest.TestCase):
    @staticmethod
    def fixture(seed=0, dtype=torch.float32):
        gen = torch.Generator().manual_seed(seed)
        old = torch.ones(128, dtype=dtype)
        native = old+torch.randn(128, generator=gen, dtype=dtype)*1e-4
        delta = native-old
        gradient = -delta
        metric = torch.exp(torch.randn(128, generator=gen, dtype=dtype))
        acquisition = torch.randn(128, 2, generator=gen, dtype=dtype)
        anatomy = torch.randn(128, 2, generator=gen, dtype=dtype)
        filtered, diagnostic = filter_update(delta, metric, acquisition, anatomy,
                                             shared_span=True, descent_gradient=gradient)
        return old, native, gradient, filtered, diagnostic

    def test_reproduce_boundary_rounding_and_exact_native_repair(self):
        old, native, gradient, filtered, diagnostic = self.fixture()
        legacy = dict(diagnostic, descent_actual_dot=float(gradient.double()@((old+filtered)-old).double()))
        self.assertTrue(diagnostic['descent_constraint_active'])
        self.assertLess(abs(legacy['descent_filtered_dot']-legacy['descent_proposal_dot']), 1e-12)
        with self.assertRaises(ValueError):
            check_joint_update({'joint_descent_constraint': True}, legacy)
        parameter = torch.nn.Parameter(native.clone())
        commit_descent([parameter], [old], filtered, gradient, diagnostic)
        self.assertTrue(diagnostic['descent_commit_repaired'])
        self.assertEqual(diagnostic['descent_fallback'], 'native_adam_after_commit_violation')
        self.assertTrue(torch.equal(parameter, native))
        self.assertEqual(diagnostic['descent_actual_dot'], diagnostic['descent_proposal_dot'])
        self.assertGreater(diagnostic['descent_candidate_committed_gap'], diagnostic['descent_tolerance'])
        check_joint_update({'joint_descent_constraint': True}, diagnostic)

    def test_quantized_proposals_over_both_dtypes_and_scales(self):
        repairs = 0
        for dtype in (torch.float32, torch.float64):
            for seed in range(16):
                for scale in (1., 1e-8):
                    old, native, gradient, filtered, diagnostic = self.fixture(seed, dtype)
                    # Derivative scale must not weaken the fixed absolute floor.
                    gradient = gradient*scale
                    parameter = torch.nn.Parameter(native.clone())
                    expected = old+filtered
                    commit_descent([parameter], [old], filtered, gradient, diagnostic)
                    check_joint_update({'joint_descent_constraint': True}, diagnostic)
                    target = native if diagnostic['descent_commit_repaired'] else expected
                    self.assertTrue(torch.equal(parameter, target))
                    repairs += diagnostic['descent_commit_repaired']
        self.assertGreater(repairs, 0)

    def test_feasible_inactive_and_rank_zero_remain_exact(self):
        old = torch.tensor([1., 2., 3.])
        native = old-torch.tensor([.125, .125, .125])
        gradient = torch.ones(3)
        for rank, filtered in ((4, (native-old)*2), (0, torch.ones(3))):
            parameter = torch.nn.Parameter(native.clone())
            diagnostic = dict(rank=rank, descent_constraint_active=False)
            commit_descent([parameter], [old], filtered, gradient, diagnostic)
            self.assertFalse(diagnostic['descent_commit_repaired'])
            self.assertTrue(torch.equal(parameter, old+filtered if rank else native))
            check_joint_update({'joint_descent_constraint': True}, diagnostic)

    def test_overflowing_candidate_falls_back_and_json_stays_finite(self):
        old = torch.tensor([2e38])
        native = torch.nextafter(old, torch.zeros_like(old))
        parameter = torch.nn.Parameter(native.clone())
        diagnostic = dict(rank=1)
        commit_descent([parameter], [old], torch.tensor([2e38]), torch.ones(1), diagnostic)
        self.assertTrue(torch.equal(parameter, native))
        self.assertTrue(diagnostic['descent_commit_repaired'])
        self.assertEqual(diagnostic['descent_candidate_actual_dot'], 'inf')
        json.dumps(diagnostic, allow_nan=False)

    def test_original_tolerance_and_exception_derivatives(self):
        for proposal in (-1., -1e-12):
            tolerance = 1e-5*max(abs(proposal), 1e-10)
            self.assertEqual(descent_tolerance(proposal), tolerance)
            diagnostic = dict(rank=4, descent_proposal_dot=proposal, descent_filtered_dot=proposal,
                              descent_actual_dot=proposal+2*tolerance)
            with self.assertRaises(ValueError) as caught:
                check_joint_update({'joint_descent_constraint': True}, diagnostic, 16)
            evidence = caught.exception.numerical_diagnostic
            self.assertEqual(evidence['visit'], 16)
            self.assertEqual(evidence['descent_tolerance'], tolerance)
            self.assertEqual(evidence['descent_actual_dot'], diagnostic['descent_actual_dot'])
            json.dumps(evidence, allow_nan=False)
        for actual in (float('nan'), float('inf')):
            with self.assertRaises(ValueError) as caught:
                check_joint_update({'joint_descent_constraint': True},
                                   dict(rank=0, descent_proposal_dot=-1., descent_actual_dot=actual), 42)
            json.dumps(caught.exception.numerical_diagnostic, allow_nan=False)

    def test_host_repair_keeps_adam_calls_rng_and_snapshot_replay(self):
        torch.manual_seed(42)
        host = Host(None, candidates()[-1], 17011, 'synthetic', device='cpu', model=Tiny())
        pixels = torch.linspace(.01, .99, 3*512*512).reshape(1, 3, 512, 512)
        proposals = []
        hook = host.native.base.register_step_post_hook(lambda optimizer, args, kwargs:
            proposals.append((flatten(host.native.params), copy.deepcopy(optimizer.state_dict()))))
        original = filter_update
        def infeasible(*args, **kwargs):
            result, diagnostic = original(*args, **kwargs)
            self.assertLess(diagnostic['descent_proposal_dot'], 0)
            diagnostic['rank'] = 4
            # Force a commit violation to exercise repair, not merely the KKT path.
            return -args[0], diagnostic
        try:
            with patch('dpa_ctta.joint_sensitivity.method.filter_update', infeasible):
                _, detail = host.step(pixels)
                self.assertTrue(detail['diagnostics']['joint']['descent_commit_repaired'])
                self.assertTrue(torch.equal(flatten(host.native.params), proposals[-1][0]))
                self.assertTrue(equal(host.native.base.state_dict(), proposals[-1][1]))
                self.assertEqual(detail['native'][0]['counts'],
                                 dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0))
                saved = host.snapshot()
                za, da = host.step(pixels.roll(4, -1)); final = host.snapshot()
                host.restore(saved)
                zb, db = host.step(pixels.roll(4, -1))
                self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(da, db))
                self.assertTrue(equal(final, host.snapshot()))
                host.check_frozen(True)
        finally:
            hook.remove(); host.close()

    @staticmethod
    def numerical_error():
        try:
            check_joint_update({'joint_descent_constraint': True},
                               dict(rank=4, descent_proposal_dot=-1., descent_filtered_dot=-1.,
                                    descent_actual_dot=-.9), 16)
        except ValueError as error:
            return error
        raise AssertionError('failure must stop')

    def test_failure_journal_persists_derivatives_without_content(self):
        error = self.numerical_error()
        with tempfile.TemporaryDirectory() as directory:
            journal = TargetJournal(directory, Mock(), 'synthetic', '0'*64)
            journal.record_failed_call(16, {'optimizer_steps': 16}, error)
            row = json.loads(journal.physical.read_text())
            self.assertEqual(row['visit'], 16)
            self.assertEqual(row['numerical_diagnostic'], error.numerical_diagnostic)
            self.assertNotIn('job_id', row)

    def test_failed_step_uses_pre_call_index_even_if_host_incremented(self):
        error = self.numerical_error()
        host = Mock(visits=15, teacher=None, f0=None)
        def step(_):
            host.visits += 1
            raise error
        host.step.side_effect = step
        journal = Mock()
        guard = Mock(); guard.meter.cost = {'optimizer_steps': 16}
        job = dict(id='synthetic', order=0, stream='online', candidate={}, seed=0)
        config = dict(output_root='unused', target_root='unused', native_seed=0,
                      manifests={'ONLINE': {0: {'sha256': 'synthetic'}}})
        with patch.object(runtime, 'read', return_value=[{}]*16), patch.object(runtime, 'rows_sha', return_value='synthetic'), \
             patch.object(runtime, 'Host', return_value=host), patch.object(runtime, 'weights'), \
             patch.object(runtime, 'attach'), patch.object(runtime, 'TargetJournal', return_value=journal), \
             patch.object(runtime, 'TargetReader'), patch.object(runtime, 'image'):
            with self.assertRaises(ValueError):
                runtime.online(config, guard, {}, job)
        self.assertEqual(journal.record_failed_call.call_args.args[0], 16)
        host.close.assert_called_once()

    def test_worker_receipt_serializes_the_failed_derivative(self):
        error = self.numerical_error()
        with tempfile.TemporaryDirectory() as directory:
            config = dict(output_root=directory, code_sha='synthetic')
            guard = types.SimpleNamespace(deadline=time.time()+60, extra=Counter())
            environment = dict(RUN_PHASE='score_synthetic', RUN_STARTED=str(time.time()),
                               RUN_JOB='{}', CUDA_VISIBLE_DEVICES='')
            with patch.dict(os.environ, environment), patch.object(runtime, 'config', return_value=config), \
                 patch.object(runtime, 'Guard', return_value=guard), patch.object(runtime.threading, 'Timer'), \
                 patch.object(runtime.torch, 'set_num_threads'), patch.object(runtime.torch, 'set_num_interop_threads'), \
                 patch('dpa_ctta.r20_model_only_search.score.score_job', side_effect=error), redirect_stderr(io.StringIO()):
                row = runtime.worker()
            saved = json.loads((Path(directory)/'attempts/score_synthetic.0.json').read_text())
            self.assertEqual(row['status'], 'FAILED_NUMERICAL')
            self.assertEqual(saved['failure']['numerical_diagnostic'], error.numerical_diagnostic)
            self.assertEqual(saved['cost']['gpu_seconds'], 0)


if __name__ == '__main__':
    unittest.main()
