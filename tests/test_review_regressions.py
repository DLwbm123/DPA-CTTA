import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

import torch
from common import make_online
from dpa_ctta.config import DPAConfig
from dpa_ctta.descriptors import FrozenDescriptor
from dpa_ctta.integrations.ctta_suite import verify_tracked_files, check_module_origins
from dpa_ctta.offline.oracle import optimize_oracle_state
from dpa_ctta.proximal import closed_form_update
from dpa_ctta.types import AtlasResult


class ReviewRegressionTests(unittest.TestCase):
    def test_projection_rank_orthogonality_rng_and_schema(self):
        for channels in (4, 32, 64):
            before = torch.get_rng_state().clone()
            descriptor = FrozenDescriptor('polyp', channels)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            p = descriptor.feature_projection
            self.assertEqual(int(torch.linalg.matrix_rank(p)), 8)
            self.assertTrue(torch.allclose(p @ p.T, torch.eye(8), atol=1e-6))
            saved = descriptor.state_dict()
            descriptor.load_state_dict(saved)
            saved.pop('descriptor_schema_version')
            with self.assertRaisesRegex(RuntimeError, 'schema'):
                descriptor.load_state_dict(saved, strict=False)
        for dim in (0, 9, 1.5, True):
            with self.assertRaises(ValueError):
                FrozenDescriptor('polyp', 4, projection_dim=dim)
        with self.assertRaises(ValueError):
            DPAConfig.from_dict({'task': 'polyp'})

    def test_proximal_rejects_invalid_inputs_without_broadcast(self):
        previous = torch.zeros(1, 16)
        precision, rhs, weights = torch.ones(1, 16), torch.ones(1, 16), torch.ones(1, 1)
        cases = [
            (previous + float('nan'), precision, rhs, weights, 1., .001),
            (previous, precision, rhs + float('nan'), weights, 1., .001),
            (previous, precision, rhs[:, :1], weights, 1., .001),
            (previous, precision, rhs.double(), weights, 1., .001),
            (previous, precision, rhs, weights * 2, 1., .001),
            (previous, precision, rhs, -weights, 1., .001),
            (previous, precision, rhs, weights + float('nan'), 1., .001),
            (previous, precision, rhs, weights, float('nan'), .001),
            (previous, precision, rhs, weights, 1., float('inf')),
            (previous, precision, rhs, torch.ones(2, 1), 1., .001),
            (previous, precision, rhs.to('meta'), weights, 1., .001),
        ]
        for old, p, b, w, lam, floor in cases:
            with self.subTest(shape=b.shape, lam=lam), self.assertRaises(ValueError):
                closed_form_update(old, AtlasResult(w, p, b), lam, floor)
        valid = closed_form_update(previous, AtlasResult(weights, precision, rhs), 1., .001)
        self.assertTrue(torch.equal(valid.state, torch.full_like(previous, .5)))

    def test_final_exception_or_nonfinite_does_not_commit_history(self):
        image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)
        for failure in ('exception', 'nan'):
            adapter = make_online()
            before = adapter.z_prev.clone()
            original = adapter.injected_model.forward
            count = 0
            def fail_last(*args, **kwargs):
                nonlocal count
                count += 1
                if count == len(adapter.atlas.anchors) + 2:
                    if failure == 'exception':
                        raise RuntimeError('final prediction fault')
                    return torch.full((1, 1, 16, 16), float('nan'))
                return original(*args, **kwargs)
            try:
                with patch.object(adapter.injected_model, 'forward', side_effect=fail_last):
                    with self.assertRaises((RuntimeError, FloatingPointError)):
                        adapter.step(image)
                self.assertTrue(torch.equal(before, adapter.z_prev))
                result = adapter.step(image)
                self.assertTrue(torch.isfinite(result.logits).all())
                self.assertFalse(torch.equal(before, adapter.z_prev))
            finally:
                adapter.injected_model.close()

    def test_nan_center_rejected_without_state_pollution(self):
        adapter = make_online()
        before = adapter.z_prev.clone()
        try:
            with torch.no_grad():
                adapter.atlas.anchors[0].state_center[0] = float('nan')
            with self.assertRaises(ValueError):
                adapter.step(torch.rand(1, 3, 16, 16))
            self.assertTrue(torch.equal(before, adapter.z_prev))
        finally:
            adapter.injected_model.close()

    def test_oracle_preserves_all_caller_gradients_and_basis_weights(self):
        adapter = make_online()
        model = adapter.injected_model
        try:
            params = list(model.parameters())
            for i, parameter in enumerate(params):
                parameter.grad = torch.full_like(parameter, .125) if i % 2 else None
            grads = [None if p.grad is None else p.grad.clone() for p in params]
            weights = [p.detach().clone() for p in params]
            image = torch.rand(1, 3, 16, 16)
            mask = torch.zeros(1, 1, 16, 16); mask[:, :, 4:12, 4:12] = 1
            states = [optimize_oracle_state(model, image, mask, steps=2) for _ in range(2)]
            self.assertTrue(torch.equal(*states))
            self.assertFalse(states[0].requires_grad)
            for p, grad, weight in zip(params, grads, weights):
                self.assertTrue(torch.equal(p, weight))
                if grad is None:
                    self.assertIsNone(p.grad)
                else:
                    self.assertTrue(torch.equal(p.grad, grad))
        finally:
            model.close()

    def test_dependency_dirty_bytes_rejected_even_with_index_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'module.py'; path.write_text('VALUE = 1\n')
            def git(*args):
                return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True)
            git('init', '-q'); git('add', 'module.py'); git('commit', '-qm', 'Synthetic dependency fixture')
            verify_tracked_files(root, ['module.py'])
            git('update-index', '--assume-unchanged', 'module.py')
            path.write_text('VALUE = 2\n')
            with self.assertRaisesRegex(RuntimeError, 'dirty pinned'):
                verify_tracked_files(root, ['module.py'])

    def test_foreign_cache_rejected_and_not_evicted(self):
        module = ModuleType('ctta_suite.foreign_test')
        module.__file__ = '/foreign/module.py'
        with patch.dict(sys.modules, {'ctta_suite.foreign_test': module}):
            with self.assertRaisesRegex(RuntimeError, 'foreign module cache'):
                check_module_origins(('ctta_suite',), Path('/expected'))
            self.assertIs(sys.modules['ctta_suite.foreign_test'], module)

    def test_deprecated_audit_cannot_generate_success(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, 'scripts/pretraining_audit.py'], cwd=root, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('DEPRECATED_AUDIT_ENTRYPOINT', result.stderr)
