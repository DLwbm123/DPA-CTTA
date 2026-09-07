import ast
import inspect
import os
from pathlib import Path
import subprocess
import sys
import unittest

import torch

from dpa_ctta.online import DPAOnlineAdapter
from dpa_ctta.proximal import closed_form_update
from common import clone_online, make_online


ROOT = Path(__file__).resolve().parents[1]


class OnlineContractTests(unittest.TestCase):
    def setUp(self):
        self.adapter = make_online()
        self.image = torch.linspace(0, 1, 3 * 16 * 16).reshape(1, 3, 16, 16)

    def tearDown(self):
        self.adapter.injected_model.close()

    def test_step_signature_has_only_image_payload(self):
        parameters = list(inspect.signature(DPAOnlineAdapter.step).parameters)
        self.assertEqual(parameters, ["self", "image"])

    def test_online_source_has_no_forbidden_calls(self):
        path = ROOT / "src" / "dpa_ctta" / "online.py"
        tree = ast.parse(path.read_text())
        called = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(called & {"backward", "grad"})
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        self.assertNotIn("torch.optim", imports)

    def test_online_source_does_not_import_offline(self):
        source = (ROOT / "src" / "dpa_ctta" / "online.py").read_text()
        self.assertNotIn("offline", source)

    def test_import_online_does_not_load_oracle(self):
        command = [
            sys.executable,
            "-c",
            "import sys, dpa_ctta.online; assert 'dpa_ctta.offline.oracle' not in sys.modules",
        ]
        subprocess.run(command, cwd=ROOT, check=True, env=os.environ.copy())

    def test_step_returns_all_contract_fields(self):
        result = self.adapter.step(self.image)
        self.assertEqual(result.logits.shape, (1, 1, 16, 16))
        self.assertEqual(result.state.shape, (1, 16))
        self.assertEqual(result.weights.shape, (1, 2))
        self.assertEqual(result.precision.shape, (1, 16))
        self.assertEqual(result.rhs.shape, (1, 16))
        self.assertIsInstance(result.contraction_bound, float)

    def test_step_updates_only_latent_state(self):
        before = {name: value.clone() for name, value in self.adapter.state_dict().items()}
        self.adapter.step(self.image)
        after = self.adapter.state_dict()
        changed = [name for name in before if not torch.equal(before[name], after[name])]
        self.assertEqual(changed, ["z_prev"])

    def test_reset_zeros_state(self):
        self.adapter.step(self.image)
        self.adapter.reset()
        self.assertEqual(int(torch.count_nonzero(self.adapter.z_prev)), 0)

    def test_same_state_and_input_replay(self):
        clone = clone_online(self.adapter)
        try:
            first = self.adapter.step(self.image)
            second = clone.step(self.image)
            self.assertTrue(torch.equal(first.logits, second.logits))
            self.assertTrue(torch.equal(first.state, second.state))
            self.assertTrue(torch.equal(first.weights, second.weights))
        finally:
            clone.injected_model.close()

    def test_state_dict_round_trip(self):
        self.adapter.step(self.image)
        clone = clone_online(self.adapter)
        try:
            for name, value in self.adapter.state_dict().items():
                self.assertTrue(torch.equal(value, clone.state_dict()[name]), name)
        finally:
            clone.injected_model.close()

    def test_source_base_gradients_are_none(self):
        self.adapter.step(self.image)
        source = self.adapter.injected_model.source_model
        self.assertTrue(all(parameter.grad is None for parameter in source.parameters()))

    def test_batchnorm_buffers_do_not_change(self):
        bn = self.adapter.injected_model.source_model.early[1]
        before = (bn.running_mean.clone(), bn.running_var.clone())
        self.adapter.step(self.image)
        self.assertTrue(torch.equal(before[0], bn.running_mean))
        self.assertTrue(torch.equal(before[1], bn.running_var))

    def test_batch_larger_than_one_rejected(self):
        with self.assertRaises(ValueError):
            self.adapter.step(self.image.repeat(2, 1, 1, 1))

    def test_full_synthetic_distillation_gradient_connectivity(self):
        frozen_logits, frozen_feature = self.adapter._zero_pass(self.image)
        query = self.adapter.atlas.descriptor(self.image, frozen_logits.sigmoid(), frozen_feature)
        atlas_result = self.adapter.atlas(query, self.adapter._frozen_feature)
        update = closed_form_update(torch.zeros(1, 16), atlas_result, 1.0, 1e-3)
        logits = self.adapter.injected_model(self.image, update.state)
        (logits.square().mean() + update.state.square().mean()).backward()
        for anchor in self.adapter.atlas.anchors:
            for name in ("image_logits", "mask_logits", "state_center", "local_map", "precision_raw"):
                gradient = getattr(anchor, name).grad
                self.assertIsNotNone(gradient, name)
                self.assertTrue(torch.isfinite(gradient).all(), name)
                self.assertGreater(int(torch.count_nonzero(gradient)), 0, name)
        for film in self.adapter.injected_model.adapters:
            for parameter in (film.gamma_basis, film.beta_basis):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)


class M0CommandTests(unittest.TestCase):
    def test_default_m0_is_dry_run(self):
        result = subprocess.run(
            [sys.executable, "scripts/m0_oracle.py"], cwd=ROOT, check=True,
            capture_output=True, text=True, env=os.environ.copy()
        )
        self.assertIn('"training_executed": false', result.stdout)

    def test_real_data_without_authorization_is_rejected(self):
        result = subprocess.run(
            [sys.executable, "scripts/m0_oracle.py", "--real-data-root", "/not/read"],
            cwd=ROOT, capture_output=True, text=True, env=os.environ.copy()
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--authorize-real-data-training", result.stderr)

    def test_authorized_real_data_still_waits_for_review(self):
        result = subprocess.run(
            [
                sys.executable,
                "scripts/m0_oracle.py",
                "--real-data-root",
                "/not/read",
                "--authorize-real-data-training",
            ],
            cwd=ROOT, capture_output=True, text=True, env=os.environ.copy()
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("BLOCKED_PENDING_CODE_REVIEW_PASS", result.stderr)


if __name__ == "__main__":
    unittest.main()
