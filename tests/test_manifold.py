import unittest

import torch

from dpa_ctta.manifold import LatentFiLM, MultiScaleLatentFiLM
from common import FailingSource, TinySource


class LatentFiLMTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.layer = LatentFiLM(4, 16)
        self.feature = torch.randn(2, 4, 8, 8)

    def test_zero_state_without_gradient_returns_same_tensor_object(self):
        output = self.layer(self.feature, torch.zeros(16))
        self.assertIs(output, self.feature)

    def test_zero_state_with_gradient_is_differentiable(self):
        state = torch.zeros(16, requires_grad=True)
        self.layer(self.feature, state).sum().backward()
        self.assertGreater(int(torch.count_nonzero(state.grad)), 0)

    def test_zero_state_bitwise_identity(self):
        self.assertTrue(torch.equal(self.layer(self.feature, torch.zeros(16)), self.feature))

    def test_nonzero_state_changes_feature(self):
        self.assertFalse(torch.equal(self.layer(self.feature, torch.ones(16)), self.feature))

    def test_batched_state_supported(self):
        output = self.layer(self.feature, torch.ones(2, 16))
        self.assertEqual(output.shape, self.feature.shape)

    def test_single_state_broadcasts(self):
        output = self.layer(self.feature, torch.ones(1, 16))
        self.assertEqual(output.shape, self.feature.shape)

    def test_wrong_feature_channels_rejected(self):
        with self.assertRaises(ValueError):
            self.layer(torch.randn(2, 3, 8, 8), torch.zeros(16))

    def test_wrong_state_dimension_rejected(self):
        with self.assertRaises(ValueError):
            self.layer(self.feature, torch.zeros(8))

    def test_wrong_state_batch_rejected(self):
        with self.assertRaises(ValueError):
            self.layer(self.feature, torch.zeros(3, 16))


class MultiScaleManifoldTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.source = TinySource().eval()
        self.wrapper = MultiScaleLatentFiLM(
            self.source, {"low": 4, "mid": 4, "tail": 1}
        )
        self.image = torch.randn(1, 3, 16, 16)

    def tearDown(self):
        self.wrapper.close()

    def test_requires_exactly_three_points(self):
        with self.assertRaises(ValueError):
            MultiScaleLatentFiLM(TinySource(), {"low": 4, "mid": 4})

    def test_missing_module_rejected(self):
        with self.assertRaises(ValueError):
            MultiScaleLatentFiLM(TinySource(), {"low": 4, "mid": 4, "missing": 1})

    def test_source_parameters_frozen(self):
        self.assertTrue(all(not parameter.requires_grad for parameter in self.source.parameters()))

    def test_zero_state_logits_bitwise_identity(self):
        baseline = self.source(self.image)
        result = self.wrapper(self.image, torch.zeros(1, 16))
        self.assertTrue(torch.equal(result, baseline))

    def test_none_state_logits_bitwise_identity(self):
        baseline = self.source(self.image)
        self.assertTrue(torch.equal(self.wrapper(self.image), baseline))

    def test_nonzero_state_changes_logits(self):
        baseline = self.source(self.image)
        self.assertFalse(torch.equal(self.wrapper(self.image, torch.ones(1, 16)), baseline))

    def test_same_state_reaches_all_three_points(self):
        captured = {name: [] for name in self.wrapper.injection_paths}
        handles = [
            dict(self.source.named_modules())[name].register_forward_hook(
                lambda _m, _i, output, key=name: captured[key].append(output.detach().clone())
            )
            for name in captured
        ]
        try:
            self.wrapper(self.image, torch.zeros(1, 16))
            self.wrapper(self.image, torch.ones(1, 16))
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(all(len(values) == 2 and not torch.equal(*values) for values in captured.values()))

    def test_all_film_bases_receive_gradient(self):
        self.wrapper(self.image, torch.ones(1, 16)).square().mean().backward()
        for adapter in self.wrapper.adapters:
            for parameter in (adapter.gamma_basis, adapter.beta_basis):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(int(torch.count_nonzero(parameter.grad)), 0)

    def test_source_gradient_remains_none(self):
        self.wrapper(self.image, torch.ones(1, 16)).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in self.source.parameters()))

    def test_source_batchnorm_buffers_unchanged(self):
        bn = self.source.early[1]
        before = (bn.running_mean.clone(), bn.running_var.clone())
        self.wrapper.train()
        self.wrapper(self.image, torch.ones(1, 16))
        self.assertTrue(torch.equal(before[0], bn.running_mean))
        self.assertTrue(torch.equal(before[1], bn.running_var))

    def test_close_removes_hooks(self):
        modules = dict(self.source.named_modules())
        self.wrapper.close()
        self.assertTrue(all(len(modules[path]._forward_hooks) == 0 for path in self.wrapper.injection_paths))

    def test_closed_wrapper_refuses_forward(self):
        self.wrapper.close()
        with self.assertRaises(RuntimeError):
            self.wrapper(self.image)

    def test_exception_resets_active_state(self):
        wrapper = MultiScaleLatentFiLM(
            FailingSource(), {"low": 4, "mid": 4, "tail": 1}
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                wrapper(self.image, torch.ones(1, 16))
            self.assertIsNone(wrapper._active_state)
        finally:
            wrapper.close()


if __name__ == "__main__":
    unittest.main()
