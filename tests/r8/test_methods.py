import unittest
import copy

import torch
from torch import nn

from dpa_ctta.r7_b_rca import correct as r7_correct
from dpa_ctta.hosts.vptta import model_input_from_pixels
from dpa_ctta.r8_ba.methods import CurrentMLP, R8A, R8B, R8Segmenter, correct, film, from_selected
from dpa_ctta.r8_ba.trainer import method_config


class Small(nn.Module):
    def __init__(self):
        super().__init__()
        self.res = nn.Module()
        self.res.conv1 = nn.Conv2d(3, 64, 1)
        self.up1 = nn.Conv2d(64, 256, 1)
        self.up3 = nn.Conv2d(256, 256, 1)
        self.seg_head = nn.Conv2d(256, 2, 1)

    def forward(self, image):
        h = torch.nn.functional.adaptive_avg_pool2d(image, (8, 8))
        h = self.up3(self.up1(self.res.conv1(h)))
        return torch.nn.functional.interpolate(self.seg_head(h), (512, 512))


class TestR8Methods(unittest.TestCase):
    def test_selected_source_loader_binds_weights_basis_and_rng(self):
        basis = torch.eye(1024, dtype=torch.float64)[:, :32]
        config = dict(id="synthetic", route="B", rank=32, film_amplitude=0.1,
                      observer="global", aux_multiplier=1.0)
        for mlp in (False, True):
            method = CurrentMLP(basis, 0.1, "global") if mlp else R8B(basis, 0.1)
            method.observer.fit_scaler(torch.randn(4, 134), "fit")
            snapshot = dict(steps=1000, binding="binding", source_seed=20260924,
                            method_config=method_config(method), method=copy.deepcopy(method.state_dict()),
                            method_digest=method.digest())
            rng = torch.get_rng_state().clone()
            loaded = from_selected(snapshot, config, basis, False, 20260924, "binding", mlp=mlp)
            self.assertEqual(loaded.digest(), method.digest())
            self.assertTrue(torch.equal(torch.get_rng_state(), rng))
            with self.assertRaisesRegex(ValueError, "weight/basis"):
                from_selected(snapshot, config, -basis, False, 20260924, "binding", mlp=mlp)

    def test_zero_film_and_rank_gradient(self):
        h = torch.randn(1, 256, 2, 2)
        for amplitude in (0.1, 0.3):
            v = torch.zeros(512, requires_grad=True)
            y = film(h, v, amplitude)
            self.assertTrue(torch.equal(y, h))
            y.sum().backward()
            self.assertGreater(v.grad.abs().sum().item(), 0)
        raw, tokens = torch.randn(134), torch.randn(64, 64)
        for cls, rank in ((R8A, 16), (R8A, 32), (R8B, 32), (R8B, 64)):
            basis = torch.eye(1024, dtype=torch.float64)[:, :rank]
            kwargs = dict(observer="current_tokens") if cls is R8B else {}
            method = cls(basis, 0.3, **kwargs)
            method.observer.fit_scaler(torch.randn(4, 134), "fit")
            state, audit = method.update(raw, tokens, method.initial())
            self.assertEqual(method.code(state).shape, (rank,))
            loss = method.fit_loss(audit, audit, method.project(torch.randn(1024)), state)
            loss.backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in method.parameters()))
        mlp = CurrentMLP(torch.eye(1024, dtype=torch.float64)[:, :32], 0.1, "global")
        mlp.observer.fit_scaler(torch.randn(4, 134), "fit")
        self.assertEqual(mlp.update(raw, tokens, mlp.initial())[0]["z"].shape, (32,))

    def test_five_step_ista_matches_frozen_r7(self):
        h, o, prior = torch.randn(64, 32), torch.randn(64), torch.randn(32)
        old, _ = r7_correct(h, o, prior, 1.)
        new, _ = correct(h, o, prior, 1.)
        self.assertTrue(torch.equal(old, new))
        twenty, audit = correct(h, o, prior, 1., 20)
        self.assertEqual(audit["energy"].shape, (21,))
        self.assertEqual(twenty.shape, (32,))

    def test_method_digest_binds_scientific_config(self):
        basis = torch.eye(1024, dtype=torch.float64)[:, :32]
        first = R8B(basis, 0.1, aux_multiplier=1.0)
        second = R8B(basis, 0.1, aux_multiplier=0.1)
        second.load_state_dict(first.state_dict())
        self.assertNotEqual(first.digest(), second.digest())

    def test_segmenter_policy_and_zero_C0(self):
        segmenter = R8Segmenter(Small(), 0.3)
        image = torch.rand(1, 3, 512, 512)
        zero = segmenter(image)
        v = torch.zeros(1024, requires_grad=True)
        changed = segmenter(image, v)
        normalized = model_input_from_pixels(image, "fundus")
        self.assertTrue(torch.equal(changed, segmenter.normalized(normalized, v)))
        self.assertTrue(torch.equal(zero, changed))
        self.assertEqual(segmenter.inference_policy["schema"], "R8_SEGMENTER_POLICY_V1")
        self.assertGreater(torch.autograd.grad(changed.mean(), v)[0].norm().item(), 0)
        segmenter.close()

    def test_zero_observer_cache_matches_both_amplitudes(self):
        original = Small()
        first = R8Segmenter(copy.deepcopy(original), 0.1)
        second = R8Segmenter(copy.deepcopy(original), 0.3)
        image = torch.rand(1, 3, 512, 512)
        with torch.no_grad():
            left = first(image, observe=True)
            right = second(image, observe=True)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(left, right)))
        first.close()
        second.close()


if __name__ == "__main__":
    unittest.main()
