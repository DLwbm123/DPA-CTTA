import unittest

import torch

from dpa_ctta.r7_b_rca import correct as r7_correct
from dpa_ctta.r8_ba.methods import CurrentMLP, R8A, R8B, correct, film


class TestR8Methods(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
