"""R6 proposal: bounded pixel weights and local logit-gradient norm controls.

Mathematical CPU reference only. This is NOT a CTTA host or a GPU runner.
No image files, labels, checkpoint, model, optimizer or network access is used.
The production implementation must retain the pinned C loss/augmentation path.
"""
from __future__ import annotations
import hashlib
import json
import math
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

ARMS = ("C", "R_BAL", "R_SCALE", "R_SHUFFLE")
RATIO_CAP = 8.0


def checked_prob(q: torch.Tensor) -> None:
    if q.ndim != 4 or q.shape[0] != 1 or q.shape[1] != 2:
        raise ValueError("expected 1 x 2 x H x W")
    if q.requires_grad or not bool(torch.isfinite(q).all()) or bool(((q < 0) | (q > 1)).any()):
        raise ValueError("detached finite probability required")


def local_seed(visit: int, channel: int) -> int:
    if type(visit) is not int or visit < 1 or channel not in (0, 1):
        raise ValueError("invalid visit/channel")
    raw = f"R6_WEIGHT_PERM_V1|20260907|{visit}|{channel}".encode("ascii")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def make_weights(q: torch.Tensor, visit: int) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    checked_prob(q)
    h = (q.detach().cpu() >= .5)
    w = torch.ones(h.shape, dtype=torch.float64)
    wp = torch.empty_like(w)
    meta = []
    for c in range(2):
        n = h[0, c].numel()
        nf = int(h[0, c].sum())
        nb = n - nf
        if nf == 0 or nb == 0:
            ratio = wf = wb = 1.0
            fallback = "ONE_PARTITION_EMPTY"
        else:
            ratio = min(RATIO_CAP, max(1.0 / RATIO_CAP, nb / nf))
            wb = n / (ratio * nf + nb)
            wf = ratio * wb
            fallback = None
            w[0, c].fill_(wb)
            w[0, c][h[0, c]] = wf
        g = torch.Generator(device="cpu").manual_seed(local_seed(visit, c))
        perm = torch.randperm(n, generator=g)
        wp[0, c] = w[0, c].reshape(-1)[perm].reshape(h.shape[-2:])
        meta.append(dict(channel=c, n_fg=nf, n_bg=nb, ratio=ratio,
                         w_fg=wf, w_bg=wb, fallback=fallback,
                         seed=local_seed(visit, c)))
    return w, wp, meta


def coefficients(z: torch.Tensor, q: torch.Tensor, visit: int):
    checked_prob(q)
    if z.shape != q.shape or not bool(torch.isfinite(z).all()):
        raise ValueError("finite same-shape logits required")
    wd, pd, meta = make_weights(q, visit)
    # Diagnostics use the weights actually rounded to the loss dtype.
    w = wd.to(device=z.device, dtype=z.dtype)
    wp = pd.to(device=z.device, dtype=z.dtype)
    with torch.no_grad():
        d = (torch.sigmoid(z.detach()) - q.to(z.device, z.dtype)).cpu().double()
        wc, pc = w.detach().cpu().double(), wp.detach().cpu().double()
        s0 = d.square().sum(dim=(-1, -2))[0]
        sw = (wc * d).square().sum(dim=(-1, -2))[0]
        sp = (pc * d).square().sum(dim=(-1, -2))[0]
        a = torch.ones(2, dtype=torch.float64)
        b = torch.ones(2, dtype=torch.float64)
        for c in range(2):
            if s0[c] > 0:
                if not (sw[c] > 0 and sp[c] > 0):
                    raise ValueError("positive weights must preserve nonzero residual norm")
                a[c] = torch.sqrt(sw[c] / s0[c])
                b[c] = torch.sqrt(sw[c] / sp[c])
        if not bool(torch.isfinite(a).all() and torch.isfinite(b).all()):
            raise ValueError("nonfinite scale")
    return (w, wp, a.to(z.device, z.dtype).reshape(1, 2, 1, 1),
            b.to(z.device, z.dtype).reshape(1, 2, 1, 1),
            dict(regions=meta, S0=s0.tolist(), Sw=sw.tolist(), Sperm=sp.tolist(),
                 scale=a.tolist(), shuffle_scale=b.tolist()))


def loss(z: torch.Tensor, q: torch.Tensor, arm: str, visit: int = 1):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    w, wp, a, b, meta = coefficients(z, q, visit)
    if arm == "C":
        out = F.binary_cross_entropy_with_logits(z, q)
    else:
        element = F.binary_cross_entropy_with_logits(z, q, reduction="none")
        applied = w if arm == "R_BAL" else a if arm == "R_SCALE" else b * wp
        out = (applied.detach() * element).mean()
    return out, meta


class ReferenceTests(unittest.TestCase):
    def data(self):
        g = torch.Generator().manual_seed(12)
        q = torch.rand((1, 2, 11, 13), generator=g, dtype=torch.float64)
        q[:, 0] *= .65
        q[:, 1] *= .55
        z = torch.randn(q.shape, generator=g, dtype=torch.float64)
        return z, q

    def test_weights_mean_bounds_and_partition_mass(self):
        _, q = self.data(); w, wp, meta = make_weights(q, 7)
        self.assertTrue(torch.allclose(w.mean((-1, -2)), torch.ones((1, 2), dtype=w.dtype), atol=1e-14, rtol=0))
        self.assertGreaterEqual(float(w.min()), 1/8)
        self.assertLessEqual(float(w.max()), 8)
        for c in range(2):
            self.assertTrue(torch.equal(w[0,c].flatten().sort().values, wp[0,c].flatten().sort().values))
            m=meta[c]
            self.assertAlmostEqual((m['n_fg']*m['w_fg']+m['n_bg']*m['w_bg'])/143, 1.)

    def test_per_channel_gradient_norms(self):
        z,q=self.data(); norms={}
        for arm in ARMS:
            x=z.clone().requires_grad_(True); ell,_=loss(x,q,arm,17)
            grad,=torch.autograd.grad(ell,x)
            norms[arm]=grad.square().sum((-1,-2))
        for arm in ('R_SCALE','R_SHUFFLE'):
            self.assertTrue(torch.allclose(norms[arm],norms['R_BAL'],atol=1e-18,rtol=1e-12))
        self.assertFalse(torch.allclose(norms['C'],norms['R_BAL'],atol=1e-18,rtol=1e-6))

    def test_soft_target_stationary_point(self):
        _,q=self.data(); q=q.clamp(.01,.99)
        for arm in ARMS:
            z=torch.logit(q).detach().requires_grad_(True)
            ell,_=loss(z,q,arm)
            grad,=torch.autograd.grad(ell,z)
            self.assertLess(float(grad.abs().max()),1e-16)

    def test_uniform_weight_degeneracy(self):
        g=torch.Generator().manual_seed(1)
        z=torch.randn((1,2,4,4),generator=g,dtype=torch.float64)
        for val in (0.,1.,.2,.8):
            q=torch.full_like(z,val)
            ref=F.binary_cross_entropy_with_logits(z,q)
            for arm in ARMS:
                actual,_=loss(z,q,arm)
                self.assertTrue(torch.equal(actual,ref))

    def test_zero_residual_branch(self):
        z=torch.zeros((1,2,4,4),dtype=torch.float64)
        q=torch.full_like(z,.5)
        _,_,a,b,meta=coefficients(z,q,1)
        self.assertEqual(meta['S0'],[0.,0.]);self.assertTrue(bool((a==1).all() and (b==1).all()))

    def test_permutation_reproducibility_and_rng_isolation(self):
        _,q=self.data(); state=torch.get_rng_state().clone()
        w,p,m=make_weights(q,20)
        w2,p2,m2=make_weights(q,20)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        self.assertTrue(torch.equal(w,w2) and torch.equal(p,p2));self.assertEqual(m,m2)
        _,p3,_=make_weights(q,21);self.assertFalse(torch.equal(p,p3))

    def test_detached_weights_and_scales(self):
        z,q=self.data();z.requires_grad_(True)
        w,wp,a,b,_=coefficients(z,q,5)
        self.assertTrue(all(not x.requires_grad for x in (w,wp,a,b)))

    def test_bad_input_rejected(self):
        z,q=self.data()
        for bad in (torch.full_like(q,float('nan')),torch.full_like(q,1.1),q.clone().requires_grad_(True)):
            with self.assertRaises(ValueError):loss(z,bad,'R_BAL')
        with self.assertRaises(ValueError):loss(torch.full_like(z,float('inf')),q,'R_BAL')


if __name__ == '__main__':
    torch.set_num_threads(2)
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ReferenceTests))
    status=dict(scope='R6 mathematical CPU reference only; no network/CTTA host/optimizer',
                tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
                python=__import__('sys').version.split()[0],torch=torch.__version__,
                cuda_initialized=torch.cuda.is_initialized(),real_data_reads=0,checkpoint_reads=0,
                success=result.wasSuccessful())
    print(json.dumps(status,indent=2))
    Path(__file__).with_name('R6_MATH_CHECK.json').write_text(json.dumps(status,indent=2)+'\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
