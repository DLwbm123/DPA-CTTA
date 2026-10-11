import copy
import unittest
import torch
from .method import Host, candidates, band_powers, entropy
from ..r37_cw_orientation.method import Host as RetainedHost, candidates as retained_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_radial_energy_channels_dc_and_finite_gradient(self):
        torch.manual_seed(4)
        z=(2+torch.rand(1,2,32,32)).requires_grad_()
        powers=band_powers(z)
        energy=torch.fft.fft2(z.abs(),norm='ortho').abs().square()
        f=torch.fft.fftfreq(32)*32
        radius=(f[:,None].square()+f[None,:].square()).sqrt()
        self.assertTrue(torch.allclose(powers.sum(-1),energy[..., (radius>0)&(radius<=16)].sum(-1)[0],atol=1e-5))
        self.assertTrue(torch.equal(powers.flip(0),band_powers(z.flip(1))))
        self.assertTrue(torch.allclose(entropy(powers),entropy(band_powers(z*2)),atol=1e-6))
        loss=(torch.tensor([1.6,1.6])-entropy(powers)).relu().square().mean()
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(float(z.grad.norm()),0)
        for v in (0.,1.):
            a=torch.full((1,2,32,32),v,requires_grad=True)
            entropy(band_powers(a)).sum().backward()
            self.assertTrue(torch.isfinite(a.grad).all());self.assertTrue(torch.equal(band_powers(a),torch.zeros(2,5)))

    def test_controls_first_arrival_causal_target_and_snapshot(self):
        self.assertEqual(candidates()[:2],retained_candidates()[:2])
        self.assertEqual(candidates()[2],retained_candidates()[3])
        torch.manual_seed(42);model=Tiny()
        x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        control=RetainedHost(None,candidates()[2],17011,'same',device='cpu',model=copy.deepcopy(model))
        disabled=Host(None,dict(candidates()[-1],spectral_strength=0.),17011,'same',device='cpu',model=copy.deepcopy(model))
        active=Host(None,candidates()[-1],17011,'same',device='cpu',model=copy.deepcopy(model))
        try:
            a,_=control.step(x);b,_=disabled.step(x)
            self.assertTrue(torch.equal(a,b));self.assertTrue(equal(control.snapshot(),disabled.snapshot()))
            a,_=active.step(x);self.assertEqual(active.diag['spectral_loss'],0.)
            self.assertTrue(torch.equal(a,b));self.assertEqual(len(active.spectral_history),1)
            target=entropy(active.spectral_history[0]);saved=active.snapshot()
            a,_=active.step(x.roll(3,-1));end=active.snapshot()
            self.assertEqual(active.diag['spectral_reference_count'],1)
            self.assertTrue(torch.equal(torch.tensor(active.diag['spectral_target_entropy']),target))
            self.assertTrue(all(not p.requires_grad for p in active.spectral_history))
            active.restore(saved);b,_=active.step(x.roll(3,-1))
            self.assertTrue(torch.equal(a,b));self.assertTrue(equal(end,active.snapshot()))
            for _ in range(55):active.spectral_history.append(torch.ones(2,5))
            self.assertEqual(len(active.spectral_history),50)
            for h in (control,disabled,active):h.check_frozen(True)
        finally:
            for h in (control,disabled,active):h.close()
