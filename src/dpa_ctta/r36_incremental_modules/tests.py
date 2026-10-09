"""Gradient geometry, nested channels, disabled parity and recoverable EMA state."""
import copy
import unittest
import torch
from .method import Host, candidates, local_structure_loss, persistence_weight
from ..r20_model_only_search.method import Host as ExistingHost
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_flat_target_no_geometry_and_finite_gradient(self):
        z=torch.randn(1,2,64,64,requires_grad=True)
        a,b=local_structure_loss(z,torch.full_like(z,.4),torch.ones_like(z))
        self.assertEqual(float(a+b),0.)
        (a+b).backward(); self.assertTrue(torch.isfinite(z.grad).all())

    def test_nontrivial_losses_gradients_and_channel_symmetry(self):
        q=torch.full((1,2,64,64),.1);q[:,0,8:56,16:48]=.9;q[:,1,20:44,24:40]=.9
        z=torch.logit(q.transpose(-2,-1)).detach().requires_grad_()
        a,b=local_structure_loss(z,q,torch.ones_like(q))
        self.assertGreater(float(a),.01);self.assertGreater(float(b),.01)
        (a+b).backward();self.assertTrue(torch.isfinite(z.grad).all());self.assertGreater(float(z.grad.norm()),0.)
        swapped=local_structure_loss(z.detach().flip(1),q.flip(1),torch.ones_like(q))
        self.assertTrue(torch.allclose(torch.stack((a,b)),torch.stack(swapped),atol=1e-6))

    def test_persistence_empty_and_stable_component(self):
        q=torch.zeros(1,2,64,64);self.assertEqual(float(persistence_weight(q).sum()),0.)
        q[:,:,16:48,16:48]=1.;w=persistence_weight(q)
        self.assertTrue(torch.isfinite(w).all());self.assertGreater(float(w.max()),.5)
        self.assertTrue((w>=0).all() and (w<=1).all());self.assertFalse(w.requires_grad)

    def test_disabled_module_exact_existing_host(self):
        torch.manual_seed(42);model=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        c=candidates()[1];disabled=dict(c,module='both',module_strength=0.)
        a=Host(None,disabled,20260907,'same',device='cpu',model=copy.deepcopy(model))
        b=ExistingHost(None,c,20260907,'same',device='cpu',model=copy.deepcopy(model))
        try:
            za,_=a.step(x);zb,_=b.step(x);self.assertTrue(torch.equal(za,zb))
            sa,sb=a.snapshot(),b.snapshot()
            for key in ('parameters','gradients','adam','grata','native_rng'):self.assertTrue(equal(sa[key],sb[key]),key)
            a.check_frozen(True);b.check_frozen(True)
        finally:a.close();b.close()

    def test_balance_snapshot_next_update_exact(self):
        torch.manual_seed(42);model=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        h=Host(None,candidates()[-1],20260907,'same',device='cpu',model=model)
        try:
            h.step(x);s=h.snapshot();a,_=h.step(x.roll(3,-1));end=h.snapshot()
            h.restore(s);b,_=h.step(x.roll(3,-1));self.assertTrue(torch.equal(a,b));self.assertTrue(equal(end,h.snapshot()))
            h.check_frozen(True)
        finally:h.close()


if __name__=='__main__':unittest.main()
