import copy
import unittest
import torch
from torch.nn import functional as F
from .method import Host, candidates, scnp_logits
from ..r37_cw_orientation.method import Host as RetainedHost, candidates as retained_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_same_class_values_and_gradient_match_explicit_neighborhood(self):
        z = torch.tensor([[[[-2., 3., .4], [1., -1., 4.], [2., .8, -3.]],
                           [[.2, -4., 2.], [3., 1., -.7], [-1., 4., .9]]]], requires_grad=True)
        q = torch.tensor([[[[.9,.2,.7],[.8,.1,.3],[.4,.6,.9]],
                           [[.1,.8,.7],[.6,.2,.3],[.4,.9,.1]]]], requires_grad=True)
        oracle = torch.empty_like(z)
        for c in range(2):
            for i in range(3):
                for j in range(3):
                    same = q[0,c,max(0,i-1):i+2,max(0,j-1):j+2].detach() >= .5
                    values = z[0,c,max(0,i-1):i+2,max(0,j-1):j+2][same == (q[0,c,i,j].detach() >= .5)]
                    oracle[0,c,i,j] = values.min() if q[0,c,i,j] >= .5 else values.max()
        actual = scnp_logits(z, q)
        self.assertTrue(torch.equal(actual, oracle))
        a = F.binary_cross_entropy_with_logits(actual, q.detach())
        b = F.binary_cross_entropy_with_logits(oracle, q.detach())
        ga, = torch.autograd.grad(a, z, retain_graph=True)
        gb, = torch.autograd.grad(b, z)
        self.assertTrue(torch.allclose(ga, gb, atol=1e-7))
        self.assertTrue(torch.isfinite(ga).all()); self.assertIsNone(q.grad)
        self.assertTrue(torch.equal(actual.flip(1), scnp_logits(z.flip(1), q.flip(1))))

    def test_uniform_masks_isolated_pixels_and_ties_remain_finite(self):
        for v in (.1, .5, .9):
            z = torch.zeros(1,2,4,4,requires_grad=True)
            q = torch.full_like(z,v); q[0,0,0,0] = 1-v
            actual = scnp_logits(z,q)
            F.binary_cross_entropy_with_logits(actual,q).backward()
            self.assertTrue(torch.equal(actual,z)); self.assertTrue(torch.isfinite(z.grad).all())

    def test_controls_disabled_parity_and_active_snapshot_continuation(self):
        self.assertEqual(candidates()[:2],retained_candidates()[:2])
        self.assertEqual(candidates()[2],retained_candidates()[3])
        torch.manual_seed(42); model=Tiny()
        x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        control=RetainedHost(None,candidates()[2],17011,'same',device='cpu',model=copy.deepcopy(model))
        disabled=Host(None,dict(candidates()[-1],scnp_strength=0.),17011,'same',device='cpu',model=copy.deepcopy(model))
        active=Host(None,candidates()[-1],17011,'same',device='cpu',model=copy.deepcopy(model))
        try:
            a,_=control.step(x); b,_=disabled.step(x)
            self.assertTrue(torch.equal(a,b)); self.assertTrue(equal(control.snapshot(),disabled.snapshot()))
            active.step(x); self.assertGreater(active.diag['SCNP_loss'],0.)
            saved=active.snapshot(); a,_=active.step(x.roll(3,-1)); end=active.snapshot()
            active.restore(saved); b,_=active.step(x.roll(3,-1))
            self.assertTrue(torch.equal(a,b)); self.assertTrue(equal(end,active.snapshot()))
            for h in (control,disabled,active):h.check_frozen(True)
        finally:
            for h in (control,disabled,active):h.close()
