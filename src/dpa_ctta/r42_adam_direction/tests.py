import copy
import unittest
import torch
from .method import Host, candidates
from ..r41_conflict_projection.method import Host as GradientHost, candidates as previous_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_adam_coordinate_scaling_can_reverse_reference_dot(self):
        p=torch.nn.Parameter(torch.zeros(2));adam=torch.optim.Adam([p],lr=1e-4)
        p.grad=torch.tensor([3.,-1.]);reference=torch.tensor([1.,2.])
        self.assertGreater(float(p.grad@reference),0.)
        adam.step();self.assertLess(float((-p.detach())@reference),0.)

    def test_disabled_first_step_and_snapshot_continuation(self):
        torch.manual_seed(42);model=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        self.assertEqual(candidates()[:2],previous_candidates()[:2]);self.assertEqual(candidates()[2],previous_candidates()[3])
        for enabled in (False,True):
            h=Host(None,dict(candidates()[-1],bn_adam_direction_projection=enabled),17011,'same',device='cpu',model=copy.deepcopy(model))
            control=GradientHost(None,candidates()[2],17011,'same',device='cpu',model=copy.deepcopy(model))
            try:
                za,_=h.step(x);zb,_=control.step(x)
                self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(h.snapshot(),control.snapshot()))
                za,_=h.step(x.roll(3,-1))
                if not enabled:
                    zb,_=control.step(x.roll(3,-1));self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(h.snapshot(),control.snapshot()))
                saved=h.snapshot();za,_=h.step(x.roll(7,-1));end=h.snapshot()
                h.restore(saved);zb,_=h.step(x.roll(7,-1))
                self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(end,h.snapshot()));h.check_frozen(True)
            finally:h.close();control.close()

    def test_post_adam_projection_keeps_moments_and_retained_cap(self):
        torch.manual_seed(42);h=Host(None,candidates()[-1],17011,'same',device='cpu',model=Tiny())
        try:
            h.step(torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512))
            before=[p.detach().clone() for p in h.native.params]
            descent=torch.zeros(sum(p.numel() for p in h.native.params));descent[0]=-.02;descent[1]=.03
            reference=torch.zeros_like(descent);reference[0]=1.
            with torch.no_grad():
                offset=0
                for p,a in zip(h.native.params,before):
                    p.copy_(a-descent[offset:offset+p.numel()].reshape_as(p));offset+=p.numel()
            moments=h.snapshot()['adam'];h.trust_before=before;h.step_reference=reference;h.gradient_conflict=True
            h._trust_after_adam(None,(),{})
            actual=torch.cat([(a-p.detach()).flatten() for a,p in zip(before,h.native.params)])
            self.assertEqual(h.diag['BN_adam_direction_active'],1);self.assertGreater(h.diag['BN_adam_removed_norm'],0.)
            self.assertGreaterEqual(float(actual@reference),-1e-7);self.assertGreater(float(actual[1]),0.)
            self.assertLessEqual(float(actual.norm()),.001501);self.assertTrue(equal(moments,h.snapshot()['adam']))
            self.assertTrue(torch.isfinite(actual).all());h.check_frozen(True)
        finally:h.close()
