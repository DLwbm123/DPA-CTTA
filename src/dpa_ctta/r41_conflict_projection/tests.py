import copy
import unittest
import torch
from .method import Host, candidates, project_conflict
from ..r39_conflict_trust.method import Host as ConflictHost, candidates as previous_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_projection_preserves_orthogonal_and_aligned_components(self):
        m=torch.tensor([2.,0.]);g=torch.tensor([-3.,4.]);p,a=project_conflict(g,m)
        self.assertTrue(torch.equal(p,torch.tensor([0.,4.])));self.assertLess(a,0.)
        self.assertGreaterEqual(float(p@m),0.)
        for reference in (m,torch.zeros_like(m)):
            g=torch.tensor([3.,4.]);p,a=project_conflict(g,reference)
            self.assertTrue(torch.equal(p,g));self.assertEqual(a,0.)

    def test_disabled_first_step_parity_and_stateful_projection(self):
        torch.manual_seed(42);model=Tiny();x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        self.assertEqual(candidates()[:2],previous_candidates()[:2]);self.assertEqual(candidates()[2],previous_candidates()[3])
        for enabled in (False,True):
            h=Host(None,dict(candidates()[-1],bn_conflict_projection=enabled),17011,'same',device='cpu',model=copy.deepcopy(model))
            control=ConflictHost(None,candidates()[2],17011,'same',device='cpu',model=copy.deepcopy(model))
            try:
                za,_=h.step(x);zb,_=control.step(x)
                self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(h.snapshot(),control.snapshot()))
                if not enabled:
                    za,_=h.step(x.roll(3,-1));zb,_=control.step(x.roll(3,-1))
                    self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(h.snapshot(),control.snapshot()))
                else:
                    saved=h.snapshot();h.step(x)
                    current=torch.cat([p.grad.detach().flatten() for p in h.native.params]).clone()
                    h.restore(saved);h.gradient_memory=-current
                    z,d=h.step(x);diag=d['diagnostics']
                    self.assertEqual(diag['BN_projection_active'],1)
                    self.assertGreater(diag['BN_projection_removed_norm'],0.)
                    self.assertGreaterEqual(diag['BN_projected_reference_dot'],-1e-5)
                    self.assertTrue(torch.isfinite(z).all());h.check_frozen(True)
                    saved=h.snapshot();za,_=h.step(x.roll(3,-1));final=h.snapshot()
                    h.restore(saved);zb,_=h.step(x.roll(3,-1))
                    self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(final,h.snapshot()))
            finally:h.close();control.close()


if __name__ == '__main__': unittest.main()
