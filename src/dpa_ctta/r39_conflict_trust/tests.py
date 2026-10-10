import copy
import unittest
import torch
from .method import Host, candidates
from ..r36_incremental_modules.method import Host as RetainedHost
from ..r38_bn_trust.method import candidates as previous_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_controls_disabled_gate_and_first_step_parity(self):
        torch.manual_seed(42); model=Tiny(); x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        self.assertEqual(candidates()[:3],previous_candidates()[:3])
        for radius in (0.,.0015):
            a=Host(None,dict(candidates()[-1],bn_trust_radius=radius),17011,'same',device='cpu',model=copy.deepcopy(model))
            b=RetainedHost(None,candidates()[2],17011,'same',device='cpu',model=copy.deepcopy(model))
            try:
                za,da=a.step(x); zb,_=b.step(x)
                self.assertTrue(torch.equal(za,zb))
                sa=a.snapshot(); sa.pop('gradient_memory',None)
                self.assertTrue(equal(sa,b.snapshot()))
                if radius:self.assertEqual(da['diagnostics']['BN_trust_scale'],1.)
            finally:a.close();b.close()

    def test_forced_conflict_cap_and_memory_continuation(self):
        torch.manual_seed(42); x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        h=Host(None,dict(candidates()[-1],bn_trust_radius=1e-5),17011,'same',device='cpu',model=Tiny())
        try:
            _,d=h.step(x);self.assertEqual(d['diagnostics']['BN_trust_scale'],1.)
            saved=h.snapshot();h.step(x)
            current=torch.cat([p.grad.detach().flatten() for p in h.native.params]).clone()
            h.restore(saved);h.gradient_memory=-current
            before=[p.detach().clone() for p in h.native.params]
            z,d=h.step(x)
            self.assertLess(d['diagnostics']['BN_gradient_cosine'],0.)
            self.assertLess(d['diagnostics']['BN_trust_scale'],1.)
            norm=torch.stack([(p-q).square().sum() for p,q in zip(h.native.params,before)]).sum().sqrt()
            self.assertLessEqual(float(norm),1.02e-5);self.assertTrue(torch.isfinite(z).all());h.check_frozen(True)
            saved=h.snapshot(); za,_=h.step(x.roll(3,-1));final=h.snapshot()
            h.restore(saved);zb,_=h.step(x.roll(3,-1))
            self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(final,h.snapshot()))
        finally:h.close()


if __name__ == '__main__': unittest.main()
