import copy
import unittest
import torch
from .method import Host, candidates
from ..r36_incremental_modules.method import Host as RetainedHost
from ..r37_cw_orientation.method import candidates as previous_candidates
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal
from ..r37_cw_orientation.run import comparison_passes


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_disabled_cap_exact_retained_state_and_controls(self):
        torch.manual_seed(42); model=Tiny(); c=candidates()
        self.assertEqual(c[:2],previous_candidates()[:2])
        self.assertEqual(c[2],previous_candidates()[3])
        x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        a=Host(None,dict(c[3],bn_trust_radius=0.),17011,'same',device='cpu',model=copy.deepcopy(model))
        b=RetainedHost(None,c[2],17011,'same',device='cpu',model=copy.deepcopy(model))
        try:
            za,_=a.step(x); zb,_=b.step(x)
            self.assertTrue(torch.equal(za,zb)); self.assertTrue(equal(a.snapshot(),b.snapshot()))
        finally: a.close(); b.close()

    def test_cap_bound_forward_and_snapshot_continuation(self):
        torch.manual_seed(42); x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)
        h=Host(None,dict(candidates()[-1],bn_trust_radius=1e-5),17011,'same',device='cpu',model=Tiny())
        try:
            before=[p.detach().clone() for p in h.native.params]
            z,d=h.step(x)
            norm=torch.stack([(p-q).square().sum() for p,q in zip(h.native.params,before)]).sum().sqrt()
            self.assertLessEqual(float(norm),1.02e-5)
            self.assertLess(d['diagnostics']['BN_trust_scale'],1.)
            self.assertGreater(d['diagnostics']['BN_trust_raw_norm'],1e-5)
            self.assertTrue(torch.isfinite(z).all()); h.check_frozen(True)
            saved=h.snapshot(); za,_=h.step(x.roll(3,-1)); final=h.snapshot()
            h.restore(saved); zb,_=h.step(x.roll(3,-1))
            self.assertTrue(torch.equal(za,zb)); self.assertTrue(equal(final,h.snapshot()))
            self.assertIsNone(h.trust_before)
        finally: h.close()

    def test_noninferiority_does_not_relax_strong_control_guard(self):
        p=dict(status='COMPLETE',baseline='CW_LSO',delta_pp='-0.05')
        self.assertTrue(comparison_passes(p,'CW_LSO',.05))
        self.assertFalse(comparison_passes(dict(p,delta_pp='-0.0501'),'CW_LSO',.05))
        w=dict(status='COMPLETE',baseline='W',delta_pp='0.9',order_delta_pp='[0.8,1.0]',
               positive_trajectories='6',imageweighted_delta_pp='0.3',worst_seed_averaged_cell_pp='-2.1')
        self.assertFalse(comparison_passes(w,'CW_LSO',.05))


if __name__=='__main__': unittest.main()
