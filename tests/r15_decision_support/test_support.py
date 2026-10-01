import unittest,torch
from dpa_ctta.r15_decision_support.view import supported
class Support(unittest.TestCase):
    def test_decision_proof_extremes_and_thresholds(self):
        gen=torch.Generator().manual_seed(20260924);a=torch.cat((torch.randn(8192,generator=gen)*8,torch.tensor([-100.,-1e-7,0.,1e-7,100.])));b=torch.cat((torch.randn(8192,generator=gen)*8,torch.tensor([100.,0.,-1e-7,0.,-100.])));q=torch.logit((.75*a.sigmoid()+.25*b.sigmoid()).clamp(1e-6,1-1e-6));out,change=supported(a,q)
        self.assertTrue(torch.equal(out.sigmoid()>=.5,q.sigmoid()>=.5));self.assertTrue(torch.equal(out[~change],a[~change]));self.assertTrue(torch.equal(out[change],q[change]));self.assertGreater(int(change.sum()),0)
