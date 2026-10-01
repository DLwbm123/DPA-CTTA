import unittest,torch
from dpa_ctta.r14_uncertainty_flip.view import gated
class Gate(unittest.TestCase):
    def test_protection_and_uncertain_formula(self):
        p=torch.tensor([.05,.2,.45,.5,.55,.8,.95]);q=1-p;z=torch.logit(p);f=torch.logit(q)
        result,mask=gated(z,f);expected=torch.logit((.75*z.sigmoid()+.25*f.sigmoid()).clamp(1e-6,1-1e-6))
        self.assertTrue(torch.equal(result[~mask],z[~mask]));self.assertTrue(torch.equal(result[mask],expected[mask]));self.assertEqual(mask.tolist(),[False,False,True,True,True,False,False])
        a,b=gated(z,f+1);self.assertTrue(torch.equal(a[~b],z[~b]));self.assertTrue(torch.equal(mask,b))
