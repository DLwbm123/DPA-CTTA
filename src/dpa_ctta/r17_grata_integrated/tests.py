"""CPU invariants for the added stopped residual objective."""
import unittest
import torch
from .method import correction_loss
from ..r16_evidence_correction.methods import ResidualHead,normalize_input

class Objective(unittest.TestCase):
    def test_zero_and_empty_exact(self):
        x=torch.randn(1,2,8,8,requires_grad=True);q=torch.rand_like(x);m=torch.ones_like(x,dtype=torch.bool)
        loss=correction_loss(x,q,q,m);g=torch.autograd.grad(loss,x)[0]
        self.assertEqual(float(loss),0);self.assertTrue(torch.equal(g,torch.zeros_like(g)))
        self.assertEqual(float(correction_loss(x,q,q,m&False)),0)
    def test_residual_gradient(self):
        x=torch.randn(1,2,8,8,requires_grad=True);a=torch.rand_like(x);b=torch.rand_like(x);m=torch.rand_like(x)>.7
        grad=torch.autograd.grad(correction_loss(x,a,b,m),x)[0];expected=torch.where(m,(a-b)/m.sum(),0)
        torch.testing.assert_close(grad,expected,rtol=1e-5,atol=1e-8)
        self.assertTrue(torch.equal(grad[~m],torch.zeros_like(grad[~m])))
    def test_capacity_and_ablated_channels(self):
        self.assertEqual(sum(p.numel() for p in ResidualHead().parameters()),9026)
        x=torch.randn(1,269,4,4);s=dict(mean=torch.zeros(1,269,1,1),std=torch.ones(1,269,1,1))
        a=normalize_input(x,s,'D_LOGIT');b=normalize_input(x,s,'D_CONTEXT')
        self.assertTrue(torch.equal(a[:,:8],b[:,:8]));self.assertEqual(torch.count_nonzero(a[:,8:]).item(),0)

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Objective))
    raise SystemExit(0 if result.wasSuccessful() else 1)
