import unittest
from types import SimpleNamespace
import torch
from dpa_ctta.source_correction.method import correction, match_logit_rms
from dpa_ctta.r78_cross_patient.objective import Field


class CorrectionTests(unittest.TestCase):
    def test_zero_identity_and_reference_detached(self):
        torch.set_num_threads(2)
        z=torch.randn(1,2,16,16,requires_grad=True)
        ref=SimpleNamespace(inputs=torch.randn(2,9,8,8),support=torch.zeros(1,2,8,8,dtype=torch.bool),h=torch.ones(1,2,8,8))
        head=Field()
        out=correction(z,ref,head,'R')
        self.assertTrue(torch.equal(out,z))
        out.sum().backward()
        self.assertIsNone(z.grad)
        self.assertIsNotNone(head.layers[-1].bias.grad)

    def test_support_sign_and_budget(self):
        z=torch.zeros(1,2,16,16)
        ref=SimpleNamespace(inputs=torch.randn(2,9,8,8),support=torch.zeros(1,2,8,8,dtype=torch.bool),h=torch.ones(1,2,8,8))
        head=Field()
        with torch.no_grad(): head.layers[-1].bias.fill_(.5)
        r=correction(z,ref,head,'R')
        self.assertLessEqual(float(r.abs().max()),1.)
        self.assertTrue(torch.equal(correction(z,ref,head,'R_BAND'),z))
        self.assertTrue(torch.equal(correction(z,ref,head,'R_NEG'),-r))
        self.assertTrue((r>0).all())

    def test_geometry_removes_only_image_channels(self):
        head=Field()
        with torch.no_grad(): head.layers[-1].weight.fill_(.2)
        x=torch.randn(2,9,8,8);y=x.clone();y[:,:4]+=10
        self.assertTrue(torch.equal(head(x,True),head(y,True)))

    def test_rms_matching_and_bounded_failure(self):
        base=torch.zeros(1,2,4,4);r=torch.full_like(base,.8)
        out,d=match_logit_rms(r,base,torch.full_like(base,.2))
        self.assertTrue(d['rms_match_exact'])
        self.assertAlmostEqual(float(out.square().mean().sqrt()),.2,places=6)
        _,d=match_logit_rms(r,base,torch.full_like(base,2.))
        self.assertFalse(d['rms_match_exact'])
        with self.assertRaises(ValueError): match_logit_rms(base,base,r)


if __name__=='__main__': unittest.main()
