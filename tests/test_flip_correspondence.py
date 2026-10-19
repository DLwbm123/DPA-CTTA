import unittest
import torch
from torch import nn
from dpa_ctta.srg_ctta.flip_correspondence import Host,aligned_teacher,bernoulli_kl

class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,16,3,padding=1);self.bn=nn.BatchNorm2d(16);self.head=nn.Conv2d(16,2,1)
    def forward(self,x):return self.head(self.bn(self.conv(x)).relu())

class Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(23);torch.set_num_threads(2)
    def test_alignment_and_mass(self):
        z=torch.randn(1,2,8,8)
        self.assertTrue(torch.equal(aligned_teacher(z),z.flip(-1)))
        self.assertTrue(torch.equal(aligned_teacher(z,True),z.flip(-1).roll(5,-1)))
        torch.testing.assert_close(aligned_teacher(z,True).sigmoid().sum((-2,-1)),aligned_teacher(z).sigmoid().sum((-2,-1)))
    def test_bernoulli_not_softmax(self):
        z=torch.tensor([[[[.4]],[[1.2]]]],requires_grad=True);t=torch.tensor([[[[-.3]],[[.7]]]],requires_grad=True)
        loss=bernoulli_kl(z,t);loss.backward()
        torch.testing.assert_close(z.grad,(z.detach().sigmoid()-t.detach().sigmoid())/2)
        self.assertIsNone(t.grad)
        self.assertGreater(float(bernoulli_kl(z+.3,t)),0)
    def test_identity_and_shape(self):
        z=torch.randn(1,2,7,9,requires_grad=True);l=bernoulli_kl(z,z);l.backward()
        self.assertLess(abs(float(l)),1e-14);self.assertLess(float(z.grad.abs().max()),1e-8)
        with self.assertRaises(ValueError):bernoulli_kl(z[:,:1],z[:,:1])
    def test_real_step_frozen_source_and_reset(self):
        b=Toy().eval().requires_grad_(False);x=torch.randn(1,3,12,12)
        saved={k:v.clone() for k,v in b.state_dict().items()};h=Host(b,'FLIP')
        pre,post,_,_,d=h.step(x,x,x,b(x),1)
        self.assertTrue(d['steps'][0]['norm_matched']);self.assertGreater(d['steps'][0]['affine_gradient_norm'],1e-12)
        self.assertFalse(torch.equal(pre,post));self.assertEqual(d['optimizer_age'],1)
        for k,v in b.state_dict().items():self.assertTrue(torch.equal(v,saved[k]))
        for p in b.parameters():self.assertIsNone(p.grad)
        h.reset_optimizer();torch.testing.assert_close(h.forward(x),post,rtol=0,atol=0)
        h.reset();torch.testing.assert_close(h.forward(x),pre,rtol=0,atol=0)
        p2,q2,_,_,_=h.step(x,x,x,b(x),1);self.assertTrue(torch.equal(pre,p2) and torch.equal(post,q2))
    def test_controls_and_adam(self):
        b=Toy().eval().requires_grad_(False);x=torch.randn(1,3,12,12)
        losses=[]
        for kind in ('FLIP','ROLL','SOURCE_FLIP','ENT_NORM'):
            h=Host(b,kind);_,_,_,_,d=h.step(x,x,x,b(x),3);losses.append(d['steps'][0]['loss'])
            self.assertTrue(d['steps'][0]['norm_matched'])
        self.assertAlmostEqual(losses[0],losses[2],12);self.assertNotEqual(losses[0],losses[1])
    def test_config_frozen(self):
        import json
        from pathlib import Path
        from dpa_ctta.srg_ctta.flip_run import validate
        c=json.loads((Path(__file__).parents[1]/'configs/flip_correspondence_ctta_v1.json').read_text());validate(c)
        c['design_version']='SPATIAL_GRADIENT_V1'
        with self.assertRaises(AssertionError):validate(c)

if __name__=='__main__':unittest.main()
