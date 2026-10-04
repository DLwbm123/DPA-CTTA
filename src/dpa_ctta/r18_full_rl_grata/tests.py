import unittest
import torch
from .method import fixed_actor,set_bn,bn_state
from .source import EPISODES
from ..r10_use_write_rl.math import Memory,use_and_write

class Qualification(unittest.TestCase):
    def test_all_actions_fixed_and_memory_retained(self):
        a=fixed_actor();x=a(torch.randn(193));y=a(torch.randn(193)*100)
        self.assertTrue(torch.equal(x,y));self.assertAlmostEqual(float(x[0].sigmoid()),.8,places=6)
        self.assertTrue(torch.equal(x[1:9],torch.zeros(8)));self.assertEqual(float(x[9].sigmoid()),.5)
        proposed=torch.arange(64,dtype=torch.float64)/64;d=torch.ones(32,dtype=torch.float64);scale=torch.ones(64,dtype=torch.float64)
        u,m=use_and_write(proposed,d,Memory.zero(),scale,x);_,n=use_and_write(proposed,d,m,scale,x)
        self.assertTrue(torch.allclose(u,.8*proposed));self.assertTrue(torch.equal(m.m,.5*u));self.assertEqual(float(n.h),.75)
        self.assertTrue(all(not p.requires_grad for p in a.parameters()))
    def test_only_bn_state_is_copied(self):
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2));before=model[0].weight.detach().clone();state={'1.weight':torch.full((2,),3.),'1.bias':torch.full((2,),4.)}
        set_bn(model,state);self.assertTrue(torch.equal(model[0].weight,before));self.assertTrue(torch.equal(model[1].weight,state['1.weight']))
    def test_finite_balanced_source_pool(self):
        self.assertEqual(len(set(EPISODES)),16);self.assertEqual(sorted(x//16 for x in EPISODES),[0]*4+[1]*4+[2]*4+[3]*4)

if __name__=='__main__':unittest.main()
