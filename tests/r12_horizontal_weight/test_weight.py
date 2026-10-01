import unittest
import torch
from dpa_ctta.r12_horizontal_weight.view import WeightedHost
from dpa_ctta.r12_horizontal_weight.run import lock,check_lock
class Base:
    def __init__(self):self.method=None;self.segmenter=self;self.visits=0;self.context={'sha256':'toy'}
    def check_frozen(self,boundary=False):pass
    def step(self,x):self.visits+=1;return x[...,0:1].expand_as(x),{}
    def snapshot(self):return {'visits':self.visits}
    def restore(self,s):self.visits=s['visits']
class Checks(unittest.TestCase):
    def test_weight_and_flip_reference(self):
        x=torch.tensor([[[[-4.,1.]]]])
        for name,weight in [('CV_H025',.25),('H_ONLY',1.)]:
            h=WeightedHost(Base(),name);z,_=h.step(x);expected=(1-weight)*torch.sigmoid(torch.tensor(-4.))+weight*torch.sigmoid(torch.tensor(1.))
            self.assertTrue(torch.allclose(z.sigmoid(),expected.expand_as(z),atol=1e-7))
            if name=='H_ONLY':self.assertTrue(torch.equal(z,torch.ones_like(x)))
            s=h.snapshot();h.step(x);h.restore(s);self.assertEqual(h.visits,1)
            h.weights=(1.,)
            with self.assertRaises(ValueError):h.step(x)
    def test_locked_weight_rejection(self):
        c={'bindings':{'checkpoint_sha256':'toy'}};lk=lock(c,'CV_H025');check_lock(lk);lk['payload']['weights']=[.5,.5]
        with self.assertRaises(ValueError):check_lock(lk)
if __name__=='__main__':unittest.main()
