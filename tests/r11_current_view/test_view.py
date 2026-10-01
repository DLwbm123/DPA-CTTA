import unittest
import torch
from dpa_ctta.r11_current_view.view import ViewHost
from dpa_ctta.r11_current_view.run import lock,check_lock
class Base:
    def __init__(self,fn):self.fn=fn;self.method=None;self.segmenter=self;self.context={'sha256':'toy'};self.visits=0
    def check_frozen(self,boundary=False):pass
    def step(self,x):self.visits+=1;return self.fn(x),{}
    def snapshot(self):return {'visits':self.visits}
    def restore(self,s):self.visits=s['visits']
class Checks(unittest.TestCase):
    def test_identity_and_alignment(self):
        x=torch.arange(12,dtype=torch.float32).reshape(1,1,3,4)/3-2
        for name in ('IDENTITY','CV_H2','CV_FLIP4'):
            h=ViewHost(Base(lambda z:z),name);out,_=h.step(x)
            if name=='IDENTITY':self.assertTrue(torch.equal(out,x))
            else:self.assertTrue(torch.allclose(out.sigmoid(),x.sigmoid(),atol=1e-7))
            saved=h.snapshot();h.step(x);h.restore(saved);self.assertEqual(h.visits,1)
    def test_probability_average_and_order(self):
        x=torch.tensor([[[[-4.,1.]]]]);h=ViewHost(Base(lambda z:z[...,0:1].expand_as(z)),'CV_H2')
        out,_=h.step(x);expected=x.sigmoid().mean()
        self.assertTrue(torch.allclose(out.sigmoid(),expected.expand_as(x),atol=1e-7));self.assertFalse(torch.allclose(out.sigmoid(),x.mean().sigmoid().expand_as(x)))
        repeat,_=h.step(x);self.assertTrue(torch.equal(out,repeat))
        h.views=((),)
        with self.assertRaises(ValueError):h.step(x)
    def test_lock_and_snapshot_reject_changes(self):
        c={'bindings':{'checkpoint_sha256':'toy'}};lk=lock(c,'CV_H2');check_lock(lk);lk['payload']['views'].append([-2])
        with self.assertRaises(ValueError):check_lock(lk)
        h=ViewHost(Base(lambda z:z),'CV_H2');s=h.snapshot();s['base']['visits']=1
        with self.assertRaises(ValueError):h.restore(s)
if __name__=='__main__':unittest.main()
