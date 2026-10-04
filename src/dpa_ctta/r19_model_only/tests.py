"""Synthetic-only checks; no data, labels, source assets or checkpoint required."""
import copy,unittest
from unittest.mock import patch
import torch
from .method import Host,snapshot,restore_state,readonly,equal,regions,divergence,accepts,reliable
from ..b1_host import reference_step
from ..hosts.vptta import model_input_from_pixels

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.conv=torch.nn.Conv2d(3,2,1);self.bn=torch.nn.BatchNorm2d(2)
    def forward(self,x):
        y=self.bn(self.conv(x));return y,[y]

class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);self.model=Toy();self.x=torch.rand(1,3,512,512);self.y=torch.rand_like(self.x)
    def host(self,arm='G',diagnostics=False):return Host(None,arm,'synthetic',device='cpu',model=copy.deepcopy(self.model),diagnostics=diagnostics)
    def test_native_and_readonly_parity(self):
        h=self.host();s=snapshot(h.native);readonly(h.native,self.x);self.assertTrue(equal(s,snapshot(h.native)))
        out,_=h.step(self.x);end=snapshot(h.native);restore_state(h.native,s)
        from ..host_diagnostic import restore
        restore(h.native.rng);expected=reference_step(h.native.model,h.native.opt,'G',model_input_from_pixels(self.x,'fundus'))
        self.assertTrue(torch.equal(out,expected.cpu()));self.assertTrue(equal(end['parameters'],snapshot(h.native)['parameters']));self.assertTrue(equal(end['adam'],snapshot(h.native)['adam']));self.assertTrue(equal(end['rng'],snapshot(h.native)['rng']));h.close()
    def test_restore_next_output_and_rejection(self):
        h=self.host('G_VAL');h.step(self.x);s=h.snapshot();out,_=h.step(self.y);end=h.snapshot();h.restore(s);again,_=h.step(self.y)
        self.assertTrue(torch.equal(out,again));self.assertTrue(equal(end,h.snapshot()))
        n=h.native;before=snapshot(n);n.step(self.x);restore_state(n,before);self.assertTrue(equal(before,snapshot(n)));h.close()
    def test_forced_rejection_returns_pre_state(self):
        h=self.host('G_VAL');pre=snapshot(h.native);z=readonly(h.native,self.x)
        with patch('dpa_ctta.r19_model_only.method.accepts',return_value=(False,'validation')):out,t=h.step(self.x)
        self.assertFalse(t['accepted']);self.assertTrue(torch.equal(out,z));self.assertEqual(h.visits,1)
        after=snapshot(h.native)
        for k in ('parameters','buffers','gradients','adam','grata','steps'):self.assertTrue(equal(pre[k],after[k]),k)
        self.assertFalse(equal(pre['native_rng'],after['native_rng']));h.close()
    def test_memory_expiry_by_arrival_and_no_reject_write(self):
        h=self.host('G_MEM');h.visits=64;h.memory.append({'time':1})
        with patch('dpa_ctta.r19_model_only.method.accepts',return_value=(False,'validation')):out,t=h.step(self.x)
        self.assertEqual(t['expired'],1);self.assertEqual(t['writes'],0);self.assertTrue(t['memory_empty']);self.assertEqual(len(h.memory),0);h.close()
    def test_empty_memory_matches_validation(self):
        a=self.host('G_VAL');b=self.host('G_MEM');za,ta=a.step(self.x);zb,tb=b.step(self.x)
        self.assertTrue(torch.equal(za,zb));self.assertEqual(ta['accepted'],tb['accepted']);a.close();b.close()
    def test_diagnostic_does_not_change_main(self):
        a=self.host();b=self.host(diagnostics=True);za,_=a.step(self.x);zb,t=b.step(self.x)
        self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(snapshot(a.native),snapshot(b.native)));self.assertIsNotNone(t['diagnostic'])
        ya,_=a.step(self.y);yb,_=b.step(self.y);self.assertTrue(torch.equal(ya,yb));a.close();b.close()
    def test_proxy_and_support(self):
        p=torch.full((1,2,32,32),.1);p[:,:,8:24,8:24]=.9;z=torch.logit(p);r=regions(z)
        self.assertIsNotNone(r);self.assertTrue(torch.all(sum(a.int() for a in r)==1));self.assertEqual(float(divergence(p,p).max()),0.)
        self.assertFalse(accepts(z,z,[.1,.1],[.1,.1])[0]);self.assertTrue(accepts(z,z,[.1,.1],[.09,.09])[0]);self.assertFalse(accepts(z,z,[.1,.1],[.08,.11])[0]);self.assertIsNone(regions(torch.full_like(z,-10)));self.assertIsNotNone(reliable(p))
    def test_half_changes_lr_only(self):
        a=self.host();b=self.host('G_HALF');_,da=a.step(self.x);_,db=b.step(self.x)
        self.assertAlmostEqual(db['native']['lr'],.5*da['native']['lr']);self.assertTrue(equal(a.native.base.state_dict()['state'],b.native.base.state_dict()['state']));a.close();b.close()

if __name__=='__main__':unittest.main()
