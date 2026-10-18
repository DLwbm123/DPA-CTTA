import copy
import unittest
import torch
from torch import nn
from dpa_ctta.srg_ctta.method import Host,entropy,strata,select,objective,reduce_region,l2,normalize_parameters


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,4,3,padding=1);self.bn=nn.BatchNorm2d(4);self.head=nn.Conv2d(4,2,1)
    def forward(self,x): return self.head(self.bn(self.conv(x)))


class RoutingTests(unittest.TestCase):
    def setUp(self): torch.set_num_threads(2);torch.manual_seed(20261019)

    def test_tied_fp32_coordinates_need_distributed_rounding(self):
        before=[torch.ones(19136)];params=[nn.Parameter(before[0]+1e-4)]
        raw,actual,cosine=normalize_parameters(params,before)
        self.assertLessEqual(abs(actual-.01),5e-7)
        self.assertGreaterEqual(cosine,.999999)
        self.assertGreater(int(torch.unique(params[0].detach()-before[0]).numel()),1)

    def test_actual_fp32_radius_and_direction(self):
        before=[torch.randn(10000)*3,torch.randn(9136)*.1]
        params=[nn.Parameter(v.clone()) for v in before]
        with torch.no_grad():
            for p in params:p.add_(torch.randn_like(p)*1e-4)
        raw,actual,cosine=normalize_parameters(params,before)
        self.assertGreater(raw,0);self.assertLessEqual(abs(actual-.01),5e-7);self.assertGreaterEqual(cosine,.999999)
        self.assertEqual(normalize_parameters([nn.Parameter(torch.zeros(5))],[torch.zeros(5)]),(0.,0.,None))

    def test_batched_reductions_preserve_value_and_gradient(self):
        v=torch.randn(1,2,32,32,requires_grad=True);regions=strata(torch.rand_like(v));mask=torch.rand_like(v)>.5
        terms=[]
        for c in range(2):
            for region in regions:
                keep=mask[0,c]&region[0,c]
                if keep.any():terms.append(v[0,c][keep].mean())
        old=torch.stack(terms).mean();new=reduce_region(v,mask,regions,True)
        self.assertTrue(torch.allclose(old,new,atol=1e-7))
        a,=torch.autograd.grad(old,v,retain_graph=True);b,=torch.autograd.grad(new,v)
        self.assertTrue(torch.allclose(a,b,atol=1e-7))
        self.assertAlmostEqual(l2([v,v[:,:,:4]]),(float(v.double().square().sum())+float(v[:,:,:4].double().square().sum()))**.5,places=10)

    def test_independent_bernoulli_entropy(self):
        p=torch.tensor([[[[.5]],[[.5]]]])
        self.assertTrue(torch.allclose(entropy(p),torch.ones_like(p)))
        self.assertLess(float(entropy(torch.tensor([0.,1.])).max()),.00003)

    def test_regions_and_exact_random_quota(self):
        p=torch.zeros(1,2,32,32);p[:,:,8:24,8:24]=1
        regions=strata(p);self.assertTrue(torch.equal(sum(r.long() for r in regions),torch.ones_like(p,dtype=torch.long)))
        u=torch.rand_like(p);a=select(u,regions,'STRAT',5);b=select(u,regions,'RANDOM',5)
        for region in regions:
            for c in range(2):
                self.assertEqual(int((a&region)[0,c].sum()),int((b&region)[0,c].sum()))
        self.assertTrue(torch.equal(b,select(u,regions,'RANDOM',5)))
        self.assertFalse(a.requires_grad)

    def test_anchor_zero_then_active(self):
        z0=torch.randn(1,2,16,16);z=z0.clone().requires_grad_();p=z0.sigmoid();mask=torch.zeros_like(p,dtype=torch.bool);mask[:,:,:8]=True;regions=strata(p)
        _,terms=objective(z,p,p,z0,mask,regions,'STRAT')
        g,=torch.autograd.grad(terms['anchor'],z)
        self.assertLess(float(g.abs().max()),1e-7)
        z=(z0+.1).requires_grad_();_,terms=objective(z,p,p,z0,mask,regions,'STRAT')
        g,=torch.autograd.grad(terms['anchor'],z)
        self.assertGreater(float(g.norm()),0)

    def test_real_affine_update_reset_and_retention(self):
        base=Toy().eval().requires_grad_(False);host=Host(base,'STRAT');x=torch.rand(1,3,32,32)
        with torch.no_grad():z0=base(x)
        before={n:b.clone() for n,b in host.model.named_buffers()}
        pre,post,m,g,d=host.step(x,x*.95,x*1.05,z0,7)
        self.assertTrue(torch.equal(pre,z0));self.assertEqual(d['optimizer_age'],2)
        self.assertTrue(all(v['norm_matched'] for v in d['steps']))
        self.assertGreater(float((post-pre).norm()),0)
        _,_,_,_,d2=host.step(x,x*.95,x*1.05,z0,7);self.assertEqual(d2['optimizer_age'],4)
        self.assertTrue(all(torch.equal(b,dict(host.model.named_buffers())[n]) for n,b in before.items()))
        host.reset();p2,_,_,_,d3=host.step(x,x*.95,x*1.05,z0,7)
        self.assertTrue(torch.equal(pre,p2));self.assertEqual(d3['optimizer_age'],2)

    def test_optimizer_reset_keeps_adapted_model_and_discards_moments(self):
        base=Toy().eval().requires_grad_(False);host=Host(base,'STRAT');x=torch.rand(1,3,32,32)
        with torch.no_grad():z0=base(x)
        _,post,_,_,_=host.step(x,x*.95,x*1.05,z0,7)
        params=[p.detach().clone() for p in host.params]
        self.assertEqual(host.segment_steps,2);self.assertTrue(host.opt.state)
        host.reset_optimizer()
        self.assertFalse(host.opt.state);self.assertEqual(host.segment_steps,0)
        self.assertTrue(all(torch.equal(p,v) for p,v in zip(host.params,params)))
        pre,_,_,_,d=host.step(x,x*.95,x*1.05,z0,7)
        self.assertTrue(torch.equal(pre,post));self.assertEqual(d['optimizer_age'],2)
        self.assertEqual(host.total_steps,4)

    def test_no_anchor_and_balanced_full_control(self):
        z0=torch.randn(1,2,32,32);z=(z0+.2).requires_grad_();p=z0.sigmoid();regs=strata(p);mask=select(entropy(p),regs,'STRAT',5)
        a,t=objective(z,p,p,z0,mask,regs,'STRAT');b,_=objective(z,p,p,z0,mask,regs,'NOANCHOR')
        self.assertTrue(torch.allclose(a-b,t['anchor'],atol=1e-7))
        self.assertTrue(select(entropy(p),regs,'ALL',5).all())


if __name__=='__main__':unittest.main()
