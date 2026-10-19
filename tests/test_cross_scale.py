import unittest,json,importlib.util
from pathlib import Path
import torch
import numpy as np
from dpa_ctta.srg_ctta.cross_scale import supporters,pool,anchored_output,relative_output,mean_check
from dpa_ctta.srg_ctta.single_bn import moments
from dpa_ctta.srg_ctta.cross_scale_run import validate,PRIMARY

class Tests(unittest.TestCase):
    def test_support_excludes_all_query_group(self):
        for i in range(8):
            j=supporters(i);self.assertEqual(len(j),4);self.assertNotIn(i,j)
            self.assertEqual(j,(4,5,6,7) if i<4 else (0,1,2,3))
    def test_pool_total_variance_includes_between(self):
        m=torch.tensor([[0.,1.],[2.,1.],[4.,1.],[6.,1.]],dtype=torch.float64);v=torch.ones_like(m)*2
        mu,var,within,between=pool(m,v)
        torch.testing.assert_close(mu,torch.tensor([3.,1.],dtype=torch.float64));torch.testing.assert_close(var,torch.tensor([7.,2.],dtype=torch.float64))
        torch.testing.assert_close(within+between,var)
        for repeat in (m[:3],m[:2]):
            with self.assertRaises(ValueError):pool(repeat,v[:len(repeat)])
    def fixture(self):
        bn=torch.nn.BatchNorm2d(2).eval()
        with torch.no_grad():
            bn.running_mean.copy_(torch.tensor([.4,-.3]));bn.running_var.copy_(torch.tensor([.8,1.2]));bn.weight.copy_(torch.tensor([1.,-.7]))
        h=torch.arange(32,dtype=torch.float32).reshape(1,2,4,4)/10
        return bn,h
    def test_query_anchor_is_distinct_from_history_mean(self):
        bn,h=self.fixture();v,m=moments(h);vh=v*4;ys=bn(h)
        yy=anchored_output(bn,h,m,vh);self.assertLess(mean_check(ys,yy),1e-5)
        mh=m+2;ss=(bn.running_var.double()+bn.eps).sqrt();sh=(vh+bn.eps).sqrt()
        mu_bad=mh-sh/ss*(mh-bn.running_mean.double())
        bad=torch.nn.functional.batch_norm(h,mu_bad.to(h),vh.to(h),bn.weight,bn.bias,False,0,bn.eps)
        self.assertGreater(float((bad.mean((0,2,3))-ys.mean((0,2,3))).abs().max()),.1)
    def test_relative_identity_and_nontrivial_scale(self):
        bn,h=self.fixture();v,m=moments(h);ys=bn(h)
        self.assertTrue(torch.equal(relative_output(ys,v,v,bn.eps),ys))
        changed=relative_output(ys,v*2,v,bn.eps);self.assertFalse(torch.equal(changed,ys));self.assertLess(mean_check(ys,changed),1e-5)
        mean=ys.double().mean((0,2,3),keepdim=True)
        expect=mean+((v*2+bn.eps)/(v+bn.eps)).sqrt()[None,:,None,None]*(ys.double()-mean)
        torch.testing.assert_close(changed,expect.float())
    def test_zero_and_nonfinite_stat_rules(self):
        bn,h=self.fixture();v,m=moments(h)
        self.assertTrue(torch.isfinite(anchored_output(bn,h,m,torch.zeros_like(v))).all())
        with self.assertRaises(ValueError):anchored_output(bn,h,m,-v)
        with self.assertRaises(ValueError):relative_output(bn(h),v,torch.full_like(v,float('nan')),bn.eps)
    def test_frozen_config(self):
        c=json.loads((Path(__file__).parents[1]/'configs/cross_content_scale_a01.json').read_text());validate(c)
        self.assertEqual(PRIMARY,'CROSS_SCALE-SOURCE:OC_DICE');c['support_groups']=[[0,2,4,6],[1,3,5,7]]
        with self.assertRaises(AssertionError):validate(c)
    def test_robustness_reduction_and_leave_one(self):
        p=Path(__file__).parents[1]/'reports/cross-content-scale-a01/a0_robustness.py'
        spec=importlib.util.spec_from_file_location('robust',p);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        rows=[dict(case=i,condition=k,arm=a,metrics=[dict(dice_percent=80.+d),dict(dice_percent=80.+d)]) for i in range(8) for k in range(4) for a,d in [('SOURCE',0),('VAR_ONLY',-2),('RB_BN',-1),('ADABN',-3)]]
        pub,private=mod.analyze(rows);self.assertEqual(pub[0]['full_mean_pp'],1.);self.assertEqual(pub[0]['loo_mean_min_pp'],1.)
        self.assertEqual(pub[0]['content_positive'],8);self.assertEqual(pub[1]['content_negative'],8);self.assertEqual(len(private),3)

if __name__=='__main__':unittest.main()
