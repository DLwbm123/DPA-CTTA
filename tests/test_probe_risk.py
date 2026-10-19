import json
from pathlib import Path
import unittest
import numpy as np
import torch
from dpa_ctta.srg_ctta.probe_risk import signals,probe_views,Host,js
from dpa_ctta.srg_ctta.spatial_gradient import Host as Original
from dpa_ctta.srg_ctta.probe_run import evidence,validate


class Checks(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_zero_intervention_and_binary_js(self):
        gen=torch.Generator().manual_seed(3);p=.1+.8*torch.rand((4,1,2,5,5),generator=gen,dtype=torch.float64)
        s=p.mean(0);s[...,0,0]=.9;s[...,1,1]=.1
        out=signals(p,p,p,s)
        self.assertEqual(out['utility'],[0.,0.]);self.assertTrue(torch.all(js(p,p)==0))
        for k in ('utility','pre_disagreement','consistency_gain'):self.assertTrue(np.isfinite(out[k]).all())

    def test_true_consistency_and_anchor_penalty(self):
        p=torch.tensor([.2,.8,.3,.7],dtype=torch.float64).reshape(4,1,1,1,1).expand(4,1,2,2,2)
        after=torch.ones_like(p)*.5;source=torch.tensor([.1,.9,.1,.9]).reshape(1,1,2,2).expand(1,2,2,2)
        d=signals(p,after,after,source)
        self.assertTrue(all(x>0 for x in d['utility']))
        d=signals(after,p,after,source)
        self.assertTrue(all(x<0 for x in d['utility']))
        d=signals(after,after+.1,after,source)
        self.assertTrue(all(x<0 for x in d['utility']))

    def test_balancing_and_flip_correspondence(self):
        x=torch.rand(1,3,5,5);a=probe_views(x,81);b=probe_views(x,81);c=probe_views(x,82)
        self.assertTrue(all(torch.equal(u,v) and f==g for (u,f),(v,g) in zip(a,b)))
        self.assertFalse(all(torch.equal(u,v) for (u,_),(v,_) in zip(a,c)))
        for y,flip in a:self.assertEqual(y.shape,x.shape)
        p=torch.ones(4,1,2,2,2,dtype=torch.float64)*.2;p[1]=.8;s=torch.tensor([.1,.9,.1,.9]).reshape(1,1,2,2).expand(1,2,2,2)
        out=signals(p,p,p,s);perm=torch.tensor([2,0,3,1])
        q=lambda a:a.flatten(-2)[...,perm].reshape_as(a)
        self.assertEqual(out,signals(q(p),q(p),q(p),q(s)))

    def test_probes_do_not_change_candidate_or_adam(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.bn=torch.nn.BatchNorm2d(3);self.head=torch.nn.Conv2d(3,2,1)
            def forward(self,x):return (self.head(self.bn(x)),)
        torch.manual_seed(3);base=Model().eval().requires_grad_(False)
        with torch.no_grad():
            base.head.weight.fill_(1.);base.head.bias.fill_(-1.5)
        a=Host(base,'ENT_NORM');b=Original(base,'ENT_NORM');x=torch.rand(1,3,8,8)
        with torch.no_grad():z=base(x)[0]
        out=a.step(x,x,x,z,83);old=b.step(x,x,x,z,83)
        self.assertTrue(torch.equal(out[0],old[0]) and torch.equal(out[1],old[1]))
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(a.params,b.params)))
        for p,q in zip(a.params,b.params):
            for key in ('step','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(a.opt.state[p][key],b.opt.state[q][key]))
        self.assertEqual(out[-1]['forwards'],14)
        self.assertEqual(out[-1]['optimizer_age'],1)
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertLessEqual(abs(out[-1]['steps'][0]['actual_step_norm']-.01),5e-7)

    def test_paired_bootstrap_and_degenerate_scores(self):
        groups=[]
        for i in range(8):
            rows=[]
            for j in range(4):
                y=(j-2)/5
                rows.append(dict(channel=j%2,gain=y,utility=y,pre_entropy=-y,pre_disagreement=-y,negative_extra_drift=-y,negative_function_change=-y,consistency_gain=-y,full_utility=-y,negative_pre_entropy=-y,negative_pre_disagreement=-y,negative_pre_anchor_drift=-y))
            groups.append(rows)
        out,quota=evidence(groups,20);self.assertAlmostEqual(out['primary'],2.);self.assertEqual(out['harm_auroc']['utility'],1.)
        self.assertEqual(out['n'],32);self.assertEqual(len(quota),22)
        for group in groups:
            for row in group:row['utility']=0.
        self.assertIsNone(evidence(groups,20)[0]['primary'])

    def test_frozen_config_and_scope_rejection(self):
        c=json.loads((Path(__file__).parents[1]/'configs/probe_risk_qualification_v1.json').read_text());validate(c)
        c['rdc']=True
        with self.assertRaises(AssertionError):validate(c)

if __name__=='__main__':unittest.main()
