import copy,math,random,unittest
from collections import deque
import torch
from dpa_ctta.r5_update_acceptance.rule import Rule,measurements,quantile


def observation(r=.01,pre=.2,trial=.1):
    n=524288
    regions=[dict(region=name,count=100 if r is not None else 0,sse=0 if r is None else r*100,mean=r) for name in ('OD_fg','OD_bg','OC_fg','OC_bg')]
    return dict(e_pre=dict(sse=pre*n,count=n,mean=pre),e_trial=dict(sse=trial*n,count=n,mean=trial),regions=regions,r=r)


class RuleTests(unittest.TestCase):
    def test_full_grid_regions_empty_partial_and_detach(self):
        q=torch.full((1,2,512,512),.5);views=q.repeat(6,1,1,1,1);pre=q.clone();trial=q.clone()
        x=measurements(pre,q,trial,views);self.assertIsNone(x['r']);self.assertTrue(all(r['mean'] is None for r in x['regions']))
        q[:,0,:100]=.95;views[:,:,0,:100]=.7;trial[:,0,:100]=.6
        x=measurements(pre,q,trial,views);self.assertEqual(x['regions'][0]['count'],51200);self.assertAlmostEqual(x['r'],float((trial.double()-pre.double()).square()[0,0,0,0]))
        self.assertEqual(x['e_pre']['count'],524288)
        for bad in (pre.requires_grad_(),torch.full_like(q,float('nan')),torch.full_like(q,1.1)):
            with self.assertRaises(ValueError):measurements(bad,q,trial,views)
    def test_window_over_128_current_excluded_and_rejections_appended(self):
        rule=Rule('C_VERIFY')
        for i in range(160):
            r=(i+1)/1000;past=list(rule.history);z=rule.decide(observation(r))
            self.assertEqual(z['past_count'],min(i,128));self.assertEqual(z['q90_past'],quantile(past))
            self.assertEqual(z['forced'],i<32)
            if i>=32:self.assertFalse(z['accept'])
            rule.append(r)
        self.assertEqual(list(rule.history),[(i+1)/1000 for i in range(32,160)])
    def test_forcing_order_empty_history_ties_and_tolerance(self):
        r=Rule('C_VERIFY')
        for i in range(32):
            z=r.decide(observation(None));r.append(None);self.assertEqual(z['reason'],'warmup')
        self.assertEqual(r.decide(observation(None))['reason'],'empty_reliable_regions')
        self.assertEqual(r.decide(observation(.01))['reason'],'insufficient_history')
        r.history=deque([.01]*32,maxlen=128)
        self.assertTrue(r.decide(observation(.01,pre=.2,trial=.2))['accept'])
        self.assertTrue(r.decide(observation(.01+5e-13,pre=.2,trial=.2+5e-13))['accept'])
        self.assertFalse(r.decide(observation(.01+2e-12))['accept'])
        self.assertAlmostEqual(quantile([0,1,2,3]),2.7)
    def test_random_one_draw_every_visit_independent_eligibility(self):
        r=Rule('C_RANDOM',.5);expected=random.Random(20260908)
        for i in range(150):
            z=r.decide(observation(.01,trial=.9));self.assertEqual(z['random_draw'],expected.random())
            self.assertEqual(z['accept'],not z['eligible'] or z['random_draw']<.5);r.append(.01)
        for p in (None,float('nan'),-1,2):
            with self.assertRaises(ValueError):Rule('C_RANDOM',p)
    def test_C_half_shadow_and_nonfinite_hard_failure(self):
        for arm in ('C','C_HALF'):
            r=Rule(arm);r.visit=32;r.history=deque([.01]*32,maxlen=128);z=r.decide(observation(.02));self.assertFalse(z['shadow_accept']);self.assertTrue(z['accept'])
        for k in ('r','e_pre'):
            x=observation();x[k]=float('nan') if k=='r' else dict(mean=float('inf'))
            with self.assertRaises(ValueError):Rule('C').decide(x)
