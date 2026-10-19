import json
from pathlib import Path
import unittest
import numpy as np
import torch
from torch import nn
from dpa_ctta.srg_ctta.image_replay import Host,Memory,descriptor,distance,KINDS
from dpa_ctta.srg_ctta.replay_run import validate,contrast,JOBS


class Toy(nn.Module):
    def __init__(self):
        super().__init__();self.bn=nn.BatchNorm2d(3);self.head=nn.Conv2d(3,2,1)
        with torch.no_grad():
            self.head.weight.fill_(.6);self.head.bias.copy_(torch.tensor([-.1,.1]))
    def forward(self,x):return self.head(self.bn(x))


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(20261018)
        self.x=torch.linspace(-2,2,32).view(1,1,1,32).expand(1,3,32,32).clone()
        self.base=Toy().eval().requires_grad_(False)

    def test_binary_distance_and_strata(self):
        z=self.base(self.x);d,n=descriptor(z)
        self.assertEqual(d.shape,(6,16));self.assertEqual(sum(n),2*32*32)
        self.assertTrue(all(i>0 for i in n));self.assertEqual(distance(d,d),0.)
        self.assertAlmostEqual(distance(d,1-d),distance(1-d,d),14)
        self.assertAlmostEqual(distance(torch.zeros(6,16,dtype=torch.float64),torch.ones(6,16,dtype=torch.float64)),1.,14)
        # Probability quantiles inside each region survive replication of rows;
        # the seven-pixel spatial band across columns is unchanged.
        dd,nn=descriptor(z.repeat_interleave(3,-2))
        torch.testing.assert_close(d,dd,rtol=0,atol=1e-14)
        self.assertEqual(nn,[3*v for v in n])

    def test_empty_and_nonfinite_are_stops(self):
        with self.assertRaises(ValueError):descriptor(torch.ones(1,2,32,32)*5)
        z=self.base(self.x).clone();z[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):descriptor(z)

    def test_diverse_pair_and_fifo(self):
        d=lambda v:torch.ones(6,16,dtype=torch.float64)*v
        a=Memory('DIVERSITY');b=Memory('FIFO')
        for i,v in enumerate((.1,.9,.85)):
            for bank in (a,b):bank.admit(torch.tensor([i]),d(v))
        self.assertEqual([v['uid'] for v in a.entries],[1,2])
        self.assertEqual([v['uid'] for v in b.entries],[2,3])
        self.assertGreater(distance(*[v['descriptor'] for v in a.entries]),distance(*[v['descriptor'] for v in b.entries]))
        # Exact duplicate descriptors use the frozen newest-ID tie rule.
        a=Memory('DIVERSITY')
        for i in range(4):a.admit(torch.tensor([i]),d(.5))
        self.assertEqual([v['uid'] for v in a.entries],[3,4])

    def test_random_reproducibility_and_ownership(self):
        a=Memory('RANDOM_MEMORY');b=Memory('RANDOM_MEMORY')
        for i in range(30):
            x=torch.tensor([float(i)]);d=torch.ones(6,16,dtype=torch.float64)*.2
            for bank in (a,b):bank.admit(x,d)
            x.fill_(-9);d.fill_(.9)
        self.assertTrue(a.exact(b));self.assertEqual(len(a.entries),2)
        self.assertTrue(all(v['x'].item()>=0 and v['descriptor'][0,0]==.2 for v in a.entries))

    def test_past_only_batch_and_fixed_slots(self):
        bank=Memory('FIFO');d=torch.ones(6,16,dtype=torch.float64)*.2
        self.assertEqual(bank.batch(self.x)[1],[])
        bank.admit(self.x,d);batch,ids=bank.batch(self.x+1)
        self.assertEqual(len(batch),3);self.assertEqual(ids,[1])
        self.assertTrue(torch.equal(batch[1],batch[2]));self.assertFalse(torch.equal(batch[0],batch[1]))
        bank.admit(self.x+1,d);self.assertEqual(bank.batch(self.x+2)[1],[1,2])

    def test_first_step_equal_frozen_and_real_function_change(self):
        frozen={n:v.clone() for n,v in self.base.state_dict().items()};out=[]
        with torch.no_grad():z0=self.base(self.x)
        for kind in KINDS:
            h=Host(self.base,kind);a,b,_,_,d=h.step(self.x,self.x,self.x,z0,1)
            self.assertTrue(d['steps'][0]['norm_matched']);self.assertEqual(d['loss_backward_calls'],3)
            self.assertEqual(d['optimizer_age'],1);self.assertFalse(torch.equal(a,b));out.append((a,b))
            self.assertEqual(d['memory_before'],0);self.assertEqual(d['replayed_arrival_ids'],[])
        for a,b in out[1:]:
            self.assertTrue(torch.equal(a,out[0][0]));self.assertTrue(torch.equal(b,out[0][1]))
        for n,v in self.base.state_dict().items():self.assertTrue(torch.equal(v,frozen[n]))
        self.assertTrue(all(p.grad is None for p in self.base.parameters()))

    def test_fresh_replay_and_full_state_repeat(self):
        a=Host(self.base,'DIVERSITY');b=Host(self.base,'DIVERSITY')
        for i in range(5):
            x=self.x+(.07*i);z0=self.base(x)
            for h in (a,b):h.reset_optimizer()
            aa=a.step(x,x,x,z0,1);bb=b.step(x,x,x,z0,1)
            self.assertTrue(torch.equal(aa[0],bb[0]) and torch.equal(aa[1],bb[1]))
            self.assertTrue(a.repeat_state_equal(b));self.assertEqual(aa[-1]['memory_seen_before'],i)
            self.assertEqual(aa[-1]['memory_before'],min(i,2));self.assertEqual(aa[-1]['optimizer_age'],1)
        a.reset();self.assertEqual(a.memory.seen,0);self.assertFalse(a.memory.entries)
        self.assertTrue(torch.equal(a.forward(self.x),self.base(self.x)))

    def test_reset_is_no_replay_control(self):
        a=Host(self.base,'DIVERSITY');b=Host(self.base,'CURRENT')
        for i in range(3):
            x=self.x+.03*i;z0=self.base(x);a.reset();b.reset()
            aa=a.step(x,x,x,z0,1);bb=b.step(x,x,x,z0,1)
            self.assertTrue(torch.equal(aa[0],bb[0]) and torch.equal(aa[1],bb[1]))

    def test_frozen_matrix_and_scoring_shapes(self):
        c=json.loads((Path(__file__).parents[1]/'configs/gotta_image_replay_v1.json').read_text());validate(c)
        self.assertEqual(len(JOBS),10)
        for shape in ((8,5,2,2),(8,5,2),(8,2,2),(8,2)):
            r=contrast('synthetic',np.ones(shape));self.assertEqual(r['delta_pp'],1.);self.assertEqual(r['ci95_pp'],[1.,1.])
        with self.assertRaises(ValueError):contrast('bad',np.ones((7,5,2)))
        c['memory_capacity']=3
        with self.assertRaises(AssertionError):validate(c)


if __name__=='__main__':unittest.main()
