import unittest
import numpy as np
from dpa_ctta.srg_ctta.e8_math import window,partitions,contrast,stratify,NAMES

class E8Tests(unittest.TestCase):
    def test_window_edges_and_full_query(self):
        self.assertEqual(window([[0,0]]),([0,0],True));self.assertEqual(window([[511,511]]),([256,256],True))
        self.assertEqual(window([[255,255],[256,256]]),([128,128],True));self.assertFalse(window([[0,0],[511,511]])[1])
    def test_sum_delta_matches_bruteforce(self):
        rng=np.random.default_rng(19);v=rng.uniform(.1,2,(8,8,5));b=np.zeros((8,8),bool);b[:4]=True;s=b.copy();s[3,3]=False;s[4,4]=True;edit=b!=s
        got,reason,n=partitions({'Q':v},b,s,edit);u=v/np.linalg.norm(v,axis=-1)[:,:,None]
        want=contrast(u[s].sum(0),u[~s].sum(0))-contrast(u[b].sum(0),u[~b].sum(0))
        self.assertEqual(reason,'COMMON');self.assertEqual(n,64);self.assertAlmostEqual(got['Q'],want,places=14)
        rev,_,_=partitions({'Q':v},~b,~s,edit);self.assertAlmostEqual(rev['Q'],got['Q'],places=14)
    def test_zero_nonfinite_and_three_pixels(self):
        v=np.ones((8,8,2));b=np.zeros((8,8),bool);b[:4]=True;s=b.copy();s[3,3]=False;edit=b!=s
        bad=v.copy();bad[3,3]=0;self.assertEqual(partitions({'Q':bad},b,s,edit)[1],'ZERO_QUERY')
        bad=v.copy();bad[7,7]=0;self.assertEqual(partitions({'Q':bad},b,s,edit)[2],63)
        bad=v.copy();bad[7,7]=np.nan
        with self.assertRaises(ValueError):partitions({'Q':bad},b,s,edit)
        small=np.zeros_like(b);small[0,:3]=True;ss=small.copy();ss[0,0]=False
        self.assertEqual(partitions({'Q':v},small,ss,small!=ss)[1],'SHORT_PARTITION')
    def test_constant_and_zero_edit(self):
        v=np.ones((8,8,4));b=np.zeros((8,8),bool);b[:4]=True;s=b.copy();s[3,3]=False
        self.assertAlmostEqual(partitions({'Q':v},b,s,b!=s)[0]['Q'],0,places=13)
        rng=np.random.default_rng(22);v=rng.normal(size=(8,8,4));self.assertEqual(partitions({'Q':v},b,b,b!=b)[0]['Q'],0)
    def test_shared_domain_and_partition_sum(self):
        b=np.zeros((8,8),bool);b[:4]=True;s=b.copy();s[3,3]=False;v=np.ones((8,8,2));w=v.copy();w[7,7]=0
        out,_,n=partitions({'A':v,'B':w},b,s,b!=s);self.assertEqual(n,63);self.assertEqual(out['A'],out['B'])
        v[:4,:,0]=np.tile([-1,1],(4,4));v[:4,:,1]=0
        self.assertEqual(partitions({'Q':v},b,s,b!=s)[1],'ZERO_PARTITION_SUM')
    def test_sampling_pointwise_model_and_no_grad(self):
        import torch
        import torch.nn.functional as F
        x=torch.arange(512,dtype=torch.float32)[None,None,None,:].expand(1,3,512,512).clone();crop=x[:,:,128:384,128:384]
        up=F.interpolate(crop,size=(512,512),mode='bilinear',align_corners=False);down=F.interpolate(up,size=(256,256),mode='bilinear',align_corners=False)
        self.assertTrue(torch.equal(down[:,:,:,1:-1],crop[:,:,:,1:-1]));self.assertNotEqual(float(down[0,0,0,0]),float(crop[0,0,0,0]))
        m=torch.nn.Conv2d(3,2,1).eval().requires_grad_(False);before={k:v.clone() for k,v in m.state_dict().items()}
        with torch.no_grad():a=F.interpolate(m(up),size=(256,256),mode='bilinear',align_corners=False);bb=m(down)
        self.assertTrue(torch.allclose(a,bb,rtol=1e-5,atol=1e-4));self.assertFalse(a.requires_grad);self.assertFalse(torch.cuda.is_initialized());self.assertTrue(all(torch.equal(v,before[k]) for k,v in m.state_dict().items()))
    def test_stratified_permutation_scope(self):
        a=[dict(block_id=i,common=True,direction='EXPAND',size_bucket='1',distance_bucket='LE1',scores={'CROP_Q':float(i)}) for i in (1,2,3)]
        ct=stratify(a);self.assertEqual(ct['permutation_positions_moved'],3);self.assertEqual([r['scores']['STRATIFIED_ROLL'] for r in a],[3,1,2])
    def test_single_reversion_pool_core(self):
        for bits in range(16):
            b=np.array([bool(bits>>i&1) for i in range(4)]);pool=[b.copy()]
            for i in range(4):v=b.copy();v[i]=~v[i];pool.append(v)
            self.assertTrue(np.array_equal(np.mean(pool,axis=0)>=.5,b))
