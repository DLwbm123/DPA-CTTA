import unittest,json
from pathlib import Path
import torch
from torch.nn import functional as F
import numpy as np
from dpa_ctta.srg_ctta.translation_phase import shifted,outputs,ARMS
from dpa_ctta.srg_ctta.phase_run import validate,summary
class Pointwise(torch.nn.Module):
    def forward(self,x):return (x[:,:2],)
class Sampled(torch.nn.Module):
    def forward(self,x):return (x[:,:2,::2,::2].repeat_interleave(2,-1).repeat_interleave(2,-2),)
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def fixture(self):
        y,x=torch.meshgrid(torch.arange(8),torch.arange(8),indexing='ij')
        return torch.stack((x.float()-3.5,y.float()-3.5,(x+y).float()/4))[None]
    def test_exact_integer_translation_and_inverse(self):
        x=self.fixture();z=x[:,:2].double();p,d=outputs(Pointwise(),x)
        for a in ('PHASE_PROB','PHASE_LOGIT','SHIFT_X','SHIFT_Y','SHIFT_XY'):
            torch.testing.assert_close(p[a],z,atol=1e-12,rtol=1e-12)
        # A threshold tie can leave hard masks equal even when fields are misregistered.
        self.assertFalse(torch.allclose(p['NOALIGN_PROB'],z,atol=1e-12,rtol=1e-12))
        self.assertEqual(d['source_fallback_pixels'],15)
    def test_stride_aliasing_nontrivial_without_new_weights(self):
        x=self.fixture();x[:,0]+=torch.arange(8)[None,:].remainder(2)*1.7
        p,d=outputs(Sampled(),x)
        self.assertGreater(d['aligned_phase_probability_rms'],.01)
        self.assertFalse(torch.equal(p['PHASE_PROB'],p['SOURCE']))
        self.assertFalse(torch.equal(p['PHASE_PROB'],p['NOALIGN_PROB']))
    def test_border_no_roll_or_interpolation_and_input_invariant(self):
        x=self.fixture();before=x.clone();p,d=outputs(Sampled(),x);q,dd=outputs(Sampled(),x)
        self.assertTrue(torch.equal(x,before));self.assertEqual(d,dd)
        self.assertTrue(torch.equal(shifted(x,0,1)[...,1:],x[...,:-1]))
        self.assertTrue(torch.equal(shifted(x,1,0)[...,1:,:],x[...,:-1,:]))
        for a in ARMS:
            self.assertTrue(torch.equal(p[a],q[a]));self.assertTrue(torch.equal(p[a][...,-1,:],p['SOURCE'][...,-1,:]));self.assertTrue(torch.equal(p[a][...,:,-1],p['SOURCE'][...,:,-1]));self.assertFalse(p[a].requires_grad)
    def test_independent_heads_and_failure(self):
        x=torch.ones(1,3,8,8)*2;p,d=outputs(Pointwise(),x)
        self.assertTrue((p['PHASE_PROB']>0).all()) # both foreground allowed; no softmax
        with self.assertRaises(ValueError):shifted(x,2,0)
        with self.assertRaises(ValueError):outputs(Pointwise(),x.repeat(2,1,1,1))
        x[:]=float('nan')
        with self.assertRaises(ValueError):outputs(Pointwise(),x)
    def test_config_and_static_score(self):
        c=json.loads((Path(__file__).parents[1]/'configs/translation_phase_p1.json').read_text());validate(c)
        self.assertEqual(summary('x',np.zeros((8,5,2)))['delta_pp'],0.)
        c['offsets'][-1]=[-1,-1]
        with self.assertRaises(AssertionError):validate(c)
if __name__=='__main__':unittest.main()
