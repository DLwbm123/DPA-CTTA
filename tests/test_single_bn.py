import unittest
from pathlib import Path
import json
import torch
import numpy as np
from dpa_ctta.srg_ctta.single_bn import moments,installed,output,check_transport,ARMS
from dpa_ctta.srg_ctta.bn_run import validate,summary,PRIMARY

class Tests(unittest.TestCase):
    def fixture(self):
        bn=torch.nn.BatchNorm2d(3).eval()
        with torch.no_grad():
            bn.running_mean.copy_(torch.tensor([.4,-.6,.3]));bn.running_var.copy_(torch.tensor([1.,.3,2.]));bn.weight.copy_(torch.tensor([1.1,-.7,0.]));bn.bias.copy_(torch.tensor([.2,.1,-.2]))
        h=torch.arange(48,dtype=torch.float32).reshape(1,3,4,4)/20
        return bn,h
    def test_exact_scale_and_centering_identity(self):
        bn,h=self.fixture();v,m=moments(h)
        before={n:x.clone() for n,x in bn.state_dict().items()}
        y={a:output(bn,h,m,v,a) for a in ARMS}
        d=check_transport(bn,h,y['SOURCE'],y['VAR_ONLY'],y['RB_BN'],m,v)
        self.assertLess(d['rb_frozen_mean_max_error'],1e-5)
        np.testing.assert_allclose(y['ADABN'].detach().mean((0,2,3)).numpy(),bn.bias.detach().numpy(),atol=1e-6)
        self.assertTrue(torch.equal(y['SOURCE'],bn(h)))
        for n,t in bn.state_dict().items():self.assertTrue(torch.equal(t,before[n]))
    def test_zero_variance_no_extra_floor(self):
        bn,h=self.fixture();h[:]=2;v,m=moments(h);self.assertTrue(torch.equal(v,torch.zeros_like(v)))
        yy={a:output(bn,h,m,v,a) for a in ARMS}
        check_transport(bn,h,yy['SOURCE'],yy['VAR_ONLY'],yy['RB_BN'],m,v)
        self.assertTrue(all(torch.isfinite(y).all() for y in yy.values()))
    def test_population_axes_and_replication(self):
        bn,h=self.fixture();v,m=moments(h)
        vv,mm=moments(h.repeat_interleave(3,-1))
        torch.testing.assert_close(v,vv);torch.testing.assert_close(m,mm)
        torch.testing.assert_close(v,h.double().reshape(3,-1).var(1,correction=0))
        with self.assertRaises(ValueError):moments(h.repeat(2,1,1,1))
        h[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):moments(h)
    def test_hook_coordinate_and_no_update(self):
        bn,h=self.fixture();model=torch.nn.Sequential(bn,torch.nn.ReLU());v,m=moments(h)
        mode='RB_BN'
        handle=bn.register_forward_hook(lambda mod,args,y:output(mod,args[0],m,v,mode))
        with torch.no_grad():a=model(h);b=model(h)
        handle.remove();self.assertTrue(torch.equal(a,b));self.assertTrue(all(p.grad is None for p in model.parameters()))
    def test_frozen_config_and_score_shape(self):
        c=json.loads((Path(__file__).parents[1]/'configs/single_bn_a0.json').read_text());validate(c)
        self.assertEqual(PRIMARY,'RB_BN-VAR_ONLY:OC_DICE')
        self.assertEqual(summary('x',np.zeros((8,5,2)))['delta_pp'],0.)
        c['bn_layer']='up3.bn'
        with self.assertRaises(AssertionError):validate(c)

if __name__=='__main__':unittest.main()
