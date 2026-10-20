import unittest,json
from pathlib import Path
import torch
from dpa_ctta.srg_ctta.rgb_diffusion import affinities,outputs,neighborhoods,ARMS
from dpa_ctta.srg_ctta.rgb_diffusion_run import validate
class Tests(unittest.TestCase):
    def fixture(self):
        q=torch.arange(12*13,dtype=torch.float64).reshape(1,1,12,13)
        x=torch.cat((q.sin(),q.cos(),(q*.1).sin()),dim=1)
        z=torch.cat(((q*.21).sin()*4,(q*.31).cos()*3),dim=1)
        return x,z
    def test_constant_logits_are_preserved(self):
        x,z=self.fixture();z.fill_(4);out,d=outputs(z,x)
        for a in ('EDGE_LOGIT','ROLL_MATCH','UNIFORM_MATCH','UNIFORM_LOGIT'):self.assertTrue(torch.allclose(out[a],z,atol=1e-14,rtol=0))
    def test_flat_rgb_reduces_to_uniform(self):
        x,z=self.fixture();x.zero_();out,d=outputs(z,x)
        self.assertEqual(d['rgb_positive_neighbor_median'],0)
        self.assertTrue(torch.equal(out['EDGE_LOGIT'],out['UNIFORM_LOGIT']))
    def test_rgb_affine_scale_invariance(self):
        x,z=self.fixture();w,s=affinities(x);v,t=affinities(x*3+7)
        self.assertTrue(torch.allclose(w,v,atol=2e-14,rtol=2e-14))
    def test_convex_local_bounds(self):
        x,z=self.fixture();out,d=outputs(z,x);n=neighborhoods(z)
        self.assertTrue((out['EDGE_LOGIT'][...,:-1,:-1]>=n.min(0).values[...,:-1,:-1]-1e-14).all())
        self.assertTrue((out['EDGE_LOGIT'][...,:-1,:-1]<=n.max(0).values[...,:-1,:-1]+1e-14).all())
    def test_controls_match_actual_output_norm(self):
        x,z=self.fixture();out,d=outputs(z,x);norm=lambda a:(out[a]-z).square().sum((-2,-1)).sqrt()
        for a in ('ROLL_MATCH','UNIFORM_MATCH'):self.assertTrue(torch.allclose(norm(a),norm('EDGE_LOGIT'),rtol=1e-13,atol=1e-13))
        self.assertGreater(d['roll_vs_edge_weight_rms'],0)
        w,_=affinities(x);r=torch.roll(w,z.shape[-1]//2+1,dims=-1)
        self.assertTrue(torch.equal(w.flatten(-2).sort(-1).values,r.flatten(-2).sort(-1).values))
    def test_repeat_source_boundary_and_head_equivariance(self):
        x,z=self.fixture();p,d=outputs(z,x);q,e=outputs(z,x);v,_=outputs(z.flip(1),x)
        self.assertTrue(torch.equal(p['SOURCE'],z));self.assertEqual(d,e)
        for a in ARMS:self.assertTrue(torch.equal(p[a],q[a]));self.assertTrue(torch.equal(p[a].flip(1),v[a]))
        for a in ('EDGE_LOGIT','ROLL_MATCH','UNIFORM_MATCH','UNIFORM_LOGIT','SMOOTH'):
            self.assertTrue(torch.equal(p[a][...,-1,:],z[...,-1,:]));self.assertTrue(torch.equal(p[a][...,:,-1],z[...,:,-1]))
    def test_native_probability_control_matches_p1(self):
        from dpa_ctta.srg_ctta.translation_phase import outputs as old
        x,z=self.fixture()
        def model(v):return (z,)
        prior,_=old(model,x);out,_=outputs(z,x)
        self.assertTrue(torch.equal(prior['SOURCE_SMOOTH'],out['SMOOTH']))
    def test_invalid_inputs_stop(self):
        x,z=self.fixture();x[0,0,0,0]=float('nan')
        with self.assertRaises(ValueError):outputs(z,x)
        x,z=self.fixture()
        with self.assertRaises(ValueError):outputs(z[:,:1],x)
    def test_frozen_config_and_static_statistics(self):
        c=json.loads((Path(__file__).parents[1]/'configs/rgb_diffusion_e3.json').read_text());validate(c)
        c['parameter_updates']=1
        with self.assertRaises(AssertionError):validate(c)
