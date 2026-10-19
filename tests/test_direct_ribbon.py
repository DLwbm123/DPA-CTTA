import unittest,json
from pathlib import Path
import torch
import numpy as np
from dpa_ctta.srg_ctta.direct_ribbon import geometry,proposal,outputs,match_injection,ARMS
from dpa_ctta.srg_ctta.ribbon_run import validate,summary

class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)
    def fixture(self,edge=128.5):
        y,x=torch.meshgrid(torch.arange(256),torch.arange(256),indexing='ij');x=x.float()
        image=(x>edge).float()[None,None].repeat(1,3,1,1)
        z=torch.stack((140.5-x,127.5-x))[None]
        return image,z
    def test_step_evidence_changes_signed_direction(self):
        image,z=self.fixture();g=geometry(image,z);e,valid,info=proposal(g,g['descriptor'])
        self.assertGreater(float(e[:,1].sum()),0);self.assertGreater(info[1]['moved'],0)
        image,z=self.fixture(126.5);g=geometry(image,z);left,_,_=proposal(g,g['descriptor'])
        self.assertLess(float(left[:,1].sum()),0)
        self.assertFalse(e.requires_grad)
    def test_constant_evidence_keep_and_zero_legal(self):
        image,z=self.fixture();pred,d=outputs(torch.zeros_like(image),z)
        self.assertEqual(d['moved'],0)
        for a in ARMS:self.assertTrue(torch.equal(pred[a],z))
    def test_locality_matched_destruction_and_immutability(self):
        image,z=self.fixture();before=z.clone();original=image.clone();p,d=outputs(image,z);q,dd=outputs(image,z)
        self.assertTrue(torch.equal(z,before));self.assertTrue(torch.equal(image,original));self.assertEqual(d,dd)
        for a in ARMS:self.assertTrue(torch.equal(p[a],q[a]))
        target=p['RIBBON']-z
        torch.testing.assert_close(p['REVERSE']-z,-target,atol=2e-6,rtol=1e-5)
        for a in ('DELTA_ROLL','EXPAND_MATCH','SHRINK_MATCH'):
            torch.testing.assert_close((p[a]-z).double().square().mean((-2,-1)).sqrt(),target.double().square().mean((-2,-1)).sqrt(),atol=1e-6,rtol=1e-4)
        self.assertTrue(torch.equal(p['RIBBON'][...,0,:],z[...,0,:]))
        self.assertFalse(torch.equal(p['RIBBON'],p['IMAGE_ROLL']))
        # A uniform straight edge gives a constant proposal: permutation can legitimately be identical.
        image[:,:,128:]=self.fixture(126.5)[0][:,:,128:]
        mixed,_=outputs(image,z)
        self.assertFalse(torch.equal(mixed['RIBBON'],mixed['DELTA_ROLL']))
    def test_single_independent_heads_nonfinite_and_empty_geometry(self):
        image,z=self.fixture();z[:,1]=-100
        p,d=outputs(image,z);self.assertTrue(torch.equal(p['RIBBON'][:,1],z[:,1]))
        self.assertEqual(d['OC_eligible'],0)
        with self.assertRaises(ValueError):geometry(image.repeat(2,1,1,1),z)
        image[:]=float('nan')
        with self.assertRaises(ValueError):geometry(image,z)
        with self.assertRaises(ValueError):match_injection(torch.zeros_like(z),torch.ones_like(z))
    def test_frozen_config_and_static_shape(self):
        c=json.loads((Path(__file__).parents[1]/'configs/direct_ribbon_b1.json').read_text());validate(c)
        self.assertEqual(summary('x',np.zeros((8,5,2)))['delta_pp'],0)
        c['displacements']=[0,-2,2]
        with self.assertRaises(AssertionError):validate(c)

if __name__=='__main__':unittest.main()
