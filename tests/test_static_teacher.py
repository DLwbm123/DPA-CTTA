import unittest
from pathlib import Path
import json
import torch
import numpy as np
from dpa_ctta.srg_ctta.static_teacher import outputs,region_counts,validate
from dpa_ctta.r78_cross_patient.score import metric

class Tests(unittest.TestCase):
    def test_fusion_is_probability_mean(self):
        z=torch.tensor([[[[-4.,-.2,3.,-2.]],[[2.,1.,-.5,4.]]]])
        t=torch.tensor([[[[2.,.4,1.,1.]],[[1.,-3.,.1,2.]]]])
        out=outputs(z,t)
        self.assertTrue(torch.equal(out['SOURCE'],z>=0));self.assertTrue(torch.equal(out['TEACHER_FLIP'],t>=0))
        self.assertTrue(torch.equal(out['PROB_AVG'],(z.double().sigmoid()+t.double().sigmoid())/2>=.5))
        self.assertTrue(torch.equal(out['PROB_AVG'],z+t>=0))
        self.assertFalse(bool(((out['PROB_AVG']!=out['SOURCE'])&(out['SOURCE']==out['TEACHER_FLIP'])).any()))
    def test_agreement_and_bad_shape(self):
        z=torch.tensor([[[[-1.,0.,2.]],[[3.,-2.,0.]]]])
        for p in outputs(z,z).values():self.assertTrue(torch.equal(p,z>=0))
        with self.assertRaises(ValueError):outputs(z[:,:1],z[:,:1])
    def test_net_counts_close_disagreement(self):
        p=np.array([1,1,0,1],bool);y=np.array([0,1,0,1],bool);b=np.array([0,0,1,1],bool)
        n=region_counts(p,y,b,np.ones_like(y));self.assertEqual(n,dict(corrected=2,spoiled=1,disagreement=3,pixels=4,net=1))
        self.assertEqual(region_counts(p,y,b,y)['net'],1)
        self.assertEqual(region_counts(p,y,b,np.zeros_like(y))['pixels'],0)
    def test_structure_and_empty_coverage(self):
        y=np.zeros((2,8,8),bool);y[0,1:7,1:7]=True;y[1,3:5,3:5]=True
        mm,rr=metric(y,y,y);self.assertEqual(mm[1]['dice_percent'],100.);self.assertEqual(mm[1]['hd95_pixel'],0.);self.assertEqual(rr['vcdr_abs_error'],0.)
        p=y.copy();p[1]=False;mm,_=metric(p,y,y);self.assertIsNone(mm[1]['assd_pixel']);self.assertTrue(mm[1]['pred_empty'])
    def test_config_weighting_and_immutability(self):
        c=json.loads((Path(__file__).parents[1]/'configs/static_flip_teacher_e2a.json').read_text());validate(c)
        self.assertEqual(8*len(c['stream_blocks'])*len(c['orders'])*len(c['arms']),240)
        self.assertEqual(8*4*len(c['arms']),96)
        c['fusion_weight']=.6
        with self.assertRaises(AssertionError):validate(c)

if __name__=='__main__':unittest.main()
