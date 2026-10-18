import unittest
import torch
from dpa_ctta.srg_ctta.repeatability import compare_outputs


class RepeatabilityTests(unittest.TestCase):
    def test_detects_subthreshold_drift_separately_from_mask_change(self):
        a=[torch.tensor([-1.,1.,0.])];b=[torch.tensor([-.9,1.,-.001])]
        r=compare_outputs(a,b)
        self.assertEqual(r['changed_logit_elements'],2)
        self.assertEqual(r['changed_prediction_pixels'],1)
        self.assertAlmostEqual(r['max_abs_logit_difference'],.1,places=6)
        self.assertEqual(compare_outputs(a,a)['changed_logit_elements'],0)
        with self.assertRaises(ValueError):compare_outputs(a,[])
        with self.assertRaises(ValueError):compare_outputs(a,[torch.zeros(1)])
        with self.assertRaises(ValueError):compare_outputs([torch.tensor([float('inf')])],[torch.tensor([float('inf')])])
