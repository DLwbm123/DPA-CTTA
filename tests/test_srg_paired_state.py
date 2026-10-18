import unittest
import json
from pathlib import Path
import copy
import numpy as np
from dpa_ctta.srg_ctta.paired_state import summary,validate


class PairedContrastTests(unittest.TestCase):
    def test_bound_content_effect_and_nonfinite_rejection(self):
        d=np.ones((8,5,2,2))*2
        r=summary('paired PRE',d)
        self.assertEqual(r['delta_pp'],2.)
        self.assertEqual(r['ci95_pp'],[2.,2.])
        self.assertEqual(r['total_cells'],160)
        self.assertEqual(r['negative_cells'],0)
        d[0]*=-1
        self.assertEqual(summary('paired PRE',d)['negative_cells'],20)
        with self.assertRaises(ValueError):summary('bad grouping',np.ones((16,10)))
        d[0,0,0,0]=np.nan
        with self.assertRaises(ValueError):summary('bad values',d)

    def test_frozen_versions_preserve_main_and_native_controls(self):
        root=Path(__file__).resolve().parents[1]
        v5=json.loads((root/'configs/stratified_reliability_paired_state_v5.json').read_text())
        v6=json.loads((root/'configs/stratified_reliability_paired_state_v6.json').read_text())
        _,old=validate(v5);_,new=validate(v6)
        self.assertEqual(set(old)-set(new),{('RG','RESET'),('NOANCHOR','RESET'),('ENT_NORM','RESET')})
        self.assertEqual(set(new)-set(old),set())
        for a in ('STRAT','RANDOM'):
            self.assertTrue(all((a,p) in new for p in ('RESET','RETAIN','FRESH')))
        self.assertTrue(all((a,p) in new for a in ('CW1','CW2') for p in ('RESET','RETAIN')))
        self.assertEqual(len(new)*160+400,2480)
        bad=copy.deepcopy(v6);bad['expected_readouts']=2960
        with self.assertRaises(AssertionError):validate(bad)
        bad=copy.deepcopy(v6);bad['design_version']='V7'
        with self.assertRaises(ValueError):validate(bad)
