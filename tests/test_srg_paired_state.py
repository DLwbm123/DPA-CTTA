import unittest
import numpy as np
from dpa_ctta.srg_ctta.paired_state import summary


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
