import unittest,json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_fill_holes
from dpa_ctta.srg_ctta.smooth_fill import outputs,ARMS
from dpa_ctta.srg_ctta.topology_projection import BG
from dpa_ctta.srg_ctta.smooth_fill_run import validate,summary
class Tests(unittest.TestCase):
    def fixture(self):
        x=np.zeros((2,12,12),bool);x[:,2:9,2:9]=True;x[:,4:6,4:6]=False;x[:,10,10]=True
        return x
    def test_fill_preserves_components_and_only_fills(self):
        x=self.fixture();y=outputs(x,x)['FILL_SMOOTH']
        self.assertTrue(y[:,10,10].all());self.assertTrue(y[:,4:6,4:6].all());self.assertTrue(np.all(y[x]))
        self.assertFalse(outputs(x,x)['TOPO_SMOOTH'][:,10,10].any())
    def test_control_same_added_mass_no_deletions(self):
        x=self.fixture();p=outputs(x,x)
        for ch in range(2):
            a=p['FILL_SMOOTH'][ch];b=p['MASS_MATCH_SMOOTH'][ch]
            self.assertEqual(a.sum(),b.sum());self.assertEqual((a!=x[ch]).sum(),(b!=x[ch]).sum());self.assertFalse((x[ch]&~b).any())
    def test_idempotence_and_empty_full(self):
        for x in (self.fixture(),np.zeros((2,6,6),bool),np.ones((2,6,6),bool)):
            p=outputs(x,x)['FILL_SMOOTH'];self.assertTrue(np.array_equal(outputs(p,p)['FILL_SMOOTH'],p))
    def test_external_background_stays_and_independent_heads(self):
        x=self.fixture();x[1]=False;y=outputs(x,x)['FILL_SMOOTH'];self.assertFalse(y[1].any());self.assertFalse(y[:,0,:].any())
    def test_inputs_repeat_no_mutation(self):
        x=self.fixture();before=x.copy();p=outputs(x,x);q=outputs(x,x)
        self.assertTrue(np.array_equal(x,before))
        for a in ARMS:self.assertTrue(np.array_equal(p[a],q[a]))
        with self.assertRaises(ValueError):outputs(x.astype(float),x)
    def test_config_and_static_score(self):
        c=json.loads((Path(__file__).parents[1]/'configs/smooth_fill_p5.json').read_text());validate(c)
        self.assertEqual(summary('z',np.zeros((8,5)))['delta_pp'],0)
        c['primary']='other'
        with self.assertRaises(AssertionError):validate(c)
