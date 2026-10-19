import unittest,json
from pathlib import Path
import numpy as np
from scipy.ndimage import label,binary_fill_holes
from dpa_ctta.srg_ctta.topology_projection import outputs,project,lcc,mass_matched,FG,BG,ARMS
from dpa_ctta.srg_ctta.topology_run import validate,summary
class Tests(unittest.TestCase):
    def fixture(self):
        a=np.zeros((12,12),bool);a[2:9,2:9]=True;a[4:6,4:6]=False;a[10,10]=True
        b=np.zeros_like(a);b[3:8,3:8]=True;b[5,5]=False
        return np.stack((a,b))
    def test_projection_and_idempotence(self):
        a=self.fixture()[0];p=project(a)
        self.assertEqual(label(p,FG)[1],1);self.assertTrue(np.array_equal(binary_fill_holes(p,structure=BG),p))
        self.assertTrue(p[4,4]);self.assertFalse(p[10,10]);self.assertTrue(np.array_equal(project(p),p))
    def test_exact_class_conditional_edit_budgets(self):
        for m in self.fixture():
            p=project(m);q=mass_matched(m,p)
            self.assertEqual(int((m&~q).sum()),int((m&~p).sum()));self.assertEqual(int((q&~m).sum()),int((p&~m).sum()))
            self.assertFalse(np.array_equal(q,p))
    def test_empty_full_tie_and_independent_heads(self):
        for a in (np.zeros((4,4),bool),np.ones((4,4),bool)):self.assertTrue(np.array_equal(project(a),a))
        m=np.zeros((5,5),bool);m[0,0]=m[4,4]=True;self.assertTrue(lcc(m)[0,0]);self.assertFalse(lcc(m)[4,4])
        x=self.fixture();x[1]=False;p=outputs(x,x)
        for a in ARMS:self.assertFalse(p[a][1].any())
    def test_no_mutation_no_nesting_repair_and_repeat(self):
        x=self.fixture();before=x.copy();p=outputs(x,x);q=outputs(x,x)
        self.assertTrue(np.array_equal(x,before))
        for a in ARMS:self.assertTrue(np.array_equal(p[a],q[a]))
        self.assertTrue(np.array_equal(p['SOURCE'],x));self.assertTrue(np.array_equal(p['SMOOTH'],x))
        with self.assertRaises(ValueError):outputs(x.astype(float),x)
    def test_config_and_score_shape(self):
        c=json.loads((Path(__file__).parents[1]/'configs/topology_projection_p2.json').read_text());validate(c)
        self.assertEqual(summary('x',np.zeros((8,5,2)))['delta_pp'],0)
        c['background_connectivity']=8
        with self.assertRaises(AssertionError):validate(c)
if __name__=='__main__':unittest.main()
