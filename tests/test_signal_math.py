import unittest
import numpy as np
from fractions import Fraction
from dpa_ctta.srg_ctta.signal_math import evidence,block_anchors,score_frame,auc,stratify

class SignalTests(unittest.TestCase):
    def test_cosine_sign_constant_zero(self):
        q=np.array([[1.,0.]])
        self.assertEqual(evidence(q,q,np.array([[0.,1.]]))[0],1)
        self.assertEqual(evidence(q,q,q)[0],0)
        self.assertIsNone(evidence(q,np.zeros_like(q),q))
        with self.assertRaises(ValueError):evidence(q*np.nan,q,q)
    def test_auc_ties_weights_undefined(self):
        self.assertEqual(auc([1,0],[1,-1]),1)
        self.assertEqual(auc([0,0],[1,-1]),.5)
        self.assertEqual(auc([0,1],[1,-1],[2,3]),0)
        self.assertIsNone(auc([1],[1]))
    def test_anchor_order_singleton(self):
        g=np.zeros((40,40),int);g[20,20]=1;eligible=np.ones_like(g,bool);p=np.zeros_like(g,float);p[:20]=1
        q,a=block_anchors(g,1,eligible,p,np.ones_like(g,bool))
        self.assertEqual(q.tolist(),[[20,20]]);self.assertEqual(list(map(len,a)),[8,8])
        self.assertEqual(a[0][0].tolist(),[19,20]);self.assertEqual(a[1][0].tolist(),[20,20])
    def test_missing_and_empty(self):
        s=np.zeros((2,40,40),bool);b=s.copy();f=np.ones((40,40,32));z=np.zeros((40,40,2))
        self.assertEqual(score_frame(f,z,s,b)[0],[])
        b[1,20,20]=True;rows,_=score_frame(f,z,s,b)
        self.assertFalse(rows[0]['common']);self.assertEqual(rows[0]['reason'],'ANCHORS_SHORT')
    def test_stratified_scope(self):
        rows=[dict(common=True,block_id=i,direction='X',size_bucket='1',distance_bucket='LE1',scores={'FEATURE':float(i)}) for i in range(3)]
        d=stratify(rows);self.assertEqual(d['permutation_positions_moved'],3);self.assertEqual(rows[0]['scores']['STRATIFIED_ROLL'],2)
    def test_full_scores_and_shift_mapping(self):
        s=np.ones((2,64,64),bool);s[1,32:]=False;b=s.copy();b[1,31,32]=False
        z=np.zeros((64,64,2));z[:,:,0]=4;z[:,:,1]=np.where(s[1],3.,-3.)
        f=np.ones((64,64,32));f[:,:,0]=np.arange(64)[None,:]+1
        rows,_=score_frame(f,z,s,b);self.assertTrue(rows[0]['common'])
        rr,_=score_frame(np.roll(f,33,axis=1),z,s,b)
        self.assertEqual(rows[0]['scores']['FEATURE_SHIFT'],rr[0]['scores']['FEATURE'])
        rows,_=score_frame(np.ones_like(f),z,s,b);self.assertEqual(rows[0]['scores']['FEATURE'],0)
    def test_utility_and_missing_group(self):
        from dpa_ctta.srg_ctta.signal_score import withdrawal_utility,aggregate
        self.assertEqual(withdrawal_utility(10,10,1,-3,-1),Fraction(-1,10))
        self.assertEqual(withdrawal_utility(0,0,0,1,0),-1)
        self.assertEqual(withdrawal_utility(1,0,0,-1,0),1)
        v=np.ones((8,4));v[0]=np.nan;r,_=aggregate(v);self.assertEqual(r['defined_contents'],7);self.assertIsNone(r['ci95'])
    def test_access_boundaries(self):
        from dpa_ctta.srg_ctta.signal_access import callback
        c=dict(output_root='/tmp/synthetic-root',source_root='/tmp/synthetic-data',checkpoint_path='/tmp/synthetic-weight',denied_prior_roots=['/tmp/synthetic-oracle'])
        a,opened=callback(c,[dict(image_path='/tmp/synthetic-data/rgb')]);a('open',('/tmp/synthetic-data/rgb',));self.assertEqual(len(opened),1)
        for p in ('/tmp/synthetic-data/gt','/tmp/synthetic-root/scorer/label','/tmp/synthetic-oracle/result','/tmp/synthetic-root/code/signal_score.py','/tmp/synthetic-root/code/__pycache__/signal_score.cpython-310.pyc'):
            with self.assertRaises(PermissionError):a('open',(p,))
        with self.assertRaises(PermissionError):a('socket.connect',())



class GuardTests(unittest.TestCase):
    def test_space_and_identity(self):
        from dpa_ctta.srg_ctta.signal_control import used
        from dpa_ctta.srg_ctta.signal_common import identity
        import tempfile,os
        from pathlib import Path
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);(p/'a').write_bytes(b'123');(p/'link').symlink_to(p/'a')
            self.assertEqual(used(p),3)
        self.assertIsInstance(identity(os.getpid()),int)
        self.assertIsNone(identity(999999999))

if __name__=="__main__":unittest.main()
