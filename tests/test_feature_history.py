import json,os,runpy,unittest
from pathlib import Path
import torch
from dpa_ctta.srg_ctta.feature_history import History,Stream,JOBS


class Checks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(20261018)
        self.f=torch.rand(32,4,4,dtype=torch.float64);self.f[-1]=2.
        self.w=torch.randn(2,32,dtype=torch.float64)*.2;self.b=torch.tensor([-.7,.2],dtype=torch.float64)
        self.z=torch.einsum('hd,dyx->hyx',self.w,self.f)+self.b[:,None,None]

    def test_against_scalar_reference_and_zero_passthrough(self):
        ref=runpy.run_path(str(Path(__file__).parents[1]/'reports/post-q1-source-preserving-history/formula_qualification.py'))['WorkingHistory']
        h=History(self.w);f=self.f.reshape(1,32,-1).expand(2,-1,-1);z=self.z.reshape(2,-1)
        self.assertTrue(torch.equal(h.emit(f,z)[0],z))
        h.add(f,z);query=self.f.clone();query[-1]=3.;qq=torch.einsum('hd,dyx->hyx',self.w,query)+self.b[:,None,None]
        q,r=h.emit(query.reshape(1,32,-1).expand(2,-1,-1),qq.reshape(2,-1))
        for head in range(2):
            model=ref(self.w[head].tolist(),float(self.b[head]),self.f.reshape(32,-1).T.tolist());model.add(self.f.reshape(32,-1).T.tolist())
            pred=model.prediction(query[:,0,0].tolist())
            self.assertAlmostEqual(float(q[head,0]),pred[1],places=10);self.assertAlmostEqual(float(r[head,0]),pred[2],places=10)
        self.assertFalse(h.active[:,-1].any())

    def test_image_mass_replication(self):
        f=self.f.reshape(1,32,-1).expand(2,-1,-1);z=self.z.reshape(2,-1);a=History(self.w);b=History(self.w)
        a.add(f,z);b.add(f.repeat(1,1,7),z.repeat(1,7))
        self.assertTrue(torch.allclose(a.n,b.n,atol=1e-12,rtol=0));self.assertTrue((a.n.sum(-1)<=1).all())
        for x,y in zip(a.emit(f,z),b.emit(f,z)):self.assertTrue(torch.allclose(x,y,atol=1e-10,rtol=0))

    def test_reset_initial_and_temporal_causality(self):
        stream=Stream(self.w);pre,post,d=stream.step(self.f,self.z)
        self.assertEqual(set(pre),set(JOBS))
        self.assertTrue(all(torch.equal(v,self.z) for v in pre.values()))
        again,_,_=stream.step(self.f,self.z)
        self.assertTrue(torch.equal(again['FEATURE_Q','RETAIN'],post['FEATURE_Q','RETAIN']))
        self.assertTrue(torch.equal(again['FEATURE_Q','RESET'],self.z))
        self.assertEqual(stream.full.images,2);stream.full.reset();self.assertEqual(stream.full.images,0)

    def test_shuffle_support_and_distinct_mean(self):
        stream=Stream(self.w);stream.step(self.f,self.z)
        self.assertTrue(torch.allclose(stream.full.n,stream.shuffled.n,atol=1e-12,rtol=0))
        self.assertFalse(torch.equal(stream.full.s,stream.shuffled.s));self.assertTrue(torch.allclose(stream.full.n,stream.margin.n,atol=1e-12,rtol=0))

    def test_covariant_geometry(self):
        f=self.f.reshape(1,32,-1).expand(2,-1,-1);z=self.z.reshape(2,-1)
        a=History(self.w);a.add(f,z);old=a.emit(f,z)
        scale=torch.linspace(.2,3.,32,dtype=torch.float64)[None,:,None];shift=torch.linspace(0.,2.,32,dtype=torch.float64)[None,:,None]
        b=History(self.w/scale[:,:,0]);b.add(f*scale+shift,z)
        for x,y in zip(old,b.emit(f*scale+shift,z)):self.assertTrue(torch.allclose(x,y,atol=1e-9,rtol=0))

    def test_clip_and_readonly_repeat(self):
        a=Stream(self.w);b=Stream(self.w)
        for _ in range(3):
            pa,qa,_=a.step(self.f,self.z);pb,qb,_=b.step(self.f,self.z)
            for key in JOBS:self.assertTrue(torch.equal(pa[key],pb[key]) and torch.equal(qa[key],qb[key]))
            clip=qa['FEATURE_CLIP','RETAIN'];q=qa['FEATURE_Q','RETAIN']
            self.assertTrue(((clip>=torch.minimum(q,self.z))&(clip<=torch.maximum(q,self.z))).all())
        self.assertTrue(torch.equal(a.full.s,b.full.s))

    def test_degenerate_and_nonfinite_stop(self):
        h=History(self.w)
        with self.assertRaises(ValueError):h.add(torch.ones(2,32,16,dtype=torch.float64),self.z.reshape(2,-1))
        z=self.z.clone();z[0,0,0]=float('nan')
        with self.assertRaises(ValueError):h.add(self.f.reshape(1,32,-1).expand(2,-1,-1),z.reshape(2,-1))

    def test_config(self):
        from dpa_ctta.srg_ctta.history_run import validate
        c=json.loads((Path(__file__).parents[1]/'configs/source_preserving_history_v1.json').read_text());validate(c)
        c['rdc']=True
        with self.assertRaises(AssertionError):validate(c)

    def test_real_checkpoint_cpu_interface(self):
        if 'DIAG_CONFIG' not in os.environ:self.skipTest('formal remote CPU admission supplies existing checkpoint and reference')
        from dpa_ctta.integrations.ctta_suite import build_reference_model
        c=json.loads(Path(os.environ['DIAG_CONFIG']).read_text());base,_=build_reference_model('fundus')
        base.load_state_dict(torch.load(c['checkpoint_path'],map_location='cpu',weights_only=True));base.eval().requires_grad_(False)
        with torch.no_grad():
            out=base(torch.rand(1,3,512,512));z,feat=out[0],out[2]
            self.assertEqual(tuple(feat.shape),(1,32,512,512));self.assertTrue(torch.equal(z,base.seg_head(feat)))
        self.assertFalse(torch.cuda.is_initialized())

if __name__=='__main__':unittest.main()
