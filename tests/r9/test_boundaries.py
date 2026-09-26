import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from test_core import fit_fixture,host_fixture,API
from dpa_ctta.r9_current_first import admission,storage,target,metrics,reuse,resolution
from dpa_ctta.r9_current_first.protocol import SPEC,disabled_config,digest
from dpa_ctta.r9_current_first.training import SourceTrainer
from dpa_ctta.r9_current_first.gradient import teacher_views
from dpa_ctta.r7_shared.numerics import COUNTS
from dpa_ctta.hosts.vptta import model_input_from_pixels

class Counter:
    steps=0
    def snapshot(self):return {'steps':self.steps}
    def restore(self,s):self.steps=s['steps']

class Boundaries(unittest.TestCase):
    def test_disabled_before_private_io(self):
        with self.assertRaises(PermissionError):admission.require_authorized(disabled_config())
        with patch.object(target,'TargetReader',side_effect=AssertionError('private read')):
            with self.assertRaises(ValueError):target.score([],None,None,None,None,{},lambda:None)

    def test_legacy_full_fork_and_no_physical_counter_import(self):
        args=fit_fixture();old=SourceTrainer(*args,20260924,'old',recipe='LEGACY')
        for _ in range(4):old.fit_step()
        old.steps=4000 # Synthetic boundary: tests restoration, not a claimed 4000-step fit.
        snap=old.snapshot()['fit'];new=SourceTrainer(*fit_fixture(),20260924,'new',recipe='LEGACY')
        physical=COUNTS.copy();new.fork_legacy(snap,'old','1'*64)
        self.assertEqual(physical,COUNTS)
        self.assertEqual(old.fit_step(),new.fit_step())
        for k,v in old.method.state_dict().items():self.assertTrue(torch.equal(v,new.method.state_dict()[k]))
        with self.assertRaises(ValueError):new.fork_legacy({'method':snap['method']},'old','1'*64)

    def test_journal_recovery_and_lease(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);w=Counter();j=storage.PhaseJournal(root/'job','fit',{'id':'bound'})
            j.begin(w);w.steps=1;j.append({'step':1});j.checkpoint(w)
            j.append({'step':2,'cost':1})
            failure=dict(class_='unused');failure={'class':'INFRASTRUCTURE','reason':'process interruption','evidence':'synthetic receipt'}
            j.begin(w,failure);self.assertEqual(w.steps,1);self.assertEqual(len(j.log.read_text().splitlines()),2)
            with self.assertRaises(ValueError):j.begin(w,failure)
            with storage.lease(root/'lease',{'id':1}):
                with self.assertRaises(BlockingIOError):
                    with storage.lease(root/'lease',{'id':1}):pass
            with self.assertRaises(ValueError):
                with storage.lease(root/'lease',{'id':2}):pass
            j.complete({'okay':True});self.assertTrue(j.completed()['result']['okay'])
            j.log.write_text('corrupted')
            with self.assertRaises(ValueError):j.completed()

    def test_metric_imbalance_and_undefined(self):
        rows=[]
        for d,n,value in [('a',100,1.),('b',1,0.),('c',1,0.),('d',1,0.)]:
            rows += [dict(domain=d,subset='remaining_dev',metrics=[dict(channel=c,dice=value) for c in ('OD','OC')]) for _ in range(n)]
        self.assertEqual(metrics.domain_macro(rows),.25)
        r=metrics.evaluate_probability(torch.zeros(1,2,8,8),torch.zeros(1,2,8,8))
        self.assertTrue(all(m['assd'] is None and m['dice']==1 for m in r))
        self.assertEqual(metrics.assd_pair([None,2.],[1.,1.]),dict(common=1,undefined=1,delta=1.))

    def test_reuse_and_all_symbolic_slots(self):
        old={k:'bound' for k in reuse.KEYS};receipt=dict(identity_sha256=digest(old),status='COMPLETE',seal='receipt')
        self.assertEqual(reuse.reuse(old,old,receipt)['new_model_forwards'],0)
        with self.assertRaises(ValueError):reuse.reuse(old,dict(old,seed_rng='changed'),receipt)
        sources={j['id']:dict(selection={'step':16000},artifacts={str(s):dict(file='artifact',sha256='1'*64,step=s) for s in (4000,8000,12000,16000)}) for j in SPEC['source_tasks']}
        for slot in SPEC['target_core_slots']+SPEC['target_final16k_slots_max']:
            r=resolution.resolve(slot,dict(recipes={'A':'SELF_TASK','B':'SELF'}),sources)
            self.assertEqual(r['slot'],slot['id'])

    def test_teacher_reference_and_backbone_frozen(self):
        h=host_fixture('B_G1');image=torch.rand(1,3,512,512);api=API()
        q,strong,_,_=teacher_views(h.segmenter,image,api,True)
        x=model_input_from_pixels(image,'fundus');weak=api.Rotate_and_Flip()
        with torch.no_grad():
            expected=[h.segmenter(image).sigmoid()]
            for i in range(5):expected.append(weak.inverse(h.segmenter.normalized(weak(x,i)),i).sigmoid())
        self.assertTrue(torch.equal(q,torch.stack(expected).mean(0)))
        expected_strong=api.normalize_image_to_0_1(torch.from_numpy(api.augmentation_strong_style({'data':x.numpy().copy()})).float())
        self.assertTrue(torch.equal(strong,expected_strong));self.assertFalse(q.requires_grad)
        before={k:v.clone() for k,v in h.segmenter.model.state_dict().items()}
        h.step(image)
        self.assertTrue(all(torch.equal(v,h.segmenter.model.state_dict()[k]) for k,v in before.items()))
        self.assertTrue(all(p.grad is None for p in h.segmenter.model.parameters()))

if __name__=='__main__':unittest.main()
