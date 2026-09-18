"""R7-F1: real loader/state boundaries, procedural Small + random full ResUNet."""
import copy,json,unittest
from unittest.mock import patch
import torch
from common import method,segmenter,pixels,source_fixture,basis
from dpa_ctta.r7_shared.host import OnlineHost
from dpa_ctta.r7_shared.preparation import inference_from_tensors,prepared_artifact,prepare_tensors
from dpa_ctta.r7_shared.context import (tensor_digest,environment,provenance,deployment_context,make_context,HASH_COST)
from dpa_ctta.r7_shared.numerics import COUNTS

class ContextTests(unittest.TestCase):
    def make(self,group='B',full=False):
        s=segmenter(full);m=None if group=='C0' else method(group)
        if m is not None:m.freeze()
        return s,m,prepared_artifact(s,m)
    def load(self,s,m,p,ablation=None):
        return inference_from_tensors(s,'C0' if m is None else m.group,False if m is None else m.static,None if m is None else m.basis,p['weights'],p['binding'],ablation)
    def test_all_groups_and_C0_cross_instance_restore_and_next_output(self):
        for group in ('A','B','C','C0'):
            a,m,p=self.make(group);h=self.load(a,m,p);h.step(pixels());saved=h.save_state()
            b=segmenter();other=self.load(b,m,p);other.load_state(saved)
            x,_=h.step(pixels(1));y,_=other.step(pixels(1));self.assertTrue(torch.equal(x,y));self.assertEqual(h.save_state()['context'],other.save_state()['context'])
            self.assertIsNot(a.model,b.model);a.close();b.close()
    def test_all_groups_and_C0_head_encoder_projection_reject_both_load_paths(self):
        for group in ('A','B','C','C0'):
            a,m,p=self.make(group);saved=self.load(a,m,p).save_state()
            for target in ('head','encoder','projection'):
                b=segmenter()
                with torch.no_grad():
                    if target=='head':b.model.seg_head.bias.add_(1)
                    elif target=='encoder':b.model.res.conv1.weight.add_(.01)
                    else:b.projection[0,0].add_(.1)
                before=COUNTS['backbone_forwards']
                with self.assertRaises(ValueError):self.load(b,m,p)
                other=OnlineHost(b,None if m is None else copy.deepcopy(m))
                with self.assertRaises(ValueError):other.load_state(saved)
                self.assertEqual(COUNTS['backbone_forwards'],before);b.close()
            a.close()
    def test_all_groups_and_C0_legacy_incomplete_packages_rejected(self):
        for group in ('A','B','C','C0'):
            s,m,p=self.make(group);h=self.load(s,m,p)
            with self.assertRaises(ValueError):h.load_state(dict(binding=None if m is None else m.digest(),state=h.state,visits=0,failed=False,first_error=None,ablation=None))
            legacy=copy.deepcopy(p);legacy['binding']=None if m is None else m.digest()
            with self.assertRaises(ValueError):self.load(s,m,legacy)
            s.close()
    def test_policy_and_actual_BN_configuration_rejected_before_forward(self):
        a,m,p=self.make();saved=self.load(a,m,p).save_state()
        for field in ('preprocessing','BN','FiLM'):
            b=segmenter();b.inference_policy[field]='incompatible'
            with self.assertRaises(ValueError):self.load(b,m,p)
            b.close()
        b=segmenter();b.model.up1[1].eps*=2
        with self.assertRaises(ValueError):self.load(b,m,p)
        with self.assertRaises(ValueError):OnlineHost(b,copy.deepcopy(m)).load_state(saved)
        self.assertEqual(b.forwards,0);a.close();b.close()
    def test_method_group_static_basis_scaler_calibration_mismatches(self):
        a,m,p=self.make();saved=self.load(a,m,p).save_state()
        for field in ('basis','scaler','calibration','static','group'):
            candidate=copy.deepcopy(m);weights=copy.deepcopy(p)
            if field=='basis':candidate.basis.mul_(-1)
            elif field=='scaler':candidate.observer.mean.add_(.1)
            elif field=='calibration':candidate.cal_raw.add_(.1);candidate.freeze()
            elif field=='static':candidate.static=True
            else:candidate=method('C');candidate.freeze()
            weights['weights']=copy.deepcopy(candidate.state_dict())
            with self.assertRaises(ValueError):self.load(a,candidate,weights)
            with self.assertRaises(ValueError):OnlineHost(a,candidate).load_state(saved)
        self.assertEqual(a.forwards,0);a.close()
    def test_ablation_explicit_trusted_derivation_and_restore_mismatch(self):
        for group,ablation in [('A','A_ISO_OBS'),('B','B_PRED_ONLY'),('C','C_CONST_R')]:
            s,m,p=self.make(group)
            if group=='C':m.set_constant_variance(torch.ones(8),'cal');p=prepared_artifact(s,m)
            full=self.load(s,m,p);saved=full.save_state()
            with self.assertRaises(ValueError):self.load(s,m,p,ablation)
            derived=copy.deepcopy(p);derived['binding']=deployment_context(p['binding'],ablation)
            h=self.load(s,m,derived,ablation)
            with self.assertRaises(ValueError):h.load_state(saved)
            with self.assertRaises(ValueError):self.load(s,m,derived)
            self.assertEqual(s.forwards,0);s.close()
    def test_tensor_canonical_names_shape_dtype_layout_and_no_address(self):
        x=torch.arange(12,dtype=torch.float32).reshape(3,4)
        expected=tensor_digest([('x',x)])
        self.assertEqual(expected,tensor_digest([('x',x.clone())]))
        self.assertEqual(expected,tensor_digest([('x',x.T.contiguous().T)]))
        for name,t in [('other',x),('x',x.reshape(4,3)),('x',x.double()),('x',x+1)]:self.assertNotEqual(expected,tensor_digest([(name,t)]))
        self.assertEqual(tensor_digest([('scalar',torch.tensor(True))]),tensor_digest([('scalar',torch.tensor(True))]))
    def test_prepared_artifact_records_original_environment_and_rejects_drift(self):
        s,m,p=self.make();source=provenance();source['source_binding_status']='PROCEDURAL_CPU';original=environment(s)
        out=prepared_artifact(s,m,source,original)
        self.assertEqual(out['binding']['payload']['source'],source)
        self.assertIsNone(out['binding']['payload']['source']['checkpoint_file_sha256'])
        self.assertIsNotNone(out['binding']['payload']['environment']['effective_state_sha256'])
        with torch.no_grad():s.model.seg_head.bias.add_(1)
        with self.assertRaises(ValueError):prepared_artifact(s,m,source,original)
        self.assertEqual(s.forwards,0);s.close()
    def test_mutation_guard_no_per_visit_tensor_hash_and_boundary_rehash(self):
        s,m,p=self.make();h=self.load(s,m,p)
        with patch('dpa_ctta.r7_shared.context.tensor_digest',side_effect=AssertionError('no per-visit scan')):
            h.step(pixels());h.step(pixels(1))
        before=s.forwards
        with torch.no_grad():s.model.seg_head.bias.add_(.1)
        with self.assertRaises(ValueError):h.step(pixels())
        self.assertEqual(s.forwards,before);self.assertTrue(h.failed);s.close()
        # Unsupported .data mutations bypass _version but boundaries still rehash.
        s,m,p=self.make();h=self.load(s,m,p);s.model.seg_head.bias.data.add_(.1)
        with self.assertRaises(ValueError):h.save_state()
        s.close()
    def test_tampered_or_wrong_source_spec_context_rejected(self):
        s,m,p=self.make()
        for which in ('sha','science','backbone'):
            bad=copy.deepcopy(p)
            if which=='sha':bad['binding']['sha256']='0'*64
            elif which=='science':bad['binding']['payload']['source']['science_sha256']['A_PSF.json']='0'*64
            else:bad['binding']['payload']['environment']['effective_state_sha256']='0'*64
            with self.assertRaises(ValueError):self.load(s,m,bad)
        s.close()
    def test_full_random_ResUNet_origin_compatibility_and_rejection(self):
        before=COUNTS.copy();cost=HASH_COST.copy();a,m,p=self.make('A',full=True);h=self.load(a,m,p);h.step(pixels());packet=h.save_state()
        b=segmenter(full=True);other=self.load(b,m,p);other.load_state(packet)
        x,_=h.step(pixels(1));y,_=other.step(pixels(1));self.assertTrue(torch.equal(x,y))
        with torch.no_grad():b.model.res.conv1.weight[0,0,0,0].add_(.01)
        calls=COUNTS['backbone_forwards']
        with self.assertRaises(ValueError):self.load(b,m,p)
        with self.assertRaises(ValueError):other.load_state(packet)
        self.assertEqual(COUNTS['backbone_forwards'],calls)
        print('R7_CONTEXT_FULL_RANDOM '+json.dumps(dict(model_parameters=sum(t.numel() for t in a.model.parameters()),counts=dict(COUNTS-before),hash_cost=dict(HASH_COST-cost),same_bytes_cross_instance=True,changed_encoder_rejected=True)),flush=True)
        a.close();b.close()
    def test_actual_prepare_pipeline_emits_six_bound_packages_with_mocked_training(self):
        # Exercise real prepare_tensors packaging/control flow, NOT1000-step
        # training. Expensive source stages are explicitly mocked; zero model F.
        import dpa_ctta.r7_shared.preparation as prep
        from contextlib import ExitStack
        data,o=source_fixture();s=segmenter();source=provenance();source['source_binding_status']='PROCEDURAL_CPU'
        class MockTrainer:
            def __init__(self,seg,m,data,oracle):self.method=m;self.fit_steps=0;self.cal_steps=0
            def fit(self):self.fit_steps=1000
            def start_calibration(self,oracle):pass
            def calibrate(self):self.cal_steps=256;self.method.freeze()
            def validate(self,oracle):return ['MOCKED_NOT_A_VALIDATION_RESULT']
        before=COUNTS['backbone_forwards']
        with ExitStack() as stack:
            for name,value in [('oracle_all',lambda seg,d,f:o[f]),('shared_basis',lambda *a:basis(32)),('a_basis',lambda *a:(basis(16),{'fixture':True})),('scaler_observations',lambda *a:torch.stack([torch.zeros(134),torch.ones(134)])),('SourceTrainer',MockTrainer),('evaluate_oracles',lambda *a:[])]:stack.enter_context(patch.object(prep,name,value))
            result=prepare_tensors(s,data,source_provenance=source)
        self.assertEqual(len(result['models']),6);expected=environment(s)
        for name,p in result['models'].items():
            self.assertEqual(p['binding']['payload']['environment'],expected);self.assertEqual(p['binding']['payload']['source'],source)
            m=method(name[0],name.endswith('STATIC'));m.load_state_dict(p['weights']);m.freeze()
            h=inference_from_tensors(s,name[0],name.endswith('STATIC'),m.basis,p['weights'],p['binding']);self.assertEqual(h.visits,0)
        self.assertEqual(COUNTS['backbone_forwards'],before);s.close()
