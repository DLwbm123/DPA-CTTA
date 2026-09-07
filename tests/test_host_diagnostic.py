import copy
import json
import os
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from dpa_ctta.host_diagnostic_analysis import distribution, paired, old_analysis
from dpa_ctta.host_diagnostic import DiagnosticPair, close, snapshot, metrics
from dpa_ctta.source_pilot import assemble
from dpa_ctta.proxy_loss import FixedProxy, ProxyProvenance
from test_vptta_host import pixels, proxy, EXTERNAL
from test_source_pilot_release import complete_fixture

H1_EVIDENCE={}


class DiagnosticAnalysisTests(unittest.TestCase):
    @staticmethod
    def record(v=1):
        m=metrics(torch.ones(1,1,8,8),torch.ones(1,1,8,8),'polyp')
        return dict(visit=v,segment='clean',sample_id='S'+str(v),group_id='G'+str(v),
            predictions={k:copy.deepcopy(m) for k in ('N','G','I','U')},reference_metrics=m,
            branches={b:dict(metrics=copy.deepcopy(m),adam_step=v,adam_calls=1) for b in ('A_cf','B_cf','C_cf','A_small','O_native','O_small')},
            counter=[v],adam_step=v,memory_size=v,retrieval_observed=False,equivalence_pass=True,
            counts=dict(reference_steps=v,watched_steps=v,reference_pushes=v,watched_pushes=v,counterfactual_steps=6*v))

    def test_record_coverage_counters_metrics_and_receipt(self):
        from dpa_ctta.host_diagnostic_run import validate_records,validate_receipt,AUTHORIZATION
        rows=[self.record(1),self.record(2)]
        expected=[{k:r[k] for k in ('visit','segment','sample_id','group_id')} for r in rows]
        validate_records(rows,expected,'polyp',(1,2))
        edits=[lambda a:a.pop(),lambda a:a.append(a[0]),lambda a:a.reverse(),
            lambda a:a[0]['counts'].update(counterfactual_steps=0),lambda a:a[0]['predictions'].pop('G'),
            lambda a:a[0]['predictions']['I'][0].update(false_negative_rate=None)]
        for edit in edits:
            altered=copy.deepcopy(rows);edit(altered)
            with self.assertRaises(ValueError):validate_records(altered,expected,'polyp',(1,2))
        with patch.object(Path,'open',side_effect=AssertionError('unauthorized IO')):
            with self.assertRaisesRegex(ValueError,'UNAUTHORIZED'):validate_receipt({},'formal')
        from dpa_ctta import host_diagnostic_run as run
        r=dict(authorization=AUTHORIZATION,stage='formal',physical_gpu=7,reference_commit=run.REFERENCE_COMMIT,tasks=['fundus','polyp'],commit='FIXTURE')
        with patch.object(run.subprocess,'check_output',side_effect=['DIFFERENT','']),patch.object(Path,'open',side_effect=AssertionError('IO after mismatch')):
            with self.assertRaisesRegex(ValueError,'COMMIT'):validate_receipt(r,'formal')
        with patch.object(run.subprocess,'check_output',side_effect=['FIXTURE','']),patch.object(run,'digest',return_value='MISMATCH'):
            with self.assertRaisesRegex(ValueError,'CONFIG'):validate_receipt(r,'formal')

    def test_execution_failure_preserves_first_visit_and_stops(self):
        from dpa_ctta import host_diagnostic_run as run
        from types import SimpleNamespace
        calls=[];outer=self
        class FakePair:
            def __init__(self,*args): self.counts={};self.visit=0
            def identity_check(self,x):return {}
            def native(self,x):
                self.visit+=1;calls.append(self.visit)
                return torch.ones(1,1,8,8),torch.ones(1,1,8,8),0.
            def diagnose(self,x,u,label,cf):
                if self.visit==2: raise ValueError('PROCEDURAL_SECOND_VISIT_FAILURE')
                label();row=outer.record(1);self.counts=row['counts'];return row
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';out=root/'new';old.mkdir();out.mkdir()
            selected=[dict(sample_id='S'+str(i),group_id='G'+str(i),image_path='FIXTURE',mask_path='FIXTURE',image_size=[8,8]) for i in (1,2)]
            (old/'registration.json').write_text(json.dumps({'tasks':{'polyp':{'checkpoint':{'path':'FIXTURE'},'selected':{'proxy':[],'query':selected}}}}))
            from contextlib import ExitStack
            with ExitStack() as stack:
                for name,value in [('check_registered_files',lambda r:None),('environment',lambda:{}),('source_proxy',lambda *a:None),
                    ('read_pixels',lambda *a:torch.zeros(1,3,8,8)),('read_mask',lambda *a:torch.ones(1,1,8,8)),('DiagnosticPair',FakePair)]:
                    stack.enter_context(patch.object(run,name,value))
                stack.enter_context(patch.object(torch,'load',return_value={}))
                stack.enter_context(patch.object(torch.cuda,'reset_peak_memory_stats'))
                stack.enter_context(patch.object(torch.cuda,'max_memory_allocated',return_value=0))
                with self.assertRaisesRegex(ValueError,'SECOND_VISIT'):run.execute({},'formal',old,out)
            self.assertEqual(calls,[1,2]);self.assertEqual(len((out/'formal_polyp.jsonl').read_text().splitlines()),1)
            failure=json.loads((out/'formal.failure.json').read_text());self.assertEqual(failure['completed_current_task'],1)
            self.assertFalse((out/'formal.completion.json').exists())
            self.assertEqual((out/'formal_polyp.jsonl').stat().st_mode & 0o777,0o600)

    def test_frozen_statistics_and_common_assd(self):
        d=distribution([-10,-2,0,1,11])
        self.assertEqual((d['positive'],d['zero'],d['negative'],d['tail_count']),(2,1,2,1))
        self.assertAlmostEqual(d['p10'],-6.8);self.assertEqual(d['worst_decile_mean'],-10)
        a=[dict(dice=.5,assd=None),dict(dice=.7,assd=2)]
        b=[dict(dice=.6,assd=100),dict(dice=.6,assd=1)]
        d=paired(a,b);self.assertEqual(d['assd_common_valid'],1);self.assertEqual(d['assd_delta_px']['mean'],1)

    def test_old_analysis_validates_before_pairing(self):
        arms,expected=complete_fixture()
        reg={'tasks':{'polyp':{'expected_visits':expected,'selected':{'query':expected[:2]}}}}
        # Exercise only the existing external coverage validator: no hidden dictionary deduplication.
        from dpa_ctta.source_pilot import validate_arm
        validate_arm(arms['A'],expected,'polyp','A')
        for edit in (lambda rows:rows.append(rows[0]),lambda rows:rows.pop(),lambda rows:rows.reverse()):
            rows=copy.deepcopy(arms['A']);edit(rows)
            with self.assertRaises(ValueError):validate_arm(rows,expected,'polyp','A')

    def test_area_denominators_and_empty_mask(self):
        logits=torch.full((1,2,8,8),-10.)
        logits[:,1,:4,:4]=10
        m=metrics(logits,torch.zeros_like(logits),'fundus')
        self.assertIsNone(m[1]['false_negative_rate'])
        self.assertEqual(m[1]['false_positive_denominator'],64)
        self.assertEqual(m[1]['pred_oc_outside_od_ratio'],1)
        self.assertIsNone(m[0]['assd'])


@unittest.skipUnless(EXTERNAL,'pinned checkout required')
class DiagnosticNativeTests(unittest.TestCase):
    def test_two_tasks_native_branch_label_and_future_isolation(self):
        for task in ('fundus','polyp'):
            with self.subTest(task=task):
                source=assemble(task,'A',None)
                state={k:v.detach().clone() for k,v in source.model.state_dict().items()};del source
                one=proxy(task);four=FixedProxy(one.pixel_rgb.repeat(4,1,1,1),one.mask.repeat(4,1,1,1),
                                                one.signed_distance.repeat(4,1,1,1),ProxyProvenance.FIXTURE)
                pair=DiagnosticPair(task,state,four)
                identity=pair.identity_check(pixels(task))
                ref,u,difference=pair.native(pixels(task))
                observed=[]
                branch_states=[]
                handle=pair.diagnostic.optimizer.register_step_post_hook(lambda opt,args,kw:branch_states.append(
                    (pair.diagnostic.prompt.data_prompt.detach().clone(),copy.deepcopy(opt.state_dict()))))
                def label():
                    self.assertEqual(pair.counts['counterfactual_steps'],4)
                    observed.append('after_nonoracle');return one.mask
                first=pair.diagnose(pixels(task),u,label,True)
                before=snapshot(pair.watched)
                second=pair.diagnose(pixels(task),u,lambda:1-one.mask,True)
                handle.remove()
                for j in range(4): close(branch_states[j],branch_states[j+6],exact=True)
                close(before,snapshot(pair.watched),exact=True)
                for b in ('A_cf','B_cf','C_cf','A_small'):
                    self.assertEqual(first['branches'][b]['update_norm'],second['branches'][b]['update_norm'])
                    self.assertEqual(first['branches'][b]['distance_from_A_cf'],second['branches'][b]['distance_from_A_cf'])
                self.assertNotEqual(first['gradients']['raw_norm']['Q'],second['gradients']['raw_norm']['Q'])
                ref,u,difference=pair.native(pixels(task,1))
                pair.diagnose(pixels(task,1),u,lambda:one.mask,False)
                counts=pair.finish()
                self.assertEqual(counts['reference_steps'],2);self.assertEqual(counts['watched_pushes'],2)
                self.assertEqual(counts['counterfactual_steps'],12)
                self.assertEqual(counts['proxy_images'],8)
                self.assertEqual(first['branches']['A_cf']['distance_from_A_cf'],0)
                H1_EVIDENCE[task]=dict(identity=identity,native_equivalent=True,labels_isolated=True,
                                     branch_isolation=True,future_equivalent=True,counts=counts)
                del pair,ref,u,state,four,first,second,before
