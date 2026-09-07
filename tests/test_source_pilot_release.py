import copy
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from dpa_ctta.source_pilot import (ARMS, ArmFailure, _run_arm, expected_visits, summarize, validate_arm)
from dpa_ctta.source_pilot_release import (AUTHORIZATION,CONFIG_SHA,execute_task,validate_receipt)
from dpa_ctta.integrations.ctta_suite import REFERENCE_COMMIT
from test_vptta_host import pixels, same, native_snapshot, EXTERNAL
from dpa_ctta.hosts.vptta import VPTTAHost,model_input_from_pixels

RELEASE_EVIDENCE={}


def metric(dice=.5):
    return dict(channel='polyp',dice=dice,region=.2,boundary=.01,assd=1.,boundary_defined=True,
                gt_empty=False,gt_full=False,pred_empty=False,pred_full=False)


def complete_fixture():
    expected=expected_visits([dict(sample_id='PROCEDURAL_S1',group_id='PROCEDURAL_G1'),dict(sample_id='PROCEDURAL_S2',group_id='PROCEDURAL_G2')])
    results={a:[] for a in ARMS}
    for arm in ARMS:
        for row in expected:
            adapting=arm!='N';visit=row['visit']
            results[arm].append(dict(row,metrics=[metric({'N':.4,'A':.5,'B':.6,'C':.7}[arm])],
                prompt_state_delta_norm=.1 if adapting else 0.,optimizer_update_norm=.01 if adapting else 0.,
                pipeline_elapsed_seconds=.02,host_step_elapsed_seconds=.01,source_versions_unchanged=True,optimizer_state_finite=True,
                adam_step=visit if adapting else 0,native_counts=[visit] if adapting else [],
                memory_size=min(visit,41) if adapting else 0,optimizer_steps_this_visit=int(adapting)))
    return results,expected


class FixtureHost:
    def __init__(self,arm,fail_at=None,initialization_only=False):
        self.device=torch.device('cpu');self.model=torch.nn.Identity();self.arm=arm;self.visits=0
        self.fail_at=fail_at;self.initialization_only=initialization_only
        if arm!='N':
            self.prompt=SimpleNamespace(data_prompt=torch.nn.Parameter(torch.tensor([5.])))
            self.optimizer=torch.optim.Adam([self.prompt.data_prompt],lr=.01)
            self.adabn=torch.nn.Identity
            self.model.sample_num=0
            self.memory_bank=SimpleNamespace(get_size=lambda:min(self.visits,41))
            self.last_proxy_loss=None
    def step(self,x):
        self.visits+=1
        if self.visits==self.fail_at: raise ValueError('PROCEDURAL_B_SECOND_VISIT_FAILURE')
        if hasattr(self,'prompt'):
            with torch.no_grad(): self.prompt.data_prompt.fill_(1.)
            self.prompt.data_prompt.grad=torch.zeros_like(self.prompt.data_prompt)
            if not self.initialization_only: self.optimizer.step()
            self.model.sample_num+=1
        return torch.zeros(1,1,7,9)


class ReleaseTests(unittest.TestCase):
    def test_expected_stream_rejects_duplicate_missing_order_channels_lifecycle_nonfinite(self):
        good,expected=complete_fixture();normal=summarize(good,expected,'polyp')
        self.assertAlmostEqual(normal['all']['C-B']['polyp']['dice_delta'],.1)
        self.assertAlmostEqual(normal['all']['B-A']['polyp']['dice_delta_percentage_points'],10)
        self.assertAlmostEqual(normal['all']['arm_metrics']['B']['polyp']['dice'],.6)
        edits=[lambda x:x['B'].append(copy.deepcopy(x['B'][0])),lambda x:x.pop('N'),
            lambda x:[x[a].pop(0) for a in ARMS],lambda x:x['B'].reverse(),
            lambda x:x['B'][0]['metrics'].append(metric()),lambda x:x['B'][0]['metrics'].clear(),
            lambda x:x['B'][0]['metrics'][0].update(channel='OD'),
            lambda x:x['B'][0]['metrics'][0].update(dice=float('nan')),
            lambda x:x['B'][0]['metrics'][0].update(assd=-1),
            lambda x:x['B'][0].update(adam_step=5),lambda x:x['B'][0].update(optimizer_state_finite=False)]
        for edit in edits:
            value=copy.deepcopy(good);edit(value)
            with self.assertRaises(ValueError):summarize(value,expected,'polyp')
        RELEASE_EVIDENCE['coverage']={'counterexamples_rejected':len(edits),'normal_mean_and_delta_unchanged':True,'N_required':True}

    def test_failure_retains_N_A_and_B_prefix(self):
        rows=[dict(sample_id='PROCEDURAL_S1',group_id='PROCEDURAL_G1'),dict(sample_id='PROCEDURAL_S2',group_id='PROCEDURAL_G2')]
        registration=dict(selected={'query':rows},expected_visits=expected_visits(rows))
        with tempfile.TemporaryDirectory(prefix='dpa_release_fixture_') as tmp:
            out=Path(tmp)
            factory=lambda task,arm,*a,**kw:FixtureHost(arm,fail_at=2 if arm=='B' else None)
            with self.assertRaises(ArmFailure):
                execute_task('polyp',registration,{},None,out,factory=factory,pixel_reader=lambda r:torch.ones(1,3,7,9)*.5,evaluator=lambda p,r:[metric()])
            self.assertEqual(len((out/'polyp_N.jsonl').read_text().splitlines()),8)
            self.assertEqual(len((out/'polyp_A.jsonl').read_text().splitlines()),8)
            self.assertEqual(len((out/'polyp_B.jsonl').read_text().splitlines()),1)
            self.assertTrue((out/'polyp_N.completion.json').exists());self.assertTrue((out/'polyp_A.completion.json').exists())
            self.assertFalse((out/'polyp_B.completion.json').exists());self.assertFalse((out/'polyp_C.jsonl').exists())
            failure=json.loads((out/'failure.json').read_text())
            self.assertEqual((failure['task'],failure['arm'],failure['visit'],failure['stage'],failure['completed_records']),('polyp','B',2,'host_step',1))
        RELEASE_EVIDENCE['failure']={'completed_N_A_visits':[8,8],'B_prefix':1,'B_failing_visit':2,'C_not_started':True,'exception_propagated_to_nonzero_entry':True}

    def test_initialization_delta_is_not_adam_update(self):
        host=FixtureHost('A',initialization_only=True)
        result=_run_arm(host,[dict(sample_id='PROCEDURAL',group_id='PROCEDURAL')],lambda r:torch.ones(1,3,7,9)*.5,lambda p,r:[metric()])
        self.assertEqual(result[0]['prompt_state_delta_norm'],4.)
        self.assertEqual(result[0]['optimizer_update_norm'],0.)
        self.assertEqual(result[0]['optimizer_steps_this_visit'],0)
        RELEASE_EVIDENCE['observation']={'initialization_state_delta':4,'actual_adam_update':0,'no_extra_step':True}

    def test_receipt_rejects_before_real_io_and_binds_commit_config_source(self):
        from dpa_ctta import source_pilot_release as release
        with patch.object(Path,'open',side_effect=AssertionError('unauthorized IO')):
            with self.assertRaisesRegex(ValueError,'UNAUTHORIZED'): validate_receipt({},'pilot')
        r=dict(authorization=AUTHORIZATION,stage='pilot',config_sha256=CONFIG_SHA,reference_commit=REFERENCE_COMMIT,
               tasks=['fundus','polyp'],physical_gpu=4,gpu_uuid='GPU-PROCEDURAL',commit='PROCEDURAL_COMMIT',
               source_spec_path='PROCEDURAL_SPEC',source_spec_sha256='PROCEDURAL_SUMMARY')
        with patch.object(release.subprocess,'check_output',side_effect=['PROCEDURAL_COMMIT','']),patch.object(release,'digest',return_value=CONFIG_SHA),patch.object(release,'validate_config'),patch.object(release,'checkout_root'),patch.object(Path,'read_text',return_value='{}'):
            with self.assertRaisesRegex(ValueError,'SOURCE_RECEIPT_SUMMARY_MISMATCH'):validate_receipt(r,'pilot')
        for field,value in [('commit','WRONG'),('config_sha256','WRONG')]:
            altered=dict(r,**{field:value})
            with patch.object(release.subprocess,'check_output',return_value='PROCEDURAL_COMMIT'),patch.object(Path,'open',side_effect=AssertionError('IO after mismatch')):
                with self.assertRaises(ValueError):validate_receipt(altered,'pilot')
        RELEASE_EVIDENCE['receipt']={'unauthorized_io_attempts':0,'mismatched_commit_config_source_summary_rejected':True,'no_old_guard_bypass':True}

    @unittest.skipUnless(EXTERNAL,'pinned checkout required')
    def test_readonly_optimizer_hooks_preserve_full_native_output_state_rng(self):
        for task in ('fundus','polyp'):
            from dpa_ctta.source_pilot import SEGMENTS,transform_pixels
            torch.manual_seed(56);direct=VPTTAHost(task)
            torch.manual_seed(56);watched=VPTTAHost(task)
            rng=torch.get_rng_state().clone()
            a=[direct.step(transform_pixels(pixels(task,4),segment)) for segment in SEGMENTS]
            after=torch.get_rng_state().clone();torch.set_rng_state(rng);b=[]
            records=_run_arm(watched,[dict(sample_id='PROCEDURAL',group_id='PROCEDURAL')],lambda r:pixels(task,4),lambda pred,r:[],capture=lambda pred,host:b.append(pred))
            same(self,a,b);same(self,native_snapshot(direct),native_snapshot(watched));self.assertTrue(torch.equal(after,torch.get_rng_state()))
            self.assertTrue(all(r['optimizer_steps_this_visit']==1 for r in records))
            RELEASE_EVIDENCE[task+'_hooks']={'full_native_four_steps':True,'output_prompt_adam_memory_rng_equal':True}
            del direct,watched,a,b
