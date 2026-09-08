import copy
import json
import random
import unittest
from collections import Counter
from pathlib import Path
import numpy as np
import torch

from dpa_ctta.m2_episodes import make_m2_episodes,validate_coverage,coverage_summary,TRANSFORMS


def old_fixture(task):
    rng=random.Random(20260907);s=list(range(32));rng.shuffle(s)
    q=list(range(40 if task=='fundus' else 64));rng.shuffle(q)
    return [dict(episode=i+1,state_index=s[i%32],query_index=q[i%len(q)],transform=TRANSFORMS[i%4]) for i in range(600)]


class EpisodeTests(unittest.TestCase):
    def test_exact_marginals_crossed_balance_unique_triples_determinism(self):
        for task in ('fundus','polyp'):
            old=old_fixture(task);new=make_m2_episodes(old,task)
            self.assertTrue(validate_coverage(old,new));self.assertEqual(new,make_m2_episodes(old,task))
            self.assertEqual(coverage_summary(old)['unique_triples'],160 if task=='fundus' else 64)
            self.assertEqual(coverage_summary(new)['unique_triples'],600)
            for field in ('query_index','state_index','transform'):
                self.assertEqual(Counter(e[field] for e in old),Counter(e[field] for e in new))

    def test_no_input_mutation_or_global_RNG_consumption(self):
        old=old_fixture('fundus');saved=copy.deepcopy(old)
        py=random.getstate();np_state=np.random.get_state();pt=torch.get_rng_state().clone()
        make_m2_episodes(old,'fundus')
        self.assertEqual(old,saved);self.assertEqual(random.getstate(),py)
        now=np.random.get_state();self.assertEqual(now[0],np_state[0]);np.testing.assert_array_equal(now[1],np_state[1]);self.assertEqual(now[2:],np_state[2:])
        self.assertTrue(torch.equal(pt,torch.get_rng_state()))

    def test_invalid_index_count_order_transform_and_duplicate_rejected(self):
        old=old_fixture('polyp')
        for field,value in [('episode',99),('state_index',-1),('query_index',64),('transform','unknown')]:
            bad=copy.deepcopy(old);bad[0][field]=value
            with self.assertRaises(ValueError):make_m2_episodes(bad,'polyp')
        with self.assertRaises(ValueError):make_m2_episodes(old[:-1],'polyp')
        new=make_m2_episodes(old,'polyp');new[1]=dict(new[0],episode=2)
        with self.assertRaises(ValueError):validate_coverage(old,new)

    def test_new_arm_validation_reuses_M1_and_rejects_old_arm_or_missing_pair(self):
        from dpa_ctta.m2_run import validate_new_arm,base_method
        from dpa_ctta.m2_analysis import validate_new_results
        from test_source_pilot_release import complete_fixture
        fixture,expected=complete_fixture();results={}
        for arm in ('D2','O2'):
            rows=copy.deepcopy(fixture['B'])
            for row in rows:row.update(arm=arm,task='polyp',counts=dict(online_adam=1,memory_pushes=1))
            results[arm]=rows;self.assertTrue(validate_new_arm(rows,expected,'polyp',arm))
        self.assertTrue(validate_new_results(results,expected,'polyp'))
        with self.assertRaises(ValueError):validate_new_results({'D2':results['D2']},expected,'polyp')
        with self.assertRaises(ValueError):base_method('R')
        broken=copy.deepcopy(results);broken['O2'][0]['counts']['memory_pushes']=0
        with self.assertRaises(ValueError):validate_new_results(broken,expected,'polyp')

    def test_old_new_scientific_callables_identical_and_host_mapping_only(self):
        from unittest.mock import patch
        from dpa_ctta import m1_run,m2_run
        self.assertIs(m1_run.OfflineEpisode,m2_run.OfflineEpisode)
        self.assertIs(m1_run.evaluate_after_step,m2_run.evaluate_after_step)
        self.assertIs(m1_run.transform_pixels,m2_run.transform_pixels)
        self.assertIs(m1_run.FixedProxy,m2_run.FixedProxy)
        sentinel=object()
        with patch.object(m2_run,'make_host',return_value=sentinel) as call:
            self.assertIs(m2_run.m2_host('fundus','D2',{},sentinel,'cpu'),sentinel)
            call.assert_called_once_with('fundus','D',{},sentinel,'cpu')

    def test_both_methods_load_same_frozen_list_without_selector_or_old_mutation(self):
        import tempfile
        from unittest.mock import patch
        from dpa_ctta.m2_registration import load_registered,anchor
        from dpa_ctta.source_pilot_release import digest
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);old=root/'old';out=root/'new';old.mkdir();out.mkdir()
            old_rows=old_fixture('fundus');new_rows=make_m2_episodes(old_rows,'fundus')
            (old/'registration.json').write_text(json.dumps({'tasks':{'fundus':{'episodes':old_rows}}}))
            (old/'receipt.run.json').write_text('{}');(old/'history.pt').write_bytes(b'PROCEDURAL_METADATA_ONLY')
            (out/'episodes.json').write_text(json.dumps(new_rows));saved=(old/'registration.json').read_bytes()
            overlay=dict(old_directory=str(old),old_registration_sha256=digest(old/'registration.json'),old_receipt_sha256=digest(old/'receipt.run.json'),asset_checks=[],old_log_identities=[],
                tasks={'fundus':dict(history=anchor(old/'history.pt'),episodes_path=str(out/'episodes.json'),episodes_sha256=digest(out/'episodes.json'))})
            (out/'registration.json').write_text(json.dumps(overlay))
            with patch('dpa_ctta.m1_data.register',side_effect=AssertionError('selector forbidden')):
                d=load_registered(out)[1]['tasks']['fundus']['episodes'];o=load_registered(out)[1]['tasks']['fundus']['episodes']
            self.assertEqual(d,new_rows);self.assertEqual(d,o);self.assertEqual((old/'registration.json').read_bytes(),saved)
            (out/'episodes.json').write_text(json.dumps(old_rows))
            with self.assertRaises(ValueError):load_registered(out)

    def test_seven_arm_interaction_and_missing_old_pair_evidence(self):
        from dpa_ctta.m2_analysis import summarize
        def rows(value):
            return [dict(domain='fixture',metrics=[dict(channel='polyp',dice=value,assd=1.,gt_empty=False,gt_full=False,pred_empty=False,pred_full=False)])]
        arms={arm:rows(value) for arm,value in {'N':.1,'A':.2,'R':.3,'D1':.4,'O1':.5,'D2':.6,'O2':.8}.items()}
        result=summarize(arms,'polyp')
        self.assertEqual(len(result['task_domain_macro_dice_percent']),7)
        self.assertAlmostEqual(result['task_comparisons_pp']['O2-D2'],20.)
        self.assertAlmostEqual(result['task_interaction_pp'],10.)
        self.assertAlmostEqual(result['domains']['fixture']['interaction']['polyp']['dice_delta_pp']['mean'],10.)
        partial=summarize({k:arms[k] for k in ['D2','O2']},'polyp')
        self.assertIsNone(partial['task_comparisons_pp']['O2-O1'])
        self.assertEqual(partial['pooled_content_paired']['O2-O1'],'NOT_AVAILABLE')
