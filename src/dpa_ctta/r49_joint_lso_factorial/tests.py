import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from . import run
from ..joint_sensitivity.factorial import Host, candidates
from ..r20_model_only_search.report import bootstrap_pair


class Tests(unittest.TestCase):
    def rows(self):
        rows = []
        for seed in (20260907, 17011, 29009):
            for order in (0, 1):
                for content in (0, 1, 2, 3):
                    for condition, value in [('C', .7), ('C_LSO', .71), ('C_JS_SHARED', .702), ('C_LSO_JS_SHARED', .71)]:
                        rows.append(dict(condition=condition, seed=seed, order=order, content=str(content),
                                         domain='A' if content < 2 else 'B', role='SEARCH',
                                         metrics=[dict(channel=c, dice=value + content * .01) for c in ('OD', 'OC')]))
        return rows

    def test_coupled_four_arm_interaction_and_dense_expected_value(self):
        rows = self.rows(); before = copy.deepcopy(rows)
        result, ci, negative = bootstrap_pair(run.interaction_rows(rows), 'J_EFFECT_WITH_LSO',
            'J_EFFECT_WITHOUT_LSO', [20260907, 17011, 29009], role='SEARCH')
        self.assertEqual(rows, before)
        self.assertEqual(result['status'], 'COMPLETE')
        for key in ('delta_pp', 'CI_low_pp', 'CI_high_pp'):
            self.assertAlmostEqual(result[key], -.2)
        self.assertEqual(len(ci), 4)
        self.assertTrue(negative)

    def test_incomplete_or_misaligned_quartets_do_not_become_evaluable(self):
        rows = self.rows()
        self.assertEqual(run.interaction_rows(rows[:-1]), [])
        rows[1]['domain'] = 'wrong'
        with self.assertRaises(ValueError): run.interaction_rows(rows)
        with self.assertRaises(ValueError): run.interaction_rows(self.rows() * 2)

    def test_primary_gate_interpretations_and_no_secondary_replacement(self):
        p = dict(status='COMPLETE', delta_pp=.2, order_delta_pp=[.2, .2],
                 positive_trajectories=6, imageweighted_delta_pp=.2,
                 worst_seed_averaged_cell_pp=-1, CI_low_pp=.1, CI_high_pp=.3)
        # passes() accepts the same JSON order vector emitted to CSV.
        p['order_delta_pp'] = '[0.2,0.2]'
        zero = dict(p, delta_pp=0., CI_low_pp=-.01, CI_high_pp=.01)
        negative = dict(p, delta_pp=-.2, CI_low_pp=-.3, CI_high_pp=-.1)
        self.assertEqual(run.interpretation(p, zero, negative), 'REDUNDANCY_COMPATIBLE')
        self.assertEqual(run.interpretation(p, negative, negative), 'INTERFERENCE_COMPATIBLE')
        self.assertEqual(run.interpretation(zero, p, p), 'NO_REGISTERED_STANDALONE_SIGNAL')
        self.assertEqual(run.interpretation(p, p, p), 'J_BENEFIT_IN_BOTH_BACKGROUNDS')
        self.assertEqual(run.interpretation(p, p, dict(status='INCOMPLETE')), 'NOT_EVALUABLE')
        self.assertFalse(run.signal(dict(p, CI_low_pp=0)))
        self.assertTrue(run.signal(dict(p, order_delta_pp=[.2, .2])))

    def test_dispatch_uses_new_identity_factory_and_retained_runtime(self):
        self.assertEqual(len(candidates()), 5)
        with patch.object(run.retained, 'main') as dispatch:
            run.main(); dispatch.assert_called_once_with()
        self.assertEqual(run.retained.ID, 'R49_JOINT_LSO_FACTORIAL')
        self.assertIs(run.retained.Host, Host)
        self.assertIs(run.retained.report, run.report)

    def test_terminal_incomplete_report_keeps_unknown_results(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'public').mkdir(); (root/'processes').mkdir()
            protocol = root/'protocol.md'; protocol.write_text('CPU fixture only')
            c = dict(output_root=str(root), protocol_path=str(protocol), candidates=candidates(),
                     seeds=[20260907, 17011, 29009], jobs=[])
            ledger = dict(attempts=[], gpu_worker_seconds=0, cpu_worker_seconds=0)
            with patch.object(run.retained.base, 'ledger', return_value=ledger), patch.object(
                    run.retained.scoring, 'load_rows', return_value=[]):
                run.report(c, dict(status='INCOMPLETE', jobs={}, reason='fixture'))
            decision = json.loads((root/'public/FINAL_DECISION.json').read_text())
            self.assertEqual(decision['interpretation'], 'NOT_EVALUABLE')
            self.assertFalse(decision['strong_development_signal'])
            self.assertIn('NA', (root/'public/REPORT.md').read_text())

    def test_complete_report_uses_registered_primary_and_interaction(self):
        rows = self.rows()
        for row in list(rows):
            if row['condition'] == 'C':
                extra = copy.deepcopy(row); extra['condition'] = 'CW_LSO'
                for metric in extra['metrics']: metric['dice'] += .02
                rows.append(extra)
        review = copy.deepcopy(rows)
        for row in review:
            row['role'] = 'SEALED_REVIEW'; row['content'] = 'review_' + row['content']
        rows += review
        for row in rows:
            row.update(subset='remaining_dev', diagnostics={}, containment_violations=0,
                       empty_foreground=[False, False])
            for metric in row['metrics']:
                metric.update(assd=None, pred_empty=False, gt_empty=False)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root/'public').mkdir(); (root/'processes').mkdir()
            protocol = root/'protocol.md'; protocol.write_text('CPU fixture only')
            jobs = [dict(id=str(i)) for i in range(30)]
            (root/'FINAL_JOBS.json').write_text(json.dumps(jobs))
            (root/'scores/final').mkdir(parents=True)
            for job in jobs: (root/'scores/final'/(job['id']+'.complete.json')).write_text('{}')
            c = dict(output_root=str(root), protocol_path=str(protocol), candidates=candidates(),
                     seeds=[20260907, 17011, 29009], jobs=jobs)
            with patch.object(run.retained.base, 'ledger', return_value=dict(attempts=[], gpu_worker_seconds=0,
                    cpu_worker_seconds=0)), patch.object(run.retained.scoring, 'load_rows', return_value=rows):
                run.report(c, dict(status='COMPLETE', jobs={j['id']: 'COMPLETE' for j in jobs}))
            decision = json.loads((root/'public/FINAL_DECISION.json').read_text())
            self.assertEqual(decision['primary'], 'C_JS_SHARED')
            self.assertTrue(decision['strong_development_signal'])
            self.assertEqual(decision['interpretation'], 'REDUNDANCY_COMPATIBLE')
