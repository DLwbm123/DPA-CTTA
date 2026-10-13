"""CPU checks for the retirement gate and anonymous profile aggregation."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from .profile import r44_retired, summaries, alive, main
from ..r10_12h_core.run import save


class Tests(unittest.TestCase):
    def test_retirement_requires_complete_coverage_and_no_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'processes').mkdir()
            state = dict(status='RUNNING_FIXED_MATRIX', jobs={str(i): 'COMPLETE' for i in range(24)})
            save(root/'RUN_STATE.json', state)
            self.assertFalse(r44_retired(root))
            state['status'] = 'COMPLETE'; save(root/'RUN_STATE.json', state)
            save(root/'LAUNCH.json', dict(pid=100)); save(root/'watchdog.json', dict(supervisor_pid=101))
            save(root/'processes/a.json', dict(pid=102, active=False, start_ticks='12'))
            with patch('dpa_ctta.joint_sensitivity.profile.alive', return_value=False):
                self.assertTrue(r44_retired(root))
                save(root/'processes/a.json', dict(pid=102, active=True))
                self.assertFalse(r44_retired(root))
                save(root/'processes/a.json', dict(pid=102, active=False))
                state['jobs']['0'] = 'NOT_RUN'; save(root/'RUN_STATE.json', state)
                with self.assertRaises(ValueError): r44_retired(root)
            state['jobs']['0'] = 'COMPLETE'; save(root/'RUN_STATE.json', state)
            with patch('dpa_ctta.joint_sensitivity.profile.alive', side_effect=lambda pid, *_: pid == 101):
                self.assertFalse(r44_retired(root))
            state['jobs'] = {str(i): 'COMPLETE' for i in range(30)}
            save(root/'RUN_STATE.json', state)
            with patch('dpa_ctta.joint_sensitivity.profile.alive', return_value=False):
                self.assertTrue(r44_retired(root, 30))
                with self.assertRaises(ValueError): r44_retired(root)
            state['status'] = 'INCOMPLETE'; save(root/'RUN_STATE.json', state)
            with self.assertRaises(RuntimeError): r44_retired(root)

    def test_summary_pairs_same_device_without_content_identifiers(self):
        common = dict(timed_arrivals=8, physical_gpu=0, initialization_seconds=1.,
                      peak_allocated_bytes=1024**2, peak_reserved_bytes=2*1024**2,
                      trainable_BN_parameters=8, counts_per_image={}, sensitivity_ranks=[], evidence_channel_means=[])
        base = dict(common, condition='CW_LSO', seconds=[1.]*8, cuda_event_seconds=[.5]*8)
        joint = dict(common, condition='CW_LSO_JS', seconds=[2.]*8, cuda_event_seconds=[1.]*8,
                     peak_allocated_bytes=2*1024**2, sensitivity_ranks=[2]*12,
                     evidence_channel_means=[[.1, .2]]*12)
        rows, ratios = summaries([dict(status='COMPLETE', result=p) for p in (base, joint)]+[dict(status='INCOMPLETE')])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1]['rank_nonzero_arrivals'], 12)
        self.assertEqual(ratios[0]['wall_time_ratio'], 2.)
        self.assertEqual(ratios[0]['allocated_ratio'], 2.)
        self.assertNotIn('seconds', rows[0])
        self.assertEqual(summaries([dict(status='INCOMPLETE')]), ([], []))
        shared = dict(base, condition='CW_LSO_JS_SHARED')
        descent = dict(joint, condition='CW_LSO_JS_SHARED_DESCENT', descent_constraint_triggers=5)
        rows, ratios = summaries([dict(status='COMPLETE', result=p) for p in (shared, descent)],
            ('CW_LSO_JS_SHARED', 'CW_LSO_JS_SHARED_DESCENT'))
        self.assertEqual(ratios[0]['wall_time_ratio'], 2.)
        self.assertEqual(rows[1]['descent_constraint_triggers'], 5)

    def test_missing_pid_is_not_alive(self):
        with patch.object(Path, 'read_text', side_effect=FileNotFoundError):
            self.assertFalse(alive(100))

    def test_formal_worker_and_supervisor_modes_are_rejected(self):
        with patch.dict('os.environ', RUN_MODE='worker', RUN_PHASE='online_CW_LSO_JS'):
            with self.assertRaises(ValueError): main()
        with patch.dict('os.environ', RUN_MODE='supervise'):
            with self.assertRaises(ValueError): main()


if __name__ == '__main__': unittest.main()
