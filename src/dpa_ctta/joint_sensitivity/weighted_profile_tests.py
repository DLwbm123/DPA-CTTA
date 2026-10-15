"""CPU checks of bounded profiling conditions, derivatives and formal-run refusal."""
import unittest
import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch
import torch
from .weighted_profile import conditions, derivatives, summary, run
from .weighted_profile_report import main as report
from .factorial import candidates as factorial_candidates
from .shared_span import candidates as shared_candidates


class Tests(unittest.TestCase):
    def test_fixed_controls_and_weighted_background(self):
        configs = conditions()
        self.assertEqual([c['id'] for c in configs],
                         ['C','C_JS_SHARED','CW_LSO_JS_SHARED','C_JS_ACQ_WEIGHTED'])
        self.assertEqual(configs[:2], [factorial_candidates()[0], factorial_candidates()[2]])
        self.assertEqual(configs[2], shared_candidates()[-1])
        configs[3]['id'] = configs[1]['id']
        del configs[3]['joint_filter']
        self.assertEqual(configs[3], configs[1])

    def test_actual_direction_not_pre_write_prediction(self):
        result = derivatives(torch.tensor([1.,2.]), torch.tensor([-1.,-1.]), torch.tensor([-2.,0.]))
        self.assertEqual(result, dict(proposal_dot=-3., actual_dot=-2., dot_change=1.))
        with self.assertRaises(ValueError):
            derivatives(torch.ones(1), torch.ones(1), torch.tensor([float('nan')]))

    def test_summary_preserves_adverse_values(self):
        self.assertEqual(summary([-3., 1., 2.]), dict(count=3, mean=0., median=1., min=-3., max=2.))

    def test_formal_or_expanded_scope_rejected_before_launch(self):
        for options in (dict(formal_experiment_authorized=True, arrivals=32, warmup=4),
                        dict(formal_experiment_authorized=False, arrivals=1951, warmup=4)):
            with self.assertRaises(ValueError): run(dict(output_root='/unused', **options))

    def test_report_matches_device_and_order_and_requires_retirement(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, PROFILE_ROOT=tmp):
            root=Path(tmp)
            for folder in ('results','processes','public'): (root/folder).mkdir()
            frozen=json.loads((Path(__file__).resolve().parents[3]/'configs/acquisition_weighted_profile_only.json').read_text())
            frozen['runner_source_commit']='synthetic'
            (root/'FROZEN.json').write_text(json.dumps(frozen))
            (root/'RUN_STATE.json').write_text(json.dumps(dict(status='COMPLETE', completed=8)))
            (root/'RESOURCE_LEDGER.json').write_text(json.dumps(dict(attempts=[],gpu_worker_seconds=1,wall_seconds=1,charging='synthetic')))
            for order, gpu in ((0,5),(1,7)):
                for index, condition in enumerate(frozen['conditions']):
                    value=dict(condition=condition,physical_gpu=gpu,order=order,arrivals=32,
                               wall_seconds={'mean':float(index+1)},cuda_event_seconds={'mean':float(index+1)},
                               peak_allocated_bytes=1024**2,peak_reserved_bytes=2*1024**2)
                    (root/'results'/f'{order}_{index}.json').write_text(json.dumps(value))
            process=root/'processes/a.json';process.write_text(json.dumps(dict(active=True)))
            with self.assertRaises(ValueError): report()
            process.write_text(json.dumps(dict(active=False)))
            with contextlib.redirect_stdout(io.StringIO()): report()
            output=json.loads((root/'public/REPORT.json').read_text())
            self.assertEqual(output['actual_arrivals'],256)
            self.assertEqual(len(output['same_device_ratios']),6)
            self.assertEqual([p['wall_time_ratio'] for p in output['same_device_ratios']], [4.,2.,4/3]*2)


if __name__ == '__main__': unittest.main()
