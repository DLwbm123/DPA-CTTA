import copy
import unittest
from unittest.mock import patch
import torch
from .method import Host, candidates, QualifiedHost
from .run import stage_success
from ..joint_sensitivity.descent import candidates as qualified_candidates
from ..r19_model_only.method import equal
from ..r20_model_only_search.tests import Tiny


class Tests(unittest.TestCase):
    def test_frozen_controls_and_four_baselines_required(self):
        q = qualified_candidates()
        self.assertEqual(candidates(), q)
        self.assertEqual([x['id'] for x in q], ['C','W','CW_LSO','CW_LSO_JS_SHARED','CW_LSO_JS_SHARED_DESCENT'])
        pairs = [dict(status='COMPLETE', baseline=b, delta_pp=.31 if b in ('C','W') else .11,
            order_delta_pp='[0.1,0.2]', positive_trajectories=6, imageweighted_delta_pp=.1,
            worst_seed_averaged_cell_pp=-1.) for b in ('C','W','CW_LSO','CW_LSO_JS_SHARED')]
        self.assertTrue(stage_success(dict(status='COMPLETE'), pairs))
        self.assertFalse(stage_success(dict(status='SCORING'), pairs))
        self.assertFalse(stage_success(dict(status='COMPLETE'), pairs[:-1]))
        self.assertFalse(stage_success(dict(status='COMPLETE'), pairs[:-1]+[pairs[0]]))
        for key, value in [('delta_pp', .09), ('order_delta_pp', '[-0.1,0.3]'),
                           ('positive_trajectories', 4), ('worst_seed_averaged_cell_pp', -2.01)]:
            changed = copy.deepcopy(pairs); changed[-1][key] = value
            self.assertFalse(stage_success(dict(status='COMPLETE'), changed))

    def test_real_cpu_host_guard_preserves_qualified_execution_and_counts(self):
        torch.set_num_threads(2); torch.manual_seed(42); model = Tiny()
        x = torch.linspace(.01, .99, 3*512*512).reshape(1,3,512,512)
        for config in candidates()[-2:]:
            a = Host(None, config, 17011, 'synthetic', device='cpu', model=copy.deepcopy(model))
            b = QualifiedHost(None, config, 17011, 'synthetic', device='cpu', model=copy.deepcopy(model))
            try:
                za, da = a.step(x); sa = a.snapshot()
                zb, db = b.step(x); sb = b.snapshot()
                self.assertTrue(torch.equal(za, zb)); self.assertTrue(equal(da, db)); self.assertTrue(equal(sa, sb))
                self.assertEqual(da['native'][0]['counts'], dict(forwards=10,backwards=5,base_adam=1,perturb=0,restore=0))
                bad = copy.deepcopy(da); bad['diagnostics']['joint'].update(descent_proposal_dot=-1.,descent_actual_dot=-.5,filtered_metric_norm=2.,proposal_metric_norm=1.)
                with patch.object(QualifiedHost, 'step', return_value=(za, bad)):
                    with self.assertRaises(ValueError): a.step(x)
            finally: a.close(); b.close()


if __name__ == '__main__': unittest.main()
