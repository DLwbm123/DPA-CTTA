import unittest
from .method import candidates
from .run import stage_success
from ..joint_sensitivity.method import candidates as original_candidates


class Tests(unittest.TestCase):
    def test_frozen_controls_and_all_four_comparisons_required(self):
        self.assertEqual(candidates()[:4], original_candidates())
        self.assertEqual(candidates()[-1]['joint_basis'], 'shared')
        rows = [dict(status='COMPLETE', baseline=b, delta_pp=.31 if b in ('C','W') else .11,
                     order_delta_pp='[0.2,0.1]', positive_trajectories=6,
                     imageweighted_delta_pp=.1, worst_seed_averaged_cell_pp=-1.)
                for b in ('C','W','CW_LSO','CW_LSO_JS')]
        self.assertTrue(stage_success(dict(status='COMPLETE'), rows))
        self.assertFalse(stage_success(dict(status='SCORING'), rows))
        self.assertFalse(stage_success(dict(status='COMPLETE'), rows[:3]))
        rows[-1]['delta_pp']=.09
        self.assertFalse(stage_success(dict(status='COMPLETE'), rows))
        rows[-1]['delta_pp']=.11; rows[1]['worst_seed_averaged_cell_pp']=-2.01
        self.assertFalse(stage_success(dict(status='COMPLETE'), rows))
