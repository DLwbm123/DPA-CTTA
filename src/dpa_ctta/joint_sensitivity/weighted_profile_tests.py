"""CPU checks of bounded profiling conditions, derivatives and formal-run refusal."""
import unittest
import torch
from .weighted_profile import conditions, derivatives, summary, run
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


if __name__ == '__main__': unittest.main()
