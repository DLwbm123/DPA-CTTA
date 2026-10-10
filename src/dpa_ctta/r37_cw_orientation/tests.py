import copy
import unittest
import torch
from .method import Host, candidates
from .run import public, passes
from ..r20_model_only_search.method import Host as ExistingHost
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal
from ..r36_incremental_modules.method import candidates as old_candidates


class Tests(unittest.TestCase):
    def test_frozen_controls_and_disabled_orientation_parity(self):
        torch.set_num_threads(2); torch.manual_seed(42)
        model = Tiny(); cs = candidates()
        self.assertEqual(cs[:2], old_candidates()[:2])
        self.assertEqual(cs[2]['host'], 'C')
        self.assertEqual(cs[2]['final_lr_multiplier'], 1.)
        self.assertEqual(cs[3]['module_strength'], .1)
        x = torch.linspace(.01, .99, 3*512*512).reshape(1,3,512,512)
        disabled = dict(cs[3], module_strength=0.)
        a = Host(None, disabled, 17011, 'same', device='cpu', model=copy.deepcopy(model))
        b = ExistingHost(None, cs[2], 17011, 'same', device='cpu', model=copy.deepcopy(model))
        try:
            za, _ = a.step(x); zb, _ = b.step(x)
            self.assertTrue(torch.equal(za, zb))
            sa, sb = a.snapshot(), b.snapshot()
            for k in ('parameters', 'gradients', 'adam', 'grata', 'native_rng'):
                self.assertTrue(equal(sa[k], sb[k]), k)
            a.check_frozen(True); b.check_frozen(True)
        finally: a.close(); b.close()

    def test_public_receipts_and_decision_boundaries(self):
        self.assertEqual(public({'result': {'identity': {'content': 'secret'},
            'prediction_sha256': 'secret', 'visits': 1951}, 'path': '/remote-home/private'}),
            {'result': {'visits': 1951}})
        pair = dict(status='COMPLETE', delta_pp='0.3', order_delta_pp='[0.1,0.5]',
                    positive_trajectories='5', imageweighted_delta_pp='0',
                    worst_seed_averaged_cell_pp='-2')
        self.assertTrue(passes(pair, .3))
        self.assertFalse(passes(dict(pair, order_delta_pp='[0,0.6]'), .3))
        self.assertFalse(passes(dict(pair, positive_trajectories='4'), .3))


if __name__ == '__main__': unittest.main()
