import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from .method import Host, candidates
from . import run
from ..joint_sensitivity.factorial import Host as FactorialHost
from ..joint_sensitivity.acquisition_weighted import Host as WeightedHost
from ..r20_model_only_search.tests import Tiny
from ..r19_model_only.method import equal


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);torch.manual_seed(42)
        cls.model=Tiny();cls.x=torch.linspace(.01,.99,3*512*512).reshape(1,3,512,512)

    def host(self, config, factory=Host):
        return factory(None,config,17011,'synthetic',device='cpu',model=copy.deepcopy(self.model))

    def test_four_configs_two_step_bitwise_updates_rng_and_counts(self):
        self.assertEqual([c['id'] for c in candidates()],['C','C_JS_SHARED','C_JS_ACQ_WEIGHTED','CW_LSO'])
        for config in candidates():
            factory=WeightedHost if config['id']=='C_JS_ACQ_WEIGHTED' else FactorialHost
            a,b=self.host(config),self.host(config,factory)
            try:
                for image in (self.x,self.x.roll(3,-1)):
                    za,da=a.step(image);sa=a.snapshot()
                    zb,db=b.step(image);sb=b.snapshot()
                    measured=da['diagnostics'].pop('update_direction')
                    self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(da,db));self.assertTrue(equal(sa,sb))
                    joint=bool(config.get('joint_strength'))
                    self.assertEqual(da['native'][0]['counts'],dict(forwards=10 if joint else 8,backwards=5 if joint else 1,base_adam=1,perturb=0,restore=0))
                    if not joint:self.assertEqual(measured['dot_change'],0)
                a.check_frozen(True);b.check_frozen(True)
            finally:a.close();b.close()

    def test_disabled_new_and_shared_are_exact_c(self):
        for config in candidates()[1:3]:
            a,b=self.host(dict(config,joint_strength=0.)),self.host(candidates()[0])
            try:
                za,da=a.step(self.x);sa=a.snapshot();zb,db=b.step(self.x)
                self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(da,db));self.assertTrue(equal(sa,b.snapshot()))
            finally:a.close();b.close()

    def test_snapshot_replay_observation_is_not_persistent_state(self):
        host=self.host(candidates()[2])
        try:
            host.step(self.x);saved=host.snapshot()
            za,da=host.step(self.x.roll(1,-1));end=host.snapshot()
            self.assertEqual(host.observation,{})
            host.restore(saved);zb,db=host.step(self.x.roll(1,-1))
            self.assertTrue(torch.equal(za,zb));self.assertTrue(equal(da,db));self.assertTrue(equal(end,host.snapshot()))
        finally:host.close()

    def test_gates_remain_separate_and_incomplete_is_not_a_loss(self):
        good=dict(status='COMPLETE',delta_pp=.2,order_delta_pp=[.2,.2],positive_trajectories=6,imageweighted_delta_pp=.2,worst_seed_averaged_cell_pp=-1,CI_low_pp=.1)
        pairs=[dict(good,candidate=run.PRIMARY,baseline=b,role='SEARCH') for b in ('C','C_JS_SHARED','CW_LSO')]
        pairs[-1]['delta_pp']=-1
        d=run.decision(pairs,True)
        self.assertTrue(d['standalone_signal']);self.assertTrue(d['improvement_over_shared']);self.assertFalse(d['improvement_over_strong_control'])
        self.assertFalse(run.decision(pairs,False)['standalone_signal'])
        pairs[0]['CI_low_pp']=0
        self.assertFalse(run.decision(pairs,True)['standalone_signal'])

    def test_paired_cells_preserve_positive_negative_and_seed_mean(self):
        domains=[]
        for seed in (1,2,3):
            for condition,value in [('C',70.),('C_JS_SHARED',72.),('C_JS_ACQ_WEIGHTED',71.),('CW_LSO',73.)]:
                domains.append(dict(condition=condition,seed=seed,order=0,role='SEARCH',domain='A',channel='OC',Dice_percent=value))
        rows=run.paired_cells(domains)
        self.assertTrue(any(r['delta_pp']>0 for r in rows));self.assertTrue(any(r['delta_pp']<0 for r in rows))
        self.assertEqual(len(rows),20)
        self.assertEqual([r['delta_pp'] for r in rows if r['candidate']==run.PRIMARY and r['baseline']=='C_JS_SHARED' and r['aggregation']=='seed_mean'],[-1.])

    def test_incomplete_report_does_not_invent_results_or_expose_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'public').mkdir();(root/'processes').mkdir()
            protocol=root/'protocol.md';protocol.write_text('generated CPU fixture')
            c=dict(output_root=str(root),protocol_path=str(protocol),candidates=candidates(),jobs=[],seeds=[20260907,17011,29009])
            with patch.object(run.retained.base,'ledger',return_value=dict(attempts=[],gpu_worker_seconds=0,cpu_worker_seconds=0)),patch.object(run.retained.scoring,'load_rows',return_value=[]):
                run.report(c,dict(status='INCOMPLETE',jobs={},reason='/data_nas/private/unpublished'))
            d=json.loads((root/'public/FINAL_DECISION.json').read_text())
            self.assertFalse(d['standalone_signal']);self.assertIn('NA',(root/'public/REPORT.md').read_text())
            self.assertNotIn('/data_nas/',(root/'public/REPORT.md').read_text())


if __name__=='__main__':unittest.main()
