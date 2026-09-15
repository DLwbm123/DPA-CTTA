import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class PlanTests(unittest.TestCase):
    def test_inherited_ten_arms_and_algorithm_blocks(self):
        old=json.loads((ROOT/'inherited/R4D_SCIENCE_PROPOSAL.json').read_text())
        new=json.loads((ROOT/'R4T_SCIENCE_PROPOSAL.json').read_text())
        self.assertEqual(old['arms'],new['arms'][:10])
        for key in ('teacher','student','RP','kernel_geometry','resources','direction_A_preservation'):
            self.assertEqual(old[key],new[key])
    def test_budget_and_one_to_three_device_totals(self):
        c=json.loads((ROOT/'R4T_SCIENCE_PROPOSAL.json').read_text())
        jobs=json.loads((ROOT/'PLAN_MATRIX.reference.json').read_text())['jobs']
        self.assertEqual(len(jobs),70)
        self.assertEqual(len({j['job_id'] for j in jobs}),70)
        self.assertEqual(sum(j['records'] for j in jobs),136570)
        self.assertEqual(sum(j['network_forwards'] for j in jobs),1112070)
        self.assertEqual(sum(j['records'] for j in jobs if j['order']<4),109256)
        self.assertEqual(sum(a['forwards_per_visit'] for a in c['arms']),114)
        self.assertEqual(c['per_actual_gpu_smoke']['network_forwards'],260)
        self.assertEqual(c['per_actual_gpu_smoke']['adam_calls'],32)
        self.assertEqual(c['formal_budget']['network_forwards']+2*260,1112590)
        self.assertEqual(c['formal_budget']['adam_calls']+2*32,136634)
        self.assertFalse(c['execution']['enabled'])
if __name__=='__main__':unittest.main(verbosity=2)
