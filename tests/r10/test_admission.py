import copy,unittest
from dpa_ctta.r10_use_write_rl.protocol import CAPS,graph
from dpa_ctta.r10_use_write_rl.profile import units,node_units,projection


class Admission(unittest.TestCase):
    def test_complete_units_and_finite_three_attempt_reserve(self):
        rows={k:dict.fromkeys(CAPS,0) for k in units()}
        for r in rows.values():r.update(measured=True,evidence='synthetic-only')
        budgets={n['id']:dict.fromkeys(CAPS,0) for n in graph()}
        for n in graph():
            if n['kind']=='online':budgets[n['id']]['disk_bytes']=n['job']['arrivals']*2*512*512*4
        for index,(name,b) in enumerate(budgets.items()):b['gpu_seconds']=index
        p=dict(measurements=rows,node_budgets=budgets,prior_cost=dict.fromkeys(CAPS,0))
        proof=projection(p);expected=sum(sorted((b['gpu_seconds'] for b in budgets.values()),reverse=True)[:3])
        self.assertEqual(proof['recovery_reserve']['gpu_seconds'],expected)
        self.assertEqual(proof['upper_bound']['gpu_seconds'],sum(b['gpu_seconds'] for b in budgets.values())+expected)
        bad=copy.deepcopy(p);bad['measurements'].pop(next(iter(rows)))
        with self.assertRaises(ValueError):projection(bad)
        bad=copy.deepcopy(p);bad['prior_cost']['gpu_seconds']=CAPS['gpu_seconds']
        self.assertEqual(projection(bad)['status'],'OVER_CAP')
