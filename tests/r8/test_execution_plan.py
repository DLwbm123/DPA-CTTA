import json
import sys
import unittest
from collections import Counter
from pathlib import Path
from dpa_ctta.r8_ba.execution_plan import tasks
from dpa_ctta.r8_ba.resources import json_size_bound, units

ROOT = Path(__file__).resolve().parents[2]


class ExecutionPlanTests(unittest.TestCase):
    def test_all_jobs_reachable_and_every_cost_charged_once(self):
        graph = json.loads((ROOT / 'docs/review/r8/TASK_GRAPH.static.json').read_text())
        work, cpu = tasks(graph)
        pending = {r['id']: r['dependencies'] for r in work}
        self.assertEqual(len(pending), len(work))
        pending.update(cpu)
        complete = set()
        while pending:
            ready = {key for key,deps in pending.items() if set(deps) <= complete}
            self.assertTrue(ready, 'cycle or missing prerequisite')
            complete.update(ready)
            pending = {k:v for k,v in pending.items() if k not in ready}
        total = Counter()
        for row in work:
            total.update(row['weights'])
        self.assertEqual(total, units(graph))
        self.assertEqual(sum(r['kind'] == 'SOURCE_JOB' for r in work), 65)
        self.assertEqual(sum(r['kind'] == 'TARGET_JOB' for r in work), 724)
        self.assertEqual(sum(r['kind'] == 'SCORE_JOB' for r in work), 724)

    def test_record_bound_extreme_numbers_and_escaped_registered_strings(self):
        template = dict(value=0.1, count=1, optional=None, content='中\\"\n')
        for number in (sys.float_info.max, -sys.float_info.max, 5e-324, -5e-324, 0.00010000000000000002):
            actual = dict(template, value=number, count=2_325_592, optional=number)
            self.assertLessEqual(len(json.dumps(actual).encode()), json_size_bound(template))
