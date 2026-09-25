import json
import unittest
from collections import Counter
from pathlib import Path

from dpa_ctta.r8_ba.target_jobs import resolve


class TestTargetJobs(unittest.TestCase):
    def test_all_724_dependencies_without_scope_changes(self):
        root = Path(__file__).resolve().parents[2]
        graph = json.loads((root / "docs/review/r8/TASK_GRAPH.static.json").read_text())
        configs = json.loads((root / "docs/review/r8/input/R8_EXPERIMENT_SPEC.json").read_text())["configs"]
        selected = {route: next(row["id"] for row in configs if row["route"] == route) for route in "AB"}
        deps = [resolve(job, graph, configs, selected) for job in graph["jobs"] if job["arrivals"]]
        self.assertEqual(len(deps), 724)
        self.assertEqual(sum(row["source_job"] is None for row in deps), 129)
        self.assertEqual(sum(row["ablation"] is not None for row in deps), 125)
        gradients = Counter(row["gradient"] for row in deps if row["gradient"])
        self.assertEqual(gradients, dict(B_G1=35, B_G3=35, COLD_G3=25))
        self.assertEqual(len({row["source_job"] for row in deps if row["source_job"]}), 65)


if __name__ == "__main__":
    unittest.main()
