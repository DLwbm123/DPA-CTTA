import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256


class TestSourceJobGraph(unittest.TestCase):
    def test_all_65_source_jobs_bind_to_frozen_graph(self):
        root = Path(__file__).resolve().parents[2]
        spec = importlib.util.spec_from_file_location("r8_source_job_test",
                                                   root / "scripts/r8/run_source_job.py")
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        graph = json.loads((root / "docs/review/r8/TASK_GRAPH.static.json").read_text())
        plans = json.loads((root / "docs/review/r8/input/R8_EXPERIMENT_SPEC.json").read_text())
        by_id = {row["id"]: row for row in plans["configs"]}
        chosen = {route: next(row["id"] for row in plans["configs"] if row["route"] == route)
                  for route in ("A", "B")}
        with tempfile.TemporaryDirectory() as directory:
            selection = Path(directory) / "selection.json"
            selection.write_text(json.dumps(dict(schema="R8_SOURCE_ONLY_GRID_SELECTION_V1",
                                                 protocol_sha256=PROTOCOL_SHA256,
                                                 selected_config=chosen)))
            seen = []
            for job in graph["jobs"]:
                if not job["stage"].startswith("SOURCE_"):
                    continue
                route = "B" if job["stage"] == "SOURCE_MLP" else job["arm"]
                selected_id = job["config"] if job["stage"] == "SOURCE_GRID" else chosen[route]
                request = dict(job_id=job["id"], config=by_id[selected_id],
                               mode=job["mode"], source_seed=job["source_seed"])
                if job["stage"] != "SOURCE_GRID":
                    request["selection_path"] = str(selection)
                actual = worker._job_and_config(request)[1]
                self.assertEqual(actual, job)
                seen.append(actual["id"])
            self.assertEqual((len(seen), len(set(seen))), (65, 65))


if __name__ == "__main__":
    unittest.main()
