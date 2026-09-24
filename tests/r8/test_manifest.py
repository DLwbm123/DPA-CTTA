import importlib.util
import unittest
from pathlib import Path


class TestManifest(unittest.TestCase):
    def test_stress_seed_binding_and_frozen_counts(self):
        path = Path(__file__).resolve().parents[2] / "scripts/r8/manifest.py"
        spec = importlib.util.spec_from_file_location("r8_manifest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        graph = module.build()
        self.assertEqual((graph["source_training_jobs"], graph["target_jobs"],
                          graph["target_arrivals"]), (65, 724, 2325592))
        stress = [job for job in graph["jobs"] if job["stage"].startswith("STRESS_")]
        self.assertEqual(len(stress), 104)
        for job in stress:
            if job["arm"] in ("VPTTA_NATIVE", "C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER"):
                self.assertIsNone(job["source_seed"])
            elif job["target_seed"] is not None:
                self.assertEqual(job["source_seed"] - 20260924,
                                 job["target_seed"] - 20260907)


if __name__ == "__main__":
    unittest.main()
