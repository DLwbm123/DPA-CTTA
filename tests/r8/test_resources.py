import importlib.util
import unittest
from pathlib import Path

from dpa_ctta.r8_ba.resources import CAPS, MEASURES, check_caps, project, units


class TestResources(unittest.TestCase):
    def test_profile_required_for_full_matrix_and_caps_stop(self):
        path = Path(__file__).resolve().parents[2] / "scripts/r8/manifest.py"
        spec = importlib.util.spec_from_file_location("r8_manifest", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        graph = module.build()
        weights = units(graph)
        self.assertEqual(CAPS["backward_calls"], 4_500_000)
        self.assertEqual(CAPS["optimizer_steps"], 4_000_000)
        self.assertEqual(sum(weights[k] for k in weights if k in
                             {"vptta", "c", "g", "zero", "r7_c", "mlp", "gradient_g1",
                              "gradient_g3", "ista20", "new_a_max", "new_b_max"}), 2325592)
        row = {name: 0 for name in MEASURES}
        row.update(measured=True, sample_units=1, gpu_seconds=0.001, peak_gpu_bytes=1)
        profile = dict(schema="R8_RESOURCE_PROFILE_V1", measured_on_physical_GPU=True,
                       physical_GPU_ids=[5], code_sha="0" * 40, fixed_disk_bytes=0,
                       units={name: row.copy() for name in weights})
        self.assertEqual(project(graph, profile, 24 * 1024**3, "0" * 40)["status"], "WITHIN_PROPOSED_CAPS")
        del profile["units"]["gradient_g3"]
        with self.assertRaisesRegex(ValueError, "complete measured"):
            project(graph, profile, 24 * 1024**3, "0" * 40)
        with self.assertRaisesRegex(RuntimeError, "GLOBAL STOP"):
            check_caps(dict(CAPS, gpu_seconds=CAPS["gpu_seconds"]))


if __name__ == "__main__":
    unittest.main()
