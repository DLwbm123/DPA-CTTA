import unittest
from pathlib import Path

from dpa_ctta.r8_ba.paths import SOURCE_ROOT, TARGET_ROOT, owned_source_path, owned_target_path


class TestPaths(unittest.TestCase):
    def test_dedicated_source_output_boundary(self):
        self.assertEqual(owned_source_path(SOURCE_ROOT / "jobs/job1"),
                         (SOURCE_ROOT / "jobs/job1").resolve())
        for path in (SOURCE_ROOT, SOURCE_ROOT / "../private/registry.json",
                     Path("/data_nas/jiangsuiyang/CTTA/jobs/r7")):
            with self.assertRaisesRegex(ValueError, "outside dedicated"):
                owned_source_path(path)

    def test_target_output_cannot_escape_to_source_or_r7(self):
        self.assertEqual(owned_target_path(TARGET_ROOT / "jobs/job1"),
                         (TARGET_ROOT / "jobs/job1").resolve())
        for path in (TARGET_ROOT, TARGET_ROOT / "../source/jobs/job1",
                     Path("/data_nas/jiangsuiyang/CTTA/jobs/r7")):
            with self.assertRaisesRegex(ValueError, "outside dedicated"):
                owned_target_path(path)


if __name__ == "__main__":
    unittest.main()
