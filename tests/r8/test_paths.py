import unittest
from pathlib import Path

from dpa_ctta.r8_ba.paths import SOURCE_ROOT, owned_source_path


class TestPaths(unittest.TestCase):
    def test_dedicated_source_output_boundary(self):
        self.assertEqual(owned_source_path(SOURCE_ROOT / "jobs/job1"),
                         (SOURCE_ROOT / "jobs/job1").resolve())
        for path in (SOURCE_ROOT, SOURCE_ROOT / "../private/registry.json",
                     Path("/data_nas/jiangsuiyang/CTTA/jobs/r7")):
            with self.assertRaisesRegex(ValueError, "outside dedicated"):
                owned_source_path(path)


if __name__ == "__main__":
    unittest.main()
