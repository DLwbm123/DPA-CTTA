import os
import unittest
from unittest.mock import patch

from dpa_ctta.r8_ba.inputs import open_source


class TestInputs(unittest.TestCase):
    def test_gpu_zero_is_rejected_before_any_source_or_checkpoint_read(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0",
                                  "CUBLAS_WORKSPACE_CONFIG": ":4096:8"}):
            with self.assertRaisesRegex(ValueError, "physical GPU 5/6/7"):
                with open_source({"schema": "R8_BOUND_SOURCE_METADATA_V1"}, "/missing-source",
                                 "/missing-checkpoint", 0.1, 0, 1024, 1024, lambda: None):
                    self.fail("unapproved GPU or source input opened")


if __name__ == "__main__":
    unittest.main()
