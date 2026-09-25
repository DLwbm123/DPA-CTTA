import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256


class TestSourceWorkerMarker(unittest.TestCase):
    def test_scaler_marker_only_after_source_recheck(self):
        path = Path(__file__).resolve().parents[2] / "scripts/r8/run_scaler.py"
        spec = importlib.util.spec_from_file_location("r8_run_scaler_test", path)
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)

        @contextmanager
        def source(*_args):
            yield None, None, {"source_reads": 1}
            if fail[0]:
                raise ValueError("source changed on exit")

        def run(root, *_args):
            root.mkdir()
            (root / "scaler_complete.json").write_text("{}")
            return {"sealed": True}

        with tempfile.TemporaryDirectory() as directory:
            fail = [True]
            for index in (0, 1):
                root = Path(directory) / f"job{index}"
                config = dict(schema="R8_SCALER_WORK_V1", physical_gpu=5,
                              code_sha="a" * 40, protocol_sha256=PROTOCOL_SHA256,
                              maximum_seconds=120, job_root=str(root), refs={},
                              source_root="unused", checkpoint_path="unused")
                cfg = Path(directory) / f"config{index}.json"
                cfg.write_text(json.dumps(config))
                bound = {"docs": {"manifest": {"checkpoint": {"sha256": "b" * 64}}}}
                with patch.dict(os.environ, {"R8_WORK_CONFIG": str(cfg),
                                             "CUDA_VISIBLE_DEVICES": "5"}), \
                        patch.object(worker, "bind_metadata", return_value=bound), \
                        patch.object(worker, "open_source", source), \
                        patch.object(worker, "owned_source_path", lambda value: Path(value)), \
                        patch.object(worker, "run_scaler", run):
                    if fail[0]:
                        with self.assertRaisesRegex(ValueError, "source changed"):
                            worker.main()
                        self.assertFalse((root / "worker_complete.json").exists())
                    else:
                        worker.main()
                        marker = json.loads((root / "worker_complete.json").read_text())
                        self.assertEqual(marker["scaler_receipt_sha256"],
                                         hashlib.sha256(b"{}").hexdigest())
                        self.assertEqual(marker["source_io_counts"], {"source_reads": 1})
                fail[0] = False


if __name__ == "__main__":
    unittest.main()
