import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dpa_ctta.r8_ba.source_artifacts import digest, read_completed
from dpa_ctta.r8_ba.trainer import SAVE_STEPS


class TestSourceArtifacts(unittest.TestCase):
    def test_selected_archive_is_bound_to_completed_worker(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            job = dict(stage="SOURCE_MLP", source_seed=20260924)
            candidate = dict(id="synthetic")
            payload = dict(job=job, config=candidate, code_sha="1" * 40, graph_spec_sha256="2" * 64)
            binding = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
            (root / "physical.jsonl").write_text("synthetic\n")
            fit = dict(binding=binding, source_seed=job["source_seed"], steps=16000,
                       physical_sha256=digest(root / "physical.jsonl"), selected_sha256={})
            validations = {}
            for step in SAVE_STEPS:
                selected = root / f"selected.{step}.pt"
                selected.write_bytes(str(step).encode())
                fit["selected_sha256"][str(step)] = digest(selected)
                directory = root / f"validation.{step}"
                directory.mkdir()
                (directory / "val_complete.json").write_text("{}")
                validations[str(step)] = digest(directory / "val_complete.json")
            (root / "fit_complete.json").write_text(json.dumps(fit))
            marker = dict(schema="R8_SOURCE_JOB_WORK_COMPLETE_V1", binding=binding,
                          binding_payload=payload, fit_receipt_sha256=digest(root / "fit_complete.json"),
                          validation_receipt_sha256=validations)
            (root / "worker_complete.json").write_text(json.dumps(marker))
            with patch("dpa_ctta.r8_ba.source_artifacts.load_validation", return_value=["validated"]) as loader:
                result = read_completed(root, job, candidate, "1" * 40, "2" * 64)
                self.assertEqual(set(result["artifacts"]), set(SAVE_STEPS))
                self.assertEqual(loader.call_count, 5)
                (root / "selected.1000.pt").write_bytes(b"altered")
                with self.assertRaisesRegex(ValueError, "archive changed"):
                    read_completed(root, job, candidate, "1" * 40, "2" * 64)


if __name__ == "__main__":
    unittest.main()
