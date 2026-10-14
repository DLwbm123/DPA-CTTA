"""Small synthetic release: split isolation, paths, labels and create-only output."""
import csv
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

SPEC = importlib.util.spec_from_file_location("prep", os.environ.get("PREP_MODULE", Path(__file__).parents[1] / "scripts/prepare_sslcl_data.py"))
prep = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prep)


class PreparationTest(unittest.TestCase):
    def test_release_and_rejections(self):
        with tempfile.TemporaryDirectory() as temp:
            source, target = Path(temp) / "source", Path(temp) / "out"
            rows = []
            for ds, spec in prep.DATASETS.items():
                hw = spec["hw"]
                row = dict.fromkeys(prep.SCORER, "")
                row.update(case_id="case", patient_id="patient", dataset=ds, split_seed="0",
                           split="test", primary_20pct_split="test", site_or_vendor="site",
                           cohort="canonical", evaluation_eligible="true",
                           image_h5_relpath=f"images/{ds}/case.h5", label_h5_relpath=f"labels/{ds}/case.h5")
                rows.append(row)
                for kind, key in [("images", "image"), ("labels", "label")]:
                    p = source / "h5/v1" / row[f"{key}_h5_relpath"]
                    p.parent.mkdir(parents=True)
                    shape = (3, hw, hw) if key == "image" and ds == "fundus" else ((hw, hw) if ds == "fundus" else (2, hw, hw))
                    with h5py.File(p, "w") as f:
                        f.create_dataset(key, data=np.zeros(shape, dtype=np.uint8))
                        f.attrs.update(patient_id="patient", case_id="case")
                        if ds != "fundus":
                            f.attrs["processed_spacing"] = "[1.0, 1.0, 1.0]"
            for seed in range(3):
                for folder in ["diagnostics", "training"]:
                    prep.write_csv(source / f"manifests/{folder}/lcrseg_v1_seed{seed}.csv", [dict(r, split_seed=str(seed)) for r in rows], prep.SCORER)
            (source / "splits").mkdir()
            result = prep.prepare(source, target)
            self.assertEqual(result["payload_files"], 6)
            self.assertTrue((target / "READY.json").exists())
            self.assertFalse((target / "PREPARING").exists())
            for p in (target / "views").rglob("online.csv"):
                self.assertNotIn("label_h5_relpath", p.read_text().splitlines()[0])
            with self.assertRaises(ValueError):
                prep.prepare(source, target)
            with self.assertRaises(ValueError):
                prep.payload(source, "../private.h5", "images")
            with self.assertRaises(ValueError):
                prep.validate_rows(rows + [dict(rows[0], case_id="other", split="val", primary_20pct_split="val")])
            with self.assertRaises(ValueError):
                prep.validate_rows([dict(rows[0], evaluation_eligible="false")])
            with self.assertRaises(ValueError):
                prep.validate_rows(rows + [rows[0]])
            with self.assertRaises(ValueError):
                prep.validate_rows([dict(rows[0], split="train", primary_20pct_split="train_labeled"),
                                    dict(rows[0], case_id="other", split="train", primary_20pct_split="train_unlabeled")])


if __name__ == "__main__":
    unittest.main()
