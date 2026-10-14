"""Create a private CTTA data package from the frozen SSL_CL HDF5 release.

No resplitting, resizing, model execution, or checksum recomputation. Only
SUMMARY.json is public; all manifests and HDF5 payloads remain private.
"""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
import shutil

import h5py
import numpy as np


DATASETS = {
    "fundus": {"hw": 384, "classes": [0, 1, 2], "labels": ["background", "disc_rim", "cup"],
               "input": "RGB/255; OD=(label>0), OC=(label==2) for two sigmoid heads"},
    "prostate": {"hw": 256, "classes": [0, 1], "labels": ["background", "whole_prostate"],
                 "input": "single channel; preserve frozen MRI normalization"},
    "mnms": {"hw": 384, "classes": [0, 1, 2, 3], "labels": ["background", "LV", "MYO", "RV"],
             "input": "single channel; preserve frozen MRI normalization; ED/ES grouped by patient"},
}
ONLINE = ["case_id", "patient_id", "dataset", "split_seed", "split", "primary_20pct_split",
          "site_or_vendor", "phase", "cohort", "evaluation_eligible", "image_h5_relpath"]
SCORER = ONLINE + ["label_h5_relpath"]


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in rows)


def payload(root, relative, kind):
    p = Path(relative)
    if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != kind:
        raise ValueError("unsafe payload path")
    result = root / "h5/v1" / p
    if not result.resolve().is_relative_to((root / "h5/v1" / kind).resolve()):
        raise ValueError("payload escapes release")
    return result


def validate_rows(rows):
    seen, roles, patient_roles = set(), defaultdict(set), defaultdict(set)
    for row in rows:
        key = (row["dataset"], row["case_id"])
        if key in seen or row["dataset"] not in DATASETS:
            raise ValueError("duplicate case or unknown dataset")
        seen.add(key)
        roles[(row["dataset"], row["patient_id"])].add(row["split"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError("unknown split")
        role = row["primary_20pct_split"]
        patient_roles[(row["dataset"], row["patient_id"])].add(role)
        if role not in {"train_labeled", "train_unlabeled", "val", "test"}:
            raise ValueError("unknown role")
        if ("train" if role.startswith("train_") else role) != row["split"]:
            raise ValueError("role/split mismatch")
        if row["evaluation_eligible"] not in {"true", "false"}:
            raise ValueError("unknown evaluation eligibility")
        if row["evaluation_eligible"] == "false" and role != "train_unlabeled":
            raise ValueError("auxiliary data leaked into evaluation/labeled training")
    if any(len(value) != 1 for value in roles.values()) or any(len(value) != 1 for value in patient_roles.values()):
        raise ValueError("patient overlap across splits or labeled/unlabeled roles")


def prepare(source, target):
    source, target = Path(source).resolve(), Path(target).resolve()
    if target.exists() or source == target or source in target.parents:
        raise ValueError("target must be new and outside the frozen source")
    seeds = {s: read_csv(source / f"manifests/diagnostics/lcrseg_v1_seed{s}.csv") for s in range(3)}
    for seed, rows in seeds.items():
        validate_rows(rows)
        if any(r["split_seed"] != str(seed) for r in rows):
            raise ValueError("wrong split seed")
        train = read_csv(source / f"manifests/training/lcrseg_v1_seed{seed}.csv")
        if {(r["dataset"], r["case_id"]) for r in train} != {(r["dataset"], r["case_id"]) for r in rows}:
            raise ValueError("training/diagnostic coverage mismatch")
        if any(r.get("label_h5_relpath") or r.get("label_sha256") for r in train if r["primary_20pct_split"] == "train_unlabeled"):
            raise ValueError("original training manifest exposes hidden labels")
    identity = lambda rows: {(r["dataset"], r["case_id"], r["image_h5_relpath"],
                              r["label_h5_relpath"], r["patient_id"], r["phase"], r["cohort"],
                              r["evaluation_eligible"], r["site_or_vendor"]) for r in rows}
    if any(identity(rows) != identity(seeds[0]) for rows in seeds.values()):
        raise ValueError("seed manifests disagree on payload/cohort")
    target.mkdir(parents=True)
    (target / "PREPARING").write_text("Incomplete until READY.json exists.\n")
    # Preserve the original release and hidden-label policy; never edit its files.
    for name in ["h5/v1", "manifests/training", "manifests/diagnostics", "splits"]:
        shutil.copytree(source / name, target / name)
    summary = {"schema_version": 1, "status": "DATA_READY_NO_EXPERIMENT", "datasets": {},
               "seeds": {}, "checks": {"patient_split_disjoint": True, "patient_labeled_unlabeled_disjoint": True,
                                         "seed_payload_identity": True},
               "verification": "All paired HDF5 metadata; one decoded middle slice/image per domain and phase. No new full-file hashes."}
    samples, sizes, slices = set(), 0, Counter()
    for row in seeds[0]:
        ds = row["dataset"]
        ip, lp = payload(target, row["image_h5_relpath"], "images"), payload(target, row["label_h5_relpath"], "labels")
        sizes += ip.stat().st_size + lp.stat().st_size
        with h5py.File(ip) as ih, h5py.File(lp) as lh:
            image, label = ih["image"], lh["label"]
            hw = DATASETS[ds]["hw"]
            expected = image.shape[1:] if ds == "fundus" else image.shape
            if image.ndim != 3 or label.shape != expected or image.shape[-2:] != (hw, hw):
                raise ValueError("image/label layout mismatch")
            if ds == "fundus" and image.shape[0] != 3:
                raise ValueError("fundus must be RGB")
            slices[ds] += 1 if ds == "fundus" else image.shape[0]
            if ds != "fundus":
                raw_spacing = ih.attrs["processed_spacing"]
                spacing = np.asarray(json.loads(raw_spacing) if isinstance(raw_spacing, str) else raw_spacing, dtype=float)
                if spacing.shape != (3,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
                    raise ValueError("missing physical volume spacing")
            for h in [ih, lh]:
                if str(h.attrs["patient_id"]) != row["patient_id"] or str(h.attrs["case_id"]) != row["case_id"]:
                    raise ValueError("HDF5 identity mismatch")
            group = (ds, row["site_or_vendor"], row["phase"])
            if group not in samples:
                x = image[:] if ds == "fundus" else image[image.shape[0] // 2]
                y = label[:] if ds == "fundus" else label[label.shape[0] // 2]
                if not np.isfinite(x).all() or not set(np.unique(y)).issubset(DATASETS[ds]["classes"]):
                    raise ValueError("decoded sample invalid")
                samples.add(group)
    for ds, spec in DATASETS.items():
        rows = [r for r in seeds[0] if r["dataset"] == ds]
        summary["datasets"][ds] = {**spec, "patients": len({r["patient_id"] for r in rows}),
            "image_label_pairs": len(rows), "domains": dict(Counter(r["site_or_vendor"] for r in rows)),
            "evaluation_domains": dict(Counter(r["site_or_vendor"] for r in rows if r["evaluation_eligible"] == "true")),
            "images_or_slices": slices[ds],
            "evaluation_eligible_pairs": sum(r["evaluation_eligible"] == "true" for r in rows)}
    for seed, rows in seeds.items():
        for ds in DATASETS:
            for role in ["train_labeled", "train_unlabeled", "val", "test"]:
                subset = [r for r in rows if r["dataset"] == ds and r["primary_20pct_split"] == role]
                folder = target / "views" / f"seed{seed}" / ds / role
                write_csv(folder / "online.csv", subset, ONLINE)
                # Auxiliary25 never enters evaluation or labeled training. The
                # scorer view only includes held-out eligible val/test cases.
                if role in {"val", "test"}:
                    write_csv(folder / "scorer.csv", [r for r in subset if r["evaluation_eligible"] == "true"], SCORER)
        summary["seeds"][str(seed)] = {ds: dict(Counter(r["primary_20pct_split"] for r in rows if r["dataset"] == ds)) for ds in DATASETS}
    for f in (target / "views").rglob("online.csv"):
        with f.open(newline="") as h:
            if set(csv.DictReader(h).fieldnames) != set(ONLINE):
                raise ValueError("online manifest schema violation")
    summary.update({"payload_files": 2 * len(seeds[0]), "payload_bytes": sizes,
                    "decoded_domain_phase_samples": len(samples), "online_views": 36, "scorer_views": 18})
    (target / "SUMMARY.json").write_text(json.dumps(summary, indent=2) + "\n")
    (target / "READY.json").write_text(json.dumps({"status": "READY", "summary": "SUMMARY.json",
                                                  "experiment_started": False}) + "\n")
    (target / "PREPARING").unlink()
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(os.environ["PREP_SOURCE"], os.environ["PREP_TARGET"]), indent=2))
