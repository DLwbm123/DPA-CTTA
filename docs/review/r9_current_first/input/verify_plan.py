#!/usr/bin/env python3
"""Validate the R9 planning manifest only. No torch, model, dataset, or network access."""
from __future__ import annotations
import collections
import json
import math
from pathlib import Path

def main() -> None:
    root = Path(__file__).resolve().parent
    data = json.loads((root / "R9_SPEC_AND_MATRIX.json").read_text(encoding="utf-8"))
    summary = json.loads((root / "SCREEN24_VERIFIED_SUMMARY.json").read_text(encoding="utf-8"))
    assert data["execution_authorized"] is False
    assert data["proposed_implementation_sha"] is None
    assert data["gpu_assignments"] is None
    assert data["output_root"] is None
    source = data["source_tasks"]
    core = data["target_core_slots"]
    endpoint = data["target_final16k_slots_max"]
    counts = data["counts"]
    assert len(source) == 43
    assert len(core) == 610
    assert len(endpoint) == 161
    for sequence in (source, core, endpoint):
        assert len({x["id"] for x in sequence}) == len(sequence)
    assert not ({x["id"] for x in core} & {x["id"] for x in endpoint})
    fit_steps = sum(x["end_step"] - x["start_step"] for x in source)
    assert fit_steps == counts["additional_fit_updates"] == 648000
    assert sum(x["start_step"] == 4000 for x in source) == 10
    assert all(x["end_step"] == 16000 for x in source)
    assert all(x["selection_locked_before_target"] for x in core + endpoint)
    by_phase = dict(collections.Counter(x["phase"] for x in core))
    assert by_phase == counts["core_by_phase"]
    for name, seq in (("core", core), ("max", core + endpoint)):
        assert sum(x["arrivals"] for x in seq) == counts[f"arrivals_{name}"]
        assert sum(x["principal_scored_visits"] for x in seq) == counts[f"principal_score_visits_{name}"]
    assert len(core) + len(endpoint) == counts["target_total_slots_max"] == 771
    assert sum(x["arrivals"] == 19510 for x in core) == 42
    assert all((x["arrivals"], x["principal_scored_visits"]) in ((1951,1695),(19510,16950))
               for x in core + endpoint)
    rows = {x["method"]: x for x in summary["main"]}
    assert math.isclose(rows["A_FULL"]["Dice_percent"], 74.00853122424891, abs_tol=1e-10)
    assert math.isclose(rows["B_FULL"]["Dice_percent"], 73.24912119185295, abs_tol=1e-10)
    for key in ("A","B","C0"):
        field = key if key == "C0" else key + "_FULL"
        mean = sum(x[key] for x in summary["domain_comparison"]) / 4
        assert math.isclose(mean, rows[field]["Dice_percent"], abs_tol=1e-9)
    result = {
        "status": "PLAN_ARITHMETIC_AND_SCHEMA_CHECKS_PASSED",
        "scope": "Planning files only; not an implementation, source-training, GPU or scientific validation.",
        "source_tasks": len(source),
        "additional_fit_updates": fit_steps,
        "core_target_slots": len(core),
        "max_total_target_slots": len(core) + len(endpoint),
        "max_arrivals": counts["arrivals_max"],
        "max_principal_score_visits": counts["principal_score_visits_max"],
        "actual_experiment_launched": False,
    }
    (root / "PLAN_VALIDATION.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                                               encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
