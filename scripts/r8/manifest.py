"""Expand the frozen R8 plan without touching models, images, or checkpoints."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "docs/review/r8/input"
SEEDS = (20260924, 20260925, 20260926, 20260927, 20260928)
ORDERS = range(5)
MODES = ("FULL", "STATIC")


def _read():
    spec = json.loads((INPUT / "R8_EXPERIMENT_SPEC.json").read_text())
    with (INPUT / "CONFIG_GRID.csv").open(newline="") as f:
        grid = list(csv.DictReader(f))
    if [r["id"] for r in grid] != [r["id"] for r in spec["configs"]]:
        raise ValueError("CSV/JSON configuration identity mismatch")
    if len(grid) != 12 or len({r["id"] for r in grid}) != 12:
        raise ValueError("R8 requires twelve distinct configurations")
    for csv_row, json_row in zip(grid, spec["configs"]):
        for key in ("route", "rank", "film_amplitude", "observer", "aux_multiplier"):
            if str(csv_row[key]) != str(json_row[key]):
                raise ValueError(f"CSV/JSON configuration mismatch: {csv_row['id']}/{key}")
    return spec, grid


def build():
    spec, grid = _read()
    if spec["execution_enabled"] is not False:
        raise ValueError("input plan is not a disabled design")
    jobs = []

    def add(stage, arm, seed=None, order=None, visits=0, config=None, mode=None):
        ident = (f"SOURCE_{config}_{mode}_{seed}" if stage == "SOURCE_GRID" else
                 "_".join(str(v) for v in (stage, arm, mode, config, seed, order) if v is not None))
        source = stage.startswith("SOURCE")
        jobs.append(dict(id=ident, stage=stage, arm=arm, mode=mode, config=config,
                         source_seed=seed if source or (seed is not None and arm not in
                                     spec["target"]["baseline_stochastic"]) else None,
                         target_seed=None if source or seed is None else seed - 20260924 + 20260907,
                         order=order, arrivals=visits,
                         scored_contents=0 if source else 1695 * (10 if stage == "STRESS_LONG10" else 1),
                         fit_steps=16000 if source else 0, status="NOT_RUN"))

    for row in grid:
        for mode in MODES:
            for seed in SEEDS[:2]:
                add("SOURCE_GRID", row["route"], seed, config=row["id"], mode=mode)
    for route in "BA":
        for mode in MODES:
            for seed in SEEDS[2:]:
                add("SOURCE_FINAL", route, seed, config="SOURCE_SELECTED_" + route, mode=mode)
    for seed in SEEDS:
        add("SOURCE_MLP", "CURRENT_MLP", seed, config="SOURCE_SELECTED_B")

    for row in grid:
        for mode in MODES:
            for seed in SEEDS[:2]:
                for order in ORDERS:
                    add("TARGET_GRID", row["route"], seed, order, 1951, row["id"], mode)
    for route in "BA":
        for mode in MODES:
            for seed in SEEDS[2:]:
                for order in ORDERS:
                    add("TARGET_FINAL", route, seed, order, 1951, "SOURCE_SELECTED_" + route, mode)
    for seed in SEEDS:
        for order in ORDERS:
            add("TARGET_MLP", "CURRENT_MLP", seed, order, 1951, "SOURCE_SELECTED_B")
    for arm in spec["target"]["baseline_stochastic"]:
        for seed in SEEDS:
            for order in ORDERS:
                add("TARGET_BASELINE", arm, seed, order, 1951)
    for arm in (*spec["target"]["baseline_deterministic"], *spec["target"]["frozen_controls"]):
        for order in ORDERS:
            add("TARGET_CONTROL", arm, order=order, visits=1951)
    for arm in spec["B_gradient_diagnostics"]["arms"]:
        for seed in SEEDS:
            for order in ORDERS:
                add("TARGET_GRADIENT", arm + "_GRADIENT_ENABLED_NOT_ZERO_BACKWARD", seed, order, 1951,
                    "SOURCE_SELECTED_B", "FULL")
    for route in "BA":
        for ablation in spec["same_weight_ablations"][route]:
            for seed in SEEDS:
                for order in ORDERS:
                    add("TARGET_ABLATION", route + "_" + ablation, seed, order, 1951,
                        "SOURCE_SELECTED_" + route, "FULL")
    for stream, visits in (("MIXED", 1951), ("LONG10", 19510)):
        for arm in spec["stress"]["stochastic_or_seed_bound_arms"]:
            for seed in SEEDS:
                add("STRESS_" + stream, arm, seed, visits=visits)
        for arm in spec["stress"]["deterministic_arms"]:
            add("STRESS_" + stream, arm, visits=visits)

    ids = [j["id"] for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate task IDs")
    counts = spec["counts"]
    stages = {s: sum(j["stage"] == s for j in jobs) for s in {j["stage"] for j in jobs}}
    expected = dict(SOURCE_GRID=counts["source_grid_training_jobs"],
                    SOURCE_FINAL=counts["source_finalist_additional_training_jobs"],
                    SOURCE_MLP=counts["source_current_mlp_training_jobs"],
                    TARGET_GRID=counts["target_grid_jobs"], TARGET_FINAL=counts["target_finalist_additional_jobs"],
                    TARGET_MLP=counts["target_current_mlp_jobs"],
                    TARGET_BASELINE=counts["target_baseline_stochastic_jobs"],
                    TARGET_CONTROL=counts["target_baseline_deterministic_jobs"] + counts["target_frozen_R7_C_jobs"],
                    TARGET_GRADIENT=counts["target_B_gradient_diagnostic_jobs"],
                    TARGET_ABLATION=counts["target_same_weight_ablations_jobs"],
                    STRESS_MIXED=52, STRESS_LONG10=52)
    if stages != expected or sum(j["arrivals"] for j in jobs) != counts["target_arrivals_total"]:
        raise ValueError("task graph disagrees with frozen count specification")
    if {j["id"] for j in jobs if j["stage"] == "SOURCE_GRID"} != {
            j["id"] for j in spec["known_initial_source_jobs"]}:
        raise ValueError("source grid disagrees with frozen source IDs")
    return dict(schema="R8_STATIC_TASK_GRAPH_V1", execution_enabled=False,
                spec_sha256=hashlib.sha256((INPUT / "R8_EXPERIMENT_SPEC.json").read_bytes()).hexdigest(),
                grid_sha256=hashlib.sha256((INPUT / "CONFIG_GRID.csv").read_bytes()).hexdigest(),
                protocol_sha256=hashlib.sha256((ROOT / "src/dpa_ctta/r8_ba/protocol.json").read_bytes()).hexdigest(),
                phase_order=["SOURCE_PROFILE_AND_BIND", "SOURCE_ORACLES_AND_BASES", "SOURCE_GRID",
                             "SOURCE_ONLY_CONFIG_AND_SNAPSHOT_SELECTION", "SOURCE_FINAL", "SOURCE_MLP",
                             "SOURCE_CAL_GRADIENT_LR_SELECTION", "LOCK_ALL_ARTIFACTS", "TARGET_GRID_AND_BASELINES_AND_ABLATIONS",
                             "TARGET_FINAL_AND_MLP_AND_GRADIENT", "STRESS_MIXED", "STRESS_LONG10", "READ_ONLY_AGGREGATION"],
                selection_placeholders="SOURCE_SELECTED_A/B remain unresolved until source-only selection",
                stages=stages, source_training_jobs=65, target_jobs=724,
                target_arrivals=counts["target_arrivals_total"], jobs=jobs)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
