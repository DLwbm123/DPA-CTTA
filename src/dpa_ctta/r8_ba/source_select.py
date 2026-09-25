"""Fixed two-seed source-only selection across all 48 discovery jobs."""
import math

from .calibration import select_checkpoint, select_family
from .protocol import PROTOCOL_SHA256
from .trainer import SAVE_STEPS


SEEDS = (20260924, 20260925)
MODES = ("FULL", "STATIC")


def select_grid(configs, rows_by_job, cost_by_config):
    """rows_by_job[(config_id, mode, seed)][step] contains 64 source-val rows."""
    if (len(configs) != 12 or len({row["id"] for row in configs}) != 12 or
            sum(row["route"] == "A" for row in configs) != 4 or
            sum(row["route"] == "B" for row in configs) != 8):
        raise ValueError("R8 complete twelve-config grid required")
    ids = {row["id"] for row in configs}
    expected = {(name, mode, seed) for name in ids for mode in MODES for seed in SEEDS}
    if set(rows_by_job) != expected or set(cost_by_config) != ids:
        raise ValueError("R8 all 48 source discovery jobs and costs required")
    if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
           for value in cost_by_config.values()):
        raise ValueError("R8 invalid source plus estimated online cost")
    points = {}
    for config in configs:
        name = config["id"]
        points[name] = {}
        for mode in MODES:
            pair = [rows_by_job[name, mode, seed] for seed in SEEDS]
            if any(set(records) != set(SAVE_STEPS) for records in pair):
                raise ValueError("R8 all five calibrated source points required")
            step, scores = select_checkpoint({s: (pair[0][s], pair[1][s]) for s in SAVE_STEPS})
            points[name][mode] = dict(source_step=step, score=scores[step],
                                      all_step_scores=scores)
    selected = {}
    for route in ("B", "A"):
        family_ids = [row["id"] for row in configs if row["route"] == route]
        selected[route] = select_family(
            {name: points[name]["FULL"]["score"] for name in family_ids},
            {name: cost_by_config[name] for name in family_ids})
    return dict(schema="R8_SOURCE_ONLY_GRID_SELECTION_V1", protocol_sha256=PROTOCOL_SHA256,
                discovery_seeds=list(SEEDS), selected_config=selected,
                per_config_mode=points, recorded_source_plus_online_cost=dict(cost_by_config))
