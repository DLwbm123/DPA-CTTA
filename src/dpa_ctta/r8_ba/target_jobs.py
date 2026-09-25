"""Frozen target-to-source dependencies, including selected configs and stress arms."""
from .gradient import STEPS

NO_NEW_SOURCE = {"N_SOURCE_EVAL", "C0_CURRENT_STATS", "VPTTA_NATIVE",
                 "C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER", "R7_C_FULL", "R7_C_STATIC"}


def resolve(job, graph, configs, selected):
    if job not in graph["jobs"] or job["arrivals"] not in (1951, 19510):
        raise ValueError("R8 registered target job required")
    arm = job["arm"].removesuffix("_GRADIENT_ENABLED_NOT_ZERO_BACKWARD")
    if arm in NO_NEW_SOURCE:
        return dict(arm=arm, source_job=None, config=None, mode=None, ablation=None, gradient=None)
    gradient = arm if arm in STEPS else None
    mlp = arm == "CURRENT_MLP"
    route = "B" if mlp or gradient else arm[0]
    if route not in ("A", "B"):
        raise ValueError("R8 target family")
    mode = job["mode"] or ("STATIC" if arm.endswith("_STATIC") else "FULL")
    candidate_id = job["config"]
    if candidate_id is None or candidate_id.startswith("SOURCE_SELECTED_"):
        candidate_id = selected[route]
    candidates = [row for row in configs if row["id"] == candidate_id and row["route"] == route]
    if len(candidates) != 1:
        raise ValueError("R8 selected target config")
    matches = []
    for source in graph["jobs"]:
        if source["arrivals"] or source["source_seed"] != job["source_seed"]:
            continue
        if mlp:
            if source["stage"] == "SOURCE_MLP":
                matches.append(source)
        elif (source["arm"] == route and source["mode"] == mode and
              (source["config"] == candidate_id or
               (source["stage"] == "SOURCE_FINAL" and selected[route] == candidate_id))):
            matches.append(source)
    if len(matches) != 1:
        raise ValueError("R8 target requires exactly one frozen source dependency")
    ablation = arm[2:] if job["stage"] == "TARGET_ABLATION" else None
    return dict(arm=arm, source_job=matches[0]["id"], config=candidates[0], mode=None if mlp else mode,
                ablation=ablation, gradient=gradient)
