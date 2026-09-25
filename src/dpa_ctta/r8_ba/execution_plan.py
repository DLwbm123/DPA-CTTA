"""The complete frozen work queue and its additive resource units."""
from collections import Counter

from .resources import category, units


def tasks(graph):
    from .scope import SCREEN
    if SCREEN:
        from .screen import tasks as screen_tasks
        return screen_tasks(graph)
    rows = []
    def add(key, kind, weights, dependencies=(), job=None, amplitude=None):
        rows.append(dict(id=key, kind=kind, weights=dict(worker_setup=1, **weights),
                         dependencies=list(dependencies), job=job, amplitude=amplitude))
    for amplitude in (0.1, 0.3):
        suffix = str(amplitude)
        add("oracle_" + suffix, "ORACLE", dict(oracle_step=768*256, oracle_query=768*6), amplitude=amplitude)
        add("bases_" + suffix, "BASES", dict(basis_vjp=1024), ("oracle_" + suffix,), amplitude=amplitude)
        add("capacity_" + suffix, "CAPACITY", dict(capacity_step=4*64*128),
            ("bases_" + suffix,), amplitude=amplitude)
    add("scaler", "SCALER", dict(scaler_observation=512*111))
    grid = [j["id"] for j in graph["jobs"] if j["stage"] == "SOURCE_GRID"]
    source = [j["id"] for j in graph["jobs"] if not j["arrivals"]]
    for job in graph["jobs"]:
        if job["arrivals"]:
            continue
        route = "mlp" if job["stage"] == "SOURCE_MLP" else job["arm"].lower()
        weights = {f"source_fit_{route}_step": 16000,
                   f"source_val_{route}_visit": 5*64*32*(1 if route == "mlp" else 2)}
        if route != "mlp":
            weights[f"source_cal_{route}_step"] = 5*1024
        deps = ["scaler", "bases_0.1", "bases_0.3"]
        if job["stage"] != "SOURCE_GRID":
            deps += ["grid_selection"]
        add(job["id"], "SOURCE_JOB", weights, deps, job=job)
    add("gradient_lr", "GRADIENT_LR", dict(gradient_lr_visit=3*3*2*64*4), ("source_index",))
    for job in graph["jobs"]:
        if not job["arrivals"]:
            continue
        add(job["id"] + "_online", "TARGET_JOB", {category(job): job["arrivals"]},
            ("artifact_lock",), job=job)
        add(job["id"] + "_score", "SCORE_JOB", dict(score_visit=job["arrivals"]),
            (job["id"] + "_online",), job=job)
    total = Counter()
    for row in rows:
        total.update(row["weights"])
    if total != units(graph) or len(rows) != 1521:
        raise ValueError("R8 execution queue differs from frozen matrix")
    return rows, {"grid_selection": grid, "source_index": source + ["grid_selection"],
                  "artifact_lock": ["source_index", "gradient_lr", "capacity_0.1", "capacity_0.3"]}
