"""Frozen R8 cap checks and a profile-required full-matrix projection."""
import math
import re
from collections import Counter
import json

SAFETY = 1.3
CAPS = dict(gpu_seconds=1024 * 3600, model_forwards=100_000_000,
            backward_calls=4_500_000, optimizer_steps=4_000_000,
            vjp_calls=50_000, disk_bytes=200 * 1024**3)
MEASURES = tuple(CAPS) + ("peak_gpu_bytes",)


def json_size_bound(value):
    """JSON size bound for fixed-schema records and registered string fields."""
    if value is None:
        return 32  # Nullable metric fields may contain a finite float on another image.
    if type(value) is bool:
        return 5
    if type(value) is int:
        return max(12, len(str(value)))
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("R8 nonfinite record template")
        return 32
    if isinstance(value, str):
        return len(json.dumps(value).encode())
    if isinstance(value, (list, tuple)):
        return 2 + max(0, len(value) - 1) * 2 + sum(json_size_bound(v) for v in value)
    if isinstance(value, dict):
        return 2 + max(0, len(value) - 1) * 2 + sum(
            len(json.dumps(str(k)).encode()) + 2 + json_size_bound(v) for k, v in value.items())
    raise ValueError("R8 record template must contain JSON values")


def category(job):
    arm = job["arm"]
    if job["arrivals"] == 0:
        raise ValueError("R8 target category received source job")
    if arm in ("VPTTA_NATIVE", "C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER"):
        return {"VPTTA_NATIVE": "vptta", "C_CTTA_FIXED_LR": "c",
                "G_CTTA_RELEASE_TRANSFER": "g"}[arm]
    if arm in ("N_SOURCE_EVAL", "C0_CURRENT_STATS"):
        return "zero"
    if arm in ("R7_C_FULL", "R7_C_STATIC"):
        return "r7_c"
    if arm == "CURRENT_MLP":
        return "mlp"
    if "G1" in arm and arm.startswith("B_"):
        return "gradient_g1"
    if "G3" in arm and (arm.startswith("B_") or arm.startswith("COLD_")):
        return "gradient_g3"
    if arm == "B_ISTA_20":
        return "ista20"
    if arm.startswith("A"):
        return "new_a_max"
    if arm.startswith("B"):
        return "new_b_max"
    raise ValueError("R8 unprofiled target arm")


def units(graph):
    if (graph.get("source_training_jobs") != 65 or graph.get("target_jobs") != 724 or
            graph.get("target_arrivals") != 2_325_592):
        raise ValueError("R8 full task graph required")
    result = Counter(oracle_step=2 * (512 + 128 + 128) * 256,
                     worker_setup=65 + 2 + 2 + 1 + 2 + 1 + 2 * 724,
                     oracle_query=2 * (512 + 128 + 128) * 3 * 2,
                     capacity_step=2 * 4 * 64 * 128,
                     scaler_observation=512 * 111,
                     basis_vjp=2048,
                     gradient_lr_visit=3 * 3 * 2 * 64 * 4,
                     score_visit=graph["target_arrivals"])
    for job in graph["jobs"]:
        if job["arrivals"]:
            result[category(job)] += job["arrivals"]
        else:
            route = "mlp" if job["stage"] == "SOURCE_MLP" else job["arm"].lower()
            result[f"source_fit_{route}_step"] += 16000
            result[f"source_val_{route}_visit"] += 5 * 64 * 32 * (1 if route == "mlp" else 2)
            if route != "mlp":
                result[f"source_cal_{route}_step"] += 5 * 1024
    return result


def check_caps(observed):
    """Stop the whole R8 package when any measured aggregate reaches its cap."""
    if set(observed) != set(CAPS):
        raise ValueError("R8 aggregate cost ledger incomplete")
    for name, limit in CAPS.items():
        value = observed[name]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("R8 invalid observed resource value")
        if value >= limit:
            raise RuntimeError(f"R8 GLOBAL STOP: {name} cap reached")


def project(graph, profile, free_gpu_bytes, code_sha):
    """Every unit needs a same-code GPU observation; disk rows exclude fixed prediction bits."""
    weights = units(graph)
    if (profile.get("schema") != "R8_RESOURCE_PROFILE_V1" or
            profile.get("measured_on_physical_GPU") is not True or
            not isinstance(code_sha, str) or re.fullmatch(r"[0-9a-f]{40}", code_sha) is None or
            profile.get("code_sha") != code_sha or
            not isinstance(profile.get("physical_GPU_ids"), list) or
            not profile["physical_GPU_ids"] or
            any(type(i) is not int or i not in (5, 6, 7) for i in profile["physical_GPU_ids"]) or
            len(profile["physical_GPU_ids"]) != len(set(profile["physical_GPU_ids"])) or
            set(profile.get("units", {})) != set(weights) or
            type(profile.get("fixed_disk_bytes")) is not int or profile["fixed_disk_bytes"] < 0 or
            type(free_gpu_bytes) is not int or free_gpu_bytes <= 0):
        raise ValueError("R8 complete measured 567 resource profile required")
    projected = {key: 0.0 for key in CAPS}
    peak = 0
    for name, count in weights.items():
        row = profile["units"][name]
        if row.get("measured") is not True or set(row) != set(MEASURES) | {"measured", "sample_units"}:
            raise ValueError(f"R8 unmeasured/incomplete profile: {name}")
        if type(row["sample_units"]) is not int or row["sample_units"] < 1:
            raise ValueError("R8 profile sample count")
        for field in MEASURES:
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"R8 invalid profile: {name}/{field}")
        for field in CAPS:
            projected[field] += count * row[field]
        peak = max(peak, row["peak_gpu_bytes"])
    projected["disk_bytes"] += profile["fixed_disk_bytes"] + graph["target_arrivals"] * 65536
    with_margin = {name: math.ceil(value * SAFETY) for name, value in projected.items()}
    if math.ceil(peak * SAFETY) >= free_gpu_bytes:
        raise RuntimeError("R8 profile exceeds available GPU memory with margin")
    for name, value in with_margin.items():
        if value >= CAPS[name]:
            raise RuntimeError(f"R8 full matrix exceeds {name} cap with margin")
    return dict(schema="R8_CAPACITY_PROJECTION_V1", units=dict(weights),
                measured=projected, with_margin=with_margin,
                peak_gpu_bytes=peak, free_gpu_bytes=free_gpu_bytes,
                safety_factor=SAFETY, caps=CAPS.copy(), status="WITHIN_PROPOSED_CAPS")
