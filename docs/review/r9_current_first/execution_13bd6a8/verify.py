"""Verify the anonymous partial rejection certificate, not full admission."""
import json
from pathlib import Path

p = json.loads(Path(__file__).with_name("RESOURCE_REJECTION.json").read_text())
assert p["status"] == "OVER_CAP" and p["execution_authorized"] is False
assert p["profile_complete"] is False and p["target_accessed"] is False
assert p["completed_source_jobs"] == p["completed_target_slots"] == 0
assert p["known_units"] + len(p["missing_units"]) == p["required_units"]
for key in p["caps"]:
    base = sum(row[key] for row in p["known_unit_contributions"].values())
    assert base == p["base_measured_projection_lower_bound"][key]
    expected = base + p["recovery_reserve"][key] + p["profile_cost"][key]
    assert abs(expected - p["required_projection_lower_bound"][key]) < 1e-6
assert p["base_measured_projection_lower_bound"]["gpu_seconds"] > p["caps"]["gpu_seconds"]
print("PASS: partial monotone rejection; full profile and scientific round remain incomplete")
