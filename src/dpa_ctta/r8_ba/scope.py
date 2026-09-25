"""Explicit process scope; original full R8 remains the default."""
import os
from pathlib import Path

SCOPE = os.environ.get("R8_SCOPE", "FULL")
if SCOPE not in ("FULL", "SCREEN24"):
    raise ValueError("Unknown R8 execution scope")
SCREEN = SCOPE == "SCREEN24"
ROOT = Path(__file__).resolve().parents[3]
GRAPH_PATH = ROOT / ("docs/review/r8_screen24/TASK_GRAPH.json" if SCREEN else "docs/review/r8/TASK_GRAPH.static.json")
SPEC_PATH = ROOT / ("docs/review/r8_screen24/SPEC.json" if SCREEN else "docs/review/r8/input/R8_EXPERIMENT_SPEC.json")
SOURCE_JOBS, TARGET_JOBS, ARRIVALS = (10,40,78040) if SCREEN else (65,724,2325592)
