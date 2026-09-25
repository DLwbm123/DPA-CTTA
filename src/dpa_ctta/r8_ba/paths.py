"""Dedicated R8 NAS ownership boundary; never create outputs in R7 trees."""
from pathlib import Path

SOURCE_ROOT = Path("/data_nas/jiangsuiyang/CTTA/r8-ba-performance-envelope-v1/source")
TARGET_ROOT = SOURCE_ROOT.parent / "target"


def owned_source_path(value):
    root = SOURCE_ROOT.resolve()
    path = Path(value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("R8 source output outside dedicated root")
    return path


def owned_target_path(value):
    root = TARGET_ROOT.resolve()
    path = Path(value).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("R8 target output outside dedicated root")
    return path
