"""Validated method configuration."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class DPAConfig:
    descriptor_schema_version: int = 2
    task: str = "fundus"
    latent_dim: int = 16
    precision_floor: float = 1e-3
    temporal_lambda: float = 1.0
    temperature: float = 1.0
    feature_projection_dim: int = 8
    reference_repo: str = "DLwbm123/CTTA"
    reference_commit: str = "dbff0d985c6c95345d9fb78f5b1daef57b392564"
    reference_package: str = "ctta-repro-suite"

    def __post_init__(self):
        if self.descriptor_schema_version != 2:
            raise ValueError("descriptor schema 2 required; old artifacts must be rebuilt")
        if self.task not in {"fundus", "polyp"}:
            raise ValueError("task must be 'fundus' or 'polyp'")
        if self.latent_dim != 16:
            raise ValueError("DPA-CTTA v0 fixes latent_dim=16")
        for name in ("precision_floor", "temporal_lambda", "temperature"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.feature_projection_dim <= 0:
            raise ValueError("feature_projection_dim must be positive")
        if self.reference_repo != "DLwbm123/CTTA" or self.reference_commit != "dbff0d985c6c95345d9fb78f5b1daef57b392564":
            raise ValueError("DPA-CTTA v0 reference repo/commit is fixed")
        if self.reference_package != "ctta-repro-suite":
            raise ValueError("DPA-CTTA v0 reference package is fixed")

    @property
    def num_classes(self):
        return 2 if self.task == "fundus" else 1

    @property
    def descriptor_dim(self):
        anatomy = 7 * self.num_classes + (2 if self.task == "fundus" else 1)
        return 9 + self.feature_projection_dim + anatomy

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if values.get("descriptor_schema_version") != 2:
            raise ValueError("explicit descriptor_schema_version=2 required")
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**values)

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))
