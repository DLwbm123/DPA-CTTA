"""The two user-frozen R8 protocol choices absent from the input ZIP."""
import hashlib
import json
from pathlib import Path


_raw = Path(__file__).with_name("protocol.json").read_bytes()
DECISIONS = json.loads(_raw)
if (DECISIONS["schema"] != "R8_PROTOCOL_DECISIONS_V1" or
        DECISIONS["source_val_selection_metric"] != "soft_Dice" or
        DECISIONS["capacity_probe_direct_latent"] != dict(
            optimizer="Adam", lr=0.03, betas=[0.9, 0.999], eps=1e-8,
            weight_decay=0.0, steps=128)):
    raise ValueError("R8 frozen protocol decisions changed")
from .scope import SCREEN, SPEC_PATH
PROTOCOL_SHA256 = hashlib.sha256(_raw + (SPEC_PATH.read_bytes() if SCREEN else b"" )).hexdigest()
CAPACITY_OPTIMIZER = DECISIONS["capacity_probe_direct_latent"]
SELECTION_METRIC = DECISIONS["source_val_selection_metric"]
