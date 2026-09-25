"""Construct the exact frozen target arm from verified in-memory assets."""
import io
import json
import random

import numpy as np
import torch

from ..b1_host import GRATA_COMMIT, Host as GraTaHost
from ..hosts.vptta import VPTTAHost
from ..integrations.ctta_suite import build_reference_model
from ..r7_shared.context import tensor_digest
from ..r7_source_prep.registry import verified
from ..r7_source_prep.runner import load_artifact, load_model
from ..source_pilot import SourceOnlyHost
from .context import capture
from .gradient import GradientHost
from .host import OnlineHost
from .methods import R8Segmenter, build, from_selected
from .native_host import NativeHost
from .protocol import PROTOCOL_SHA256
from .r7_control import FrozenR7Host
from .worker_budget import attach_model


def load_deployed(row, candidate, source_seed, mode, mlp=False):
    if row["config"] != candidate or row["source_seed"] != source_seed or row["mode"] != mode:
        raise ValueError("R8 target locked source config/mode/seed")
    raw = verified(row["artifact"]["path"], row["artifact"]["sha256"], 128 * 1024**2)
    artifact = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if mlp:
        if artifact.get("schema") != "R8_SOURCE_JOURNAL_V1":
            raise ValueError("R8 MLP selected source archive schema")
        artifact = artifact["snapshot"]
        if artifact.get("steps") != row["source_step"]:
            raise ValueError("R8 MLP selected source point")
    basis = artifact["method"]["basis"]
    if mlp:
        method = from_selected(artifact, candidate, basis, False, source_seed, row["binding"], mlp=True)
        method.freeze()
    else:
        if (artifact.get("schema") != "R8_CALIBRATED_METHOD_V1" or
                artifact.get("identity", {}).get("binding") != row["binding"] or
                artifact["identity"].get("source_step") != row["source_step"]):
            raise ValueError("R8 target calibrated source artifact binding")
        method = build(candidate, basis, static=mode == "STATIC", seed=source_seed)
        method.load_state_dict(artifact["method"], strict=True)
        method.freeze()
        if method.digest() != artifact.get("method_digest"):
            raise ValueError("R8 target source method digest")
    return method


def construct(job, resolved, assets, checkpoint_raw, refs, code_sha):
    arm, candidate = resolved["arm"], resolved["config"]
    seed = job["target_seed"] if job["target_seed"] is not None else 20260907
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    state = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    if arm in ("VPTTA_NATIVE", "C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER", "N_SOURCE_EVAL"):
        if arm == "VPTTA_NATIVE":
            native = VPTTAHost("fundus", source_state=state, device="cuda:0")
        elif arm == "N_SOURCE_EVAL":
            native = SourceOnlyHost("fundus", state, device="cuda:0")
        else:
            native = GraTaHost(arm[0], state=state, device="cuda:0")
        identity = dict(code_sha=code_sha, protocol_sha256=PROTOCOL_SHA256,
                        checkpoint_sha256=assets["checkpoint_sha256"],
                        registration_sha256=refs["target"]["sha256"], seed=job["target_seed"])
        host = NativeHost(native, arm, identity)
        return host, host.close
    if arm.startswith("R7_C_"):
        row = assets["r7_inventory"]
        inventory = json.loads(verified(row["path"], row["sha256"], 1024**2))
        segmenter = load_model(checkpoint_raw, device="cuda:0")
        native = load_artifact(segmenter, arm.removeprefix("R7_"), row["artifact_root"], inventory)
        return FrozenR7Host(native, arm, code_sha, row["sha256"]), segmenter.close
    model, _ = build_reference_model("fundus")
    model.load_state_dict(state, strict=True)
    amplitude = 0.1 if candidate is None else candidate["film_amplitude"]
    segmenter = R8Segmenter(model, amplitude, device="cuda:0")
    attach_model(segmenter.model)
    source = dict(checkpoint_sha256=assets["checkpoint_sha256"],
                  source_manifest_sha256=refs["manifest"]["sha256"],
                  source_split_sha256=refs["split"]["sha256"], spec_sha256=assets["spec_sha256"],
                  protocol_sha256=PROTOCOL_SHA256)
    if arm == "C0_CURRENT_STATS":
        # No oracle or basis is used by the deterministic current-statistics control.
        source.update(source_oracle_sha256="0" * 64, basis_sha256="0" * 64)
        config = dict(id=arm, film_amplitude=amplitude)
        return OnlineHost(segmenter, None, config, source, capture(segmenter, None, config, source)), segmenter.close
    row = assets["source_jobs"][resolved["source_job"]]
    method = load_deployed(row, candidate, job["source_seed"], resolved["mode"], mlp=arm == "CURRENT_MLP")
    source.update(source_oracle_sha256=row["oracle_receipt_sha256"], basis_sha256=row["bases_receipt_sha256"])
    config = candidate.copy()
    if resolved["gradient"]:
        arm = resolved["gradient"]
        scale = torch.tensor(row["gradient_scale"], dtype=torch.float64)
        lr = assets["gradient_lr_selection"][arm]
        config.update(gradient_arm=arm, gradient_lr=lr, grata_commit=GRATA_COMMIT,
                      scale_sha256=tensor_digest([("scale", scale)]))
        host = GradientHost(segmenter, method, config, source, capture(segmenter, method, config, source),
                            arm, scale, lr)
    else:
        host = OnlineHost(segmenter, method, config, source, capture(segmenter, method, config, source),
                          resolved["ablation"])
    return host, segmenter.close
