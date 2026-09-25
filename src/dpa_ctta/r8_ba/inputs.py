"""R8 source identity gate and bounded, source-only GPU input scope."""
import io
import copy
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import torch

from ..integrations.ctta_suite import build_reference_model
from ..r7_shared.io import checked_path
from ..r7_source_prep.registry import Reader, audit, verified
from .methods import R8Segmenter


def bind_metadata(refs):
    """Audit private registered metadata without decoding images or constructing a model."""
    if not isinstance(refs, dict) or set(refs) != {"manifest", "split", "target"}:
        raise ValueError("R8 source metadata references")
    docs = {}
    for name, ref in refs.items():
        if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
            raise ValueError("R8 source metadata file identity")
        docs[name] = json.loads(verified(ref["path"], ref["sha256"], 32 * 1024**2))
    result = audit(docs["manifest"], docs["split"], docs["target"])
    return dict(schema="R8_BOUND_SOURCE_METADATA_V1", refs=copy.deepcopy(refs),
                docs=docs, audit=result)


def _gpu_policy(physical_gpu):
    if (type(physical_gpu) is not int or physical_gpu not in (5, 6, 7) or
            os.environ.get("CUDA_VISIBLE_DEVICES") != str(physical_gpu) or
            os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"):
        raise ValueError("R8 physical GPU 5/6/7 and deterministic environment required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("R8 exactly one visible CUDA device required")


@contextmanager
def open_source(bound, source_root, checkpoint_path, amplitude, physical_gpu,
                max_asset_bytes, max_decoded_bytes, guard):
    """Yield (source data, R8 segmenter, IO counts); verify all used input on exit."""
    if (not isinstance(bound, dict) or bound.get("schema") != "R8_BOUND_SOURCE_METADATA_V1" or
            amplitude not in (0.1, 0.3) or not callable(guard) or
            type(max_asset_bytes) is not int or max_asset_bytes <= 0 or
            type(max_decoded_bytes) is not int or max_decoded_bytes <= 0):
        raise ValueError("R8 source input policy")
    _gpu_policy(physical_gpu)
    if bind_metadata(bound["refs"]) != bound:
        raise ValueError("R8 source metadata changed after binding")
    root = checked_path(source_root)
    if not root.is_dir():
        raise ValueError("R8 source root missing")
    docs = bound["docs"]
    groups = bound["audit"]["groups"]
    if groups * 5 * 512 * 512 * 4 > max_decoded_bytes:
        raise ValueError("R8 full source decode exceeds memory cap")
    checkpoint = docs["manifest"]["checkpoint"]
    counts = Counter()
    reader = Reader(docs["manifest"], docs["split"], docs["target"], root,
                    counts, max_asset_bytes)
    segmenter, first = None, None
    try:
        guard()
        data = reader.data()
        guard()
        raw = verified(checkpoint_path, checkpoint["sha256"], max_asset_bytes, counts)
        if len(raw) != checkpoint["bytes"]:
            raise ValueError("R8 source checkpoint size")
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
        model, _ = build_reference_model("fundus")
        model.load_state_dict(state, strict=True)
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in model.parameters()):
            raise ValueError("R8 source backbone dtype")
        segmenter = R8Segmenter(model, amplitude, device="cuda:0")
        from .worker_budget import attach_model
        attach_model(segmenter.model)
        guard()
        yield data, segmenter, counts
    except BaseException as exc:
        first = exc
        raise
    finally:
        try:
            if segmenter is not None:
                segmenter.close()
        finally:
            try:
                reader.after_check()
                verified(checkpoint_path, checkpoint["sha256"], max_asset_bytes, counts)
                for ref in bound["refs"].values():
                    verified(ref["path"], ref["sha256"], 32 * 1024**2, counts)
            except BaseException as exc:
                raise exc from first
