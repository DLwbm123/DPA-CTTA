"""Content-bound R8 deployment identity, separate from the frozen R7 schema."""
import copy
import re

import torch
from torch import nn

from ..r7_shared.context import json_digest, layout, tensor_digest, tensors
from .protocol import PROTOCOL_SHA256

SOURCE_KEYS = {"checkpoint_sha256", "source_manifest_sha256", "source_split_sha256",
               "source_oracle_sha256", "basis_sha256", "spec_sha256", "protocol_sha256"}


def capture(segmenter, method, config, source):
    if not isinstance(source, dict) or set(source) != SOURCE_KEYS or any(
            not isinstance(v, str) or re.fullmatch(r"[0-9a-f]{64}", v) is None for v in source.values()):
        raise ValueError("complete R8 source identity required")
    if source["protocol_sha256"] != PROTOCOL_SHA256:
        raise ValueError("R8 frozen protocol identity mismatch")
    if not isinstance(config, dict) or not config.get("id"):
        raise ValueError("R8 configuration identity required")
    if segmenter.inference_policy.get("schema") != "R8_SEGMENTER_POLICY_V1":
        raise ValueError("R8 segmenter policy required")
    if (segmenter.inference_policy.get("FiLM") !=
            f"up1_up3_256_each_gamma_beta_expm1_{segmenter.amplitude}_tanh_v1" or
            config.get("film_amplitude") != segmenter.amplitude):
        raise ValueError("R8 FiLM/config amplitude mismatch")
    if any(p.requires_grad for p in segmenter.model.parameters()):
        raise ValueError("R8 backbone must be frozen")
    for module in segmenter.model.modules():
        if module.training:
            raise ValueError("R8 backbone eval required")
        if isinstance(module, nn.BatchNorm2d) and (module.track_running_stats or module.running_mean is not None or module.running_var is not None):
            raise ValueError("R8 current statistics BN required")
    if method is not None and (method.stage != "online" or any(p.requires_grad for p in method.parameters()) or
                               method.amplitude != segmenter.amplitude or not method.observer.fitted):
        raise ValueError("R8 method must be frozen and matched to segmenter")
    if method is not None and config.get("rank") != method.rank:
        raise ValueError("R8 method/config rank mismatch")
    if method is not None and hasattr(method, "observation") and config.get("observer") != method.observation:
        raise ValueError("R8 method/config observer mismatch")
    if method is not None and hasattr(method, "aux_multiplier") and config.get("aux_multiplier") != method.aux_multiplier:
        raise ValueError("R8 method/config auxiliary weight mismatch")
    if method is not None and hasattr(method, "assert_deployment_eta"):
        method.assert_deployment_eta()
    payload = dict(schema="R8_DEPLOYMENT_PAYLOAD_V1", source=copy.deepcopy(source), config=copy.deepcopy(config),
                   environment=dict(policy=copy.deepcopy(segmenter.inference_policy),
                                    backbone_sha256=tensor_digest(tensors(segmenter.model)),
                                    projection_sha256=tensor_digest([("projection", segmenter.projection)]),
                                    model_layout=layout(segmenter.model), torch=torch.__version__,
                                    cuda=torch.version.cuda if segmenter.projection.device.type == "cuda" else None),
                   method=None if method is None else dict(kind=type(method).__name__,
                                                           mode="STATIC" if method.static else "FULL",
                                                           digest=method.digest()))
    return dict(schema="R8_DEPLOYMENT_CONTEXT_V1", payload=payload, sha256=json_digest(payload))


def require(actual, expected):
    if (not isinstance(expected, dict) or set(expected) != {"schema", "payload", "sha256"} or
            expected["schema"] != "R8_DEPLOYMENT_CONTEXT_V1" or
            expected["sha256"] != json_digest(expected["payload"]) or actual != expected):
        raise ValueError("R8 deployment identity mismatch")
