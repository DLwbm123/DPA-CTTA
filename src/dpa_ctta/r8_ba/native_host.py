"""Journal adapter for the pinned N, VPTTA, C and G current-image hosts."""
import copy
import re

import numpy as np
import torch

from ..host_diagnostic import rng as native_rng
from ..r7_shared.context import json_digest, tensor_digest
from ..r7_shared.numerics import COUNTS, finite
from .protocol import PROTOCOL_SHA256
from .rng import capture as capture_rng, restore as restore_rng


def _cpu(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_cpu(item) for item in value)
    return copy.deepcopy(value)


class NativeHost:
    def __init__(self, native, arm, identity):
        if arm not in ("N_SOURCE_EVAL", "VPTTA_NATIVE", "C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER"):
            raise ValueError("R8 registered native arm")
        if (set(identity) != {"code_sha", "protocol_sha256", "checkpoint_sha256", "registration_sha256", "seed"} or
                identity["protocol_sha256"] != PROTOCOL_SHA256 or
                any(not isinstance(identity[k], str) or re.fullmatch(
                    r"[0-9a-f]{40}" if k == "code_sha" else r"[0-9a-f]{64}", identity[k]) is None
                    for k in identity if k != "seed") or
                (identity["seed"] is not None and identity["seed"] not in range(20260907, 20260912))):
            raise ValueError("R8 native source/protocol identity")
        self.native, self.arm = native, arm
        self.is_cg = arm in ("C_CTTA_FIXED_LR", "G_CTTA_RELEASE_TRANSFER")
        self.is_vptta = arm == "VPTTA_NATIVE"
        if self.is_cg and native.arm != arm[0]:
            raise ValueError("R8 native C/G mismatch")
        if self.is_vptta and (native.mode != "base" or native.extra_weight != 0 or native.iters != 1):
            raise ValueError("R8 unmodified native VPTTA required")
        if native.model.training and not self.is_cg:
            raise ValueError("R8 native inference model must start in eval mode")
        self.adaptive = set(native.names) if self.is_cg else set()
        self.frozen_digest = tensor_digest(self._frozen_tensors())
        payload = dict(schema="R8_NATIVE_DEPLOYMENT_V1", arm=arm, identity=copy.deepcopy(identity),
                       initial_model_sha256=tensor_digest(list(native.model.state_dict().items())),
                       native_commit=getattr(native, "reference_commit", None),
                       torch=str(torch.__version__), cuda=torch.version.cuda)
        self.context = dict(payload=payload, sha256=json_digest(payload))
        self.visits, self.failed = 0, False
        self.versions = self._versions()
        self.handles = [native.model.register_forward_pre_hook(self._forward)]
        if self.is_cg or self.is_vptta:
            optimizer = native.base if self.is_cg else native.optimizer
            parameter = native.params[0] if self.is_cg else next(native.prompt.parameters())
            self.handles.append(parameter.register_hook(self._backward))
            self.handles.append(optimizer.register_step_post_hook(self._optimizer))

    def _frozen_tensors(self):
        return [(name, value) for name, value in self.native.model.state_dict().items()
                if name not in self.adaptive]

    def _versions(self):
        return tuple((name, id(value), value._version) for name, value in
                     list(self.native.model.named_parameters()) + list(self.native.model.named_buffers())
                     if name not in self.adaptive)

    def _forward(self, *_):
        COUNTS["backbone_forwards"] += 1

    def _backward(self, gradient):
        COUNTS["native_backward_calls"] += 1
        return gradient

    def _optimizer(self, *_):
        COUNTS["native_Adam"] += 1

    def check_frozen(self, boundary=False):
        if self._versions() != self.versions:
            raise ValueError("R8 native frozen tensor changed")
        if boundary and tensor_digest(self._frozen_tensors()) != self.frozen_digest:
            raise ValueError("R8 native frozen tensor content changed")

    def step(self, pixels):
        if self.failed:
            raise RuntimeError("R8 native host stopped at first failure")
        before = COUNTS.copy()
        try:
            self.check_frozen()
            output = self.native.step(pixels)
            logits, detail = output if self.is_cg else (output, {})
            finite(logits)
            if logits.shape != (1, 2, 512, 512):
                raise ValueError("R8 native prediction shape")
            self.check_frozen()
            delta = dict(COUNTS - before)
            expected = {"backbone_forwards": 9 if self.arm.startswith("G_") else
                        8 if self.is_cg else 2 if self.is_vptta else 1}
            if self.is_cg or self.is_vptta:
                expected.update(native_backward_calls=2 if self.arm.startswith("G_") else 1,
                                native_Adam=1)
            if delta != expected:
                raise ValueError("R8 native physical operation count mismatch")
            self.visits += 1
            return logits.detach().cpu(), dict(visit=self.visits, state_committed=True,
                                               counts=delta, native=detail)
        except BaseException:
            self.failed = True
            raise

    def snapshot(self):
        if self.failed:
            raise ValueError("R8 failed native host cannot snapshot")
        self.check_frozen(boundary=True)
        n = self.native
        state = dict(parameters={key: _cpu(value) for key, value in n.model.named_parameters()
                                 if key in self.adaptive},
                     gradients={key: None if value.grad is None else _cpu(value.grad)
                                for key, value in n.model.named_parameters()})
        if self.is_cg:
            state.update(base=_cpu(n.base.state_dict()), grata=_cpu(n.opt.state_dict()),
                         counts=n.counts.copy(), steps=n.steps, last=copy.deepcopy(n.last))
        if self.is_vptta:
            state.update(prompt=_cpu(n.prompt.state_dict()), optimizer=_cpu(n.optimizer.state_dict()),
                         prompt_gradients={key: None if value.grad is None else _cpu(value.grad)
                                           for key, value in n.prompt.named_parameters()},
                         memory=[(key.hex(), torch.from_numpy(value.copy())) for key, value in n.memory_bank.memory.items()],
                         all_keys=None if not hasattr(n.memory_bank, "all_keys") else
                         torch.from_numpy(n.memory_bank.all_keys.copy()),
                         counters=[(key, module.sample_num, module.new_sample)
                                   for key, module in n.model.named_modules() if isinstance(module, n.adabn)],
                         started=n._started)
        return dict(schema="R8_NATIVE_SNAPSHOT_V1", context_sha256=self.context["sha256"],
                    visits=self.visits, state=state, rng=capture_rng(), physical_counts=dict(COUNTS))

    def restore(self, snapshot):
        if (self.visits != 0 or snapshot.get("schema") != "R8_NATIVE_SNAPSHOT_V1" or
                snapshot.get("context_sha256") != self.context["sha256"] or
                type(snapshot.get("visits")) is not int or snapshot["visits"] < 0):
            raise ValueError("R8 native snapshot identity")
        self.check_frozen(boundary=True)
        n, state = self.native, snapshot["state"]
        params = dict(n.model.named_parameters())
        if set(state["parameters"]) != self.adaptive or set(state["gradients"]) != set(params):
            raise ValueError("R8 native snapshot parameter coverage")
        with torch.no_grad():
            for key, value in state["parameters"].items():
                if value.shape != params[key].shape or value.dtype != params[key].dtype:
                    raise ValueError("R8 native snapshot parameter layout")
                finite(value)
                params[key].copy_(value)
        for key, value in state["gradients"].items():
            if value is not None:
                finite(value)
            params[key].grad = None if value is None else value.to(params[key]).clone()
        if self.is_cg:
            if state["steps"] != snapshot["visits"]:
                raise ValueError("R8 native C/G snapshot step mismatch")
            n.base.load_state_dict(state["base"])
            n.opt.load_state_dict(state["grata"])
            n.counts, n.steps, n.last = state["counts"].copy(), state["steps"], copy.deepcopy(state["last"])
        if self.is_vptta:
            n.prompt.load_state_dict(state["prompt"], strict=True)
            n.optimizer.load_state_dict(state["optimizer"])
            for key, value in n.prompt.named_parameters():
                saved = state["prompt_gradients"][key]
                value.grad = None if saved is None else saved.to(value).clone()
            n.memory_bank.memory = {bytes.fromhex(key): value.numpy().copy() for key, value in state["memory"]}
            if state["all_keys"] is not None:
                n.memory_bank.all_keys = state["all_keys"].numpy().copy()
            modules = dict(n.model.named_modules())
            expected = {key for key, module in modules.items() if isinstance(module, n.adabn)}
            if {row[0] for row in state["counters"]} != expected or state["started"] != (snapshot["visits"] > 0):
                raise ValueError("R8 native AdaBN snapshot counters")
            for key, count, new_sample in state["counters"]:
                modules[key].sample_num, modules[key].new_sample = count, new_sample
            if state["started"]:
                n._initial_hook.remove()
                n._started = True
                n.prompt.eval()
        restore_rng(snapshot["rng"])
        if self.is_cg:
            n.rng = native_rng()
        COUNTS.clear()
        COUNTS.update(snapshot["physical_counts"])
        self.visits = snapshot["visits"]
        self.failed = False
        self.check_frozen(boundary=True)

    def close(self):
        for handle in self.handles:
            handle.remove()
