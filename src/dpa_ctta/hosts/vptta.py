"""Native VPTTA image step, plus one optional fixed-proxy loss in its native update."""

import ast
import copy
import importlib
import math

import torch

from ..integrations.ctta_suite import checkout_root, native_imports, INPUT_SIZES
from ..medical_losses import medical_loss
from ..proxy_loss import FixedProxy, proxy_supervision


def model_input_from_pixels(pixel_rgb, task):
    """Already-resized float32 pixels; no hidden resize, file I/O, or double normalization."""
    if task not in INPUT_SIZES:
        raise ValueError("unknown task")
    size = INPUT_SIZES[task]
    if (not isinstance(pixel_rgb, torch.Tensor) or pixel_rgb.ndim != 4
            or pixel_rgb.shape[0] == 0 or pixel_rgb.shape[1:] != (3, size, size)
            or pixel_rgb.dtype != torch.float32 or pixel_rgb.device.type != "cpu"
            or not torch.isfinite(pixel_rgb).all() or ((pixel_rgb < 0) | (pixel_rgb > 1)).any()):
        raise ValueError("already-resized CPU float32 pixel_rgb in [0,1] required")
    if task == "fundus":
        lo = pixel_rgb.amin((1, 2, 3), keepdim=True)
        span = pixel_rgb.amax((1, 2, 3), keepdim=True) - lo
        if (span == 0).any():
            raise ValueError("native Fundus min-max normalization is undefined for a constant image")
        return (pixel_rgb - lo) / span
    mean = pixel_rgb.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = pixel_rgb.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    return (pixel_rgb - mean) / std


def native_step_from_source(path, adabn, *, proxy=False):
    """Compile only the pinned run() image block; no constructor/loader/evaluator executes.

    The base AST is unchanged from model.eval through memory.push. The enabled
    variant changes only `loss = bn_loss / times` to add the proxy term.
    Upstream source is used at runtime, not copied or relicensed into this package.
    """
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "VPTTA")
    run = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "run")
    loop = next(n for n in run.body if isinstance(n, ast.For))
    texts = [ast.unparse(n) for n in loop.body]
    start = texts.index("self.model.eval()")
    stop = next(i for i, text in enumerate(texts) if text.startswith("self.memory_bank.push("))
    body = copy.deepcopy(loop.body[start:stop + 1])
    if any(isinstance(n, ast.Name) and n.id in {"y", "data", "path"} for stmt in body for n in ast.walk(stmt)):
        raise RuntimeError("unexpected label/data reference in native image block")
    assignments = [n for stmt in body for n in ast.walk(stmt)
                   if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "loss"]
    if len(assignments) != 1 or ast.unparse(assignments[0].value) != "bn_loss / times":
        raise RuntimeError("unexpected native loss/update structure")
    if proxy:
        assignments[0].value = ast.BinOp(assignments[0].value, ast.Add(),
                                        ast.parse("self.extra_weight * self._proxy_term()", mode="eval").body)
    function = ast.parse("def step(self, x):\n    pass\n").body[0]
    function.body = body + [ast.Return(ast.Name("pred_logit", ast.Load()))]
    code = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"torch": torch, "AdaBN": adabn}
    exec(compile(code, str(path), "exec"), namespace)
    return namespace["step"]


class VPTTAHost:
    def __init__(self, task, *, root=None, mode="base", extra_weight=0.0,
                 proxy_factory=None, beta_boundary=0.1, neighbor=16, source_state=None, device="cpu"):
        if task not in INPUT_SIZES or mode not in {"base", "proxy_rehearsal"}:
            raise ValueError("task/mode outside the minimal host contract")
        if not math.isfinite(extra_weight) or extra_weight < 0 or (mode == "base" and extra_weight != 0):
            raise ValueError("finite nonnegative extra_weight; base requires zero")
        if not math.isfinite(beta_boundary) or beta_boundary < 0 or type(neighbor) is not int or neighbor < 1:
            raise ValueError("invalid boundary weight or neighbor count")
        root, self.reference_commit = checkout_root(root)
        self.task, self.mode, self.extra_weight = task, mode, extra_weight
        self.beta_boundary, self.neighbor, self.iters = beta_boundary, neighbor, 1
        self.last_proxy_loss = None
        with native_imports(root, task) as directory:
            Prompt = importlib.import_module("utils.prompt").Prompt
            Memory = importlib.import_module("utils.memory").Memory
            self.adabn = importlib.import_module("utils.convert").AdaBN
            # Match native construction order and RNG: prompt, then random full model.
            self.prompt = Prompt(prompt_alpha=.01, image_size=INPUT_SIZES[task])
            if task == "fundus":
                cls = importlib.import_module("networks.ResUnet_TTA").ResUnet
                self.model = cls(resnet="resnet34", num_classes=2, pretrained=False, warm_n=5).cpu()
            else:
                cls = importlib.import_module("networks.PraNet_Res2Net_TTA").PraNet
                self.model = cls(warm_n=5).cpu()
            self.native_step = native_step_from_source(directory / "vptta.py", self.adabn)
            self._proxy_step = native_step_from_source(directory / "vptta.py", self.adabn, proxy=True) if extra_weight else None
            self.memory_bank = Memory(size=40, dimension=self.prompt.data_prompt.numel())
        self.model.eval()
        if source_state is not None:
            load_source_state(self.model, source_state)
        self.device = torch.device(device)
        self._started = False
        if extra_weight:
            if not callable(proxy_factory):
                raise ValueError("enabled proxy branch requires a prepared-proxy factory")
            self.proxy = proxy_factory()
            if type(self.proxy) is not FixedProxy:
                raise ValueError("only FixedProxy source/fixture payloads are accepted")
            self._proxy_input = model_input_from_pixels(self.proxy.pixel_rgb, task)
            shape = (len(self._proxy_input), 2 if task == "fundus" else 1, *self._proxy_input.shape[-2:])
            medical_loss(torch.zeros(shape), self.proxy.mask, self.proxy.signed_distance, beta_boundary)
            # ponytail: one full frozen source clone isolates native hooks/buffers; replace
            # with a reviewed functional forward only if this extra model memory matters.
            self._proxy_model = copy.deepcopy(self.model).eval().requires_grad_(False)
            self._proxy_model.to(self.device)
            self._proxy_input = self._proxy_input.to(self.device)
            self.proxy = FixedProxy(self.proxy.pixel_rgb.to(self.device), self.proxy.mask.to(self.device),
                                    None if self.proxy.signed_distance is None else self.proxy.signed_distance.to(self.device),
                                    self.proxy.provenance)
        self.model.to(self.device)
        self.prompt.to(self.device)
        self.device = self.prompt.data_prompt.device  # Resolve unindexed CUDA / CPU:0 to actual tensor device.
        # Preserve native base .grad behavior; Adam owns final-device prompt objects only.
        self.optimizer = torch.optim.Adam(self.prompt.parameters(), lr=.05 if task == "fundus" else .01,
                                          betas=(.9, .99), weight_decay=0)
        self.audit_initial_state()
        def before_first_forward(_model, _args):
            self.audit_initial_state()
            self._started = True
            self._initial_hook.remove()
        self._initial_hook = self.model.register_forward_pre_hook(before_first_forward)

    def audit_initial_state(self):
        """First-forward guard catches the known load-only-live integration error."""
        if self._started or self.optimizer.state or self.memory_bank.get_size():
            raise RuntimeError("initial-state audit must precede all forwards")
        params = [p for group in self.optimizer.param_groups for p in group['params']]
        if [id(p) for p in params] != [id(p) for p in self.prompt.parameters()]:
            raise RuntimeError("optimizer does not own current prompt parameters")
        if any(m.sample_num != 0 for m in self.model.modules() if isinstance(m, self.adabn)):
            raise RuntimeError("native counters already advanced")
        for model in (self.model, self.prompt):
            if any(t.device != self.device for t in model.state_dict().values()):
                raise RuntimeError("final device mismatch")
        if self.extra_weight:
            live, clone = self.model.state_dict(), self._proxy_model.state_dict()
            if live.keys() != clone.keys() or any(
                not torch.equal(t, clone[k]) or t.data_ptr() == clone[k].data_ptr()
                or clone[k].device != self.device for k, t in live.items()
            ) or any(p.requires_grad for p in self._proxy_model.parameters()):
                raise RuntimeError("SOURCE_CLONE_STATE_MISMATCH")
            if any(t.device != self.device or t.dtype != torch.float32 for t in
                   (self._proxy_input, self.proxy.pixel_rgb, self.proxy.mask, self.proxy.signed_distance) if t is not None):
                raise RuntimeError("prepared proxy device/dtype mismatch")

    def _proxy_term(self):
        native = dict(self.model.named_modules())
        for name, module in self._proxy_model.named_modules():
            if isinstance(module, self.adabn):
                module.sample_num = native[name].sample_num
                module.new_sample = False
        # Source buffers are immutable in native eval; the clone never touches native
        # bn_loss/feature hooks. No graph-saved buffer is restored in-place.
        self.last_proxy_loss = proxy_supervision(self._proxy_model, self.prompt, self._proxy_input,
                                                self.proxy, self.beta_boundary)
        return self.last_proxy_loss.total

    def step(self, pixel_rgb):
        if (not isinstance(pixel_rgb, torch.Tensor) or pixel_rgb.ndim != 4
                or pixel_rgb.shape[0] != 1 or pixel_rgb.requires_grad):
            raise ValueError("one fixed current pixel_rgb image required; no online labels")
        model_input = model_input_from_pixels(pixel_rgb, self.task).to(self.device)
        if self.extra_weight == 0:
            return self.native_step(self, model_input)
        return self._proxy_step(self, model_input)


def load_source_state(model, source_state):
    """Strict tensor-only in-memory state. Disk deserialization belongs to the runner."""
    own = model.state_dict()
    if not isinstance(source_state, dict) or own.keys() != source_state.keys():
        raise ValueError("source state keys mismatch")
    for key, value in source_state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != own[key].shape
                or value.dtype != own[key].dtype or not torch.isfinite(value).all()):
            raise ValueError("source state shape/dtype/nonfinite mismatch: " + key)
    model.load_state_dict(source_state, strict=True)
