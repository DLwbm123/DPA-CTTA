"""The three explicitly backward-enabled B latent diagnostic arms."""
import copy

import torch
from torch.nn import functional as F

from ..b1_host import GRATA_COMMIT, official
from ..hosts.vptta import model_input_from_pixels
from ..r7_shared.context import tensor_digest
from ..r7_shared.numerics import COUNTS, finite
from .host import OnlineHost
from .methods import R8B

STEPS = {"B_G1": 1, "B_G3": 3, "COLD_G3": 3}
LR = {0.001, 0.01, 0.1}


class GradientHost(OnlineHost):
    def __init__(self, segmenter, method, config, source, expected_context, arm, scale, lr, api=None):
        if (arm not in STEPS or not isinstance(method, R8B) or method.static or
                lr not in LR or scale.shape != (method.rank,) or scale.dtype != torch.float64):
            raise ValueError("R8 gradient diagnostic configuration")
        finite(scale)
        if (scale < 1e-3).any():
            raise ValueError("R8 gradient scale floor")
        self.arm, self.scale, self.lr = arm, scale.detach().clone(), lr
        self.scale_sha256 = tensor_digest([("scale", self.scale)])
        if (config.get("gradient_arm") != arm or config.get("gradient_lr") != lr or
                config.get("scale_sha256") != self.scale_sha256 or
                config.get("grata_commit") != GRATA_COMMIT):
            raise ValueError("R8 gradient identity")
        super().__init__(segmenter, method, config, source, expected_context)
        self.api = official() if api is None else api

    def check_frozen(self, boundary=False):
        super().check_frozen(boundary)
        config = self.context["payload"]["config"]
        if (config["gradient_arm"] != self.arm or config["gradient_lr"] != self.lr or
                config["grata_commit"] != GRATA_COMMIT or
                tensor_digest([("scale", self.scale)]) != self.scale_sha256):
            raise ValueError("R8 gradient configuration changed")

    def step(self, current_image):
        if self.failed:
            raise RuntimeError("R8 gradient host stopped at first error")
        before = COUNTS.copy()
        try:
            self.check_frozen()
            x = model_input_from_pixels(current_image, "fundus")
            original = x.to(self.segmenter.projection.device)
            with torch.no_grad():
                if self.arm == "COLD_G3":
                    zero_logits = self.segmenter(current_image)
                    next_state = dict(z=torch.zeros_like(self.state["z"]),
                                      d=torch.zeros_like(self.state["d"]), counter=self.visits + 1)
                else:
                    zero_logits, raw, tokens = self.segmenter(current_image, observe=True)
                    next_state, _ = self.method.update(raw, tokens, self.state)
                weak = self.api.Rotate_and_Flip()
                probabilities = [zero_logits.sigmoid()]
                for factor in range(5):
                    logits = self.segmenter.normalized(weak(original, factor).cpu())
                    probabilities.append(weak.inverse(logits.to(original.device), factor).cpu().sigmoid())
                target = torch.stack(probabilities).mean(0).detach()
                strong = self.api.augmentation_strong_style({"data": x.numpy().copy()})
                strong = self.api.normalize_image_to_0_1(
                    torch.from_numpy(strong).float().to(original.device)).cpu()
            initial = next_state["z"].detach().clone()
            if self.arm == "COLD_G3":
                initial.zero_()
            u0 = initial / self.scale
            u = torch.nn.Parameter(u0.clone())
            optimizer = torch.optim.Adam([u], lr=self.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
            losses = []
            for _ in range(STEPS[self.arm]):
                optimizer.zero_grad(set_to_none=True)
                ambient = (self.method.basis @ (self.scale * u)).float()
                strong_logits = self.segmenter.normalized(strong, ambient)
                loss = F.binary_cross_entropy_with_logits(strong_logits, target) + 0.01 * (u - u0).square().mean()
                finite(loss)
                loss.backward()
                COUNTS["target_backward_calls"] += 1
                finite(u.grad)
                optimizer.step()
                COUNTS["target_Adam"] += 1
                finite(u)
                losses.append(float(loss.detach()))
            corrected = (self.scale * u.detach()).clone()
            if self.arm != "COLD_G3":
                next_state["z"] = corrected
            self.method.validate_state(next_state)
            with torch.no_grad():
                logits = self.segmenter(current_image, (self.method.basis @ corrected).float())
            finite(logits)
            if logits.shape != (1, 2, 512, 512):
                raise ValueError("R8 gradient prediction shape")
            counts = dict(COUNTS - before)
            if (counts.get("backbone_forwards", 0) != 7 + STEPS[self.arm] or
                    counts.get("target_backward_calls", 0) != STEPS[self.arm] or
                    counts.get("target_Adam", 0) != STEPS[self.arm]):
                raise ValueError("R8 gradient physical count mismatch")
            self.state = copy.deepcopy(next_state)
            self.visits += 1
            return logits.detach(), dict(visit=self.visits, state_committed=True, counts=counts,
                                         arm=self.arm, loss=losses, lr=self.lr)
        except Exception:
            self.failed = True
            raise
