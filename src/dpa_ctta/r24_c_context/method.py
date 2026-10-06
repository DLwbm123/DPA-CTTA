"""C with matched static, per-view, or original-image-conditioned residuals."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from ..r20_model_only_search.method import Host as Base, Adapter
from ..r8_ba.rng import capture, restore


class ContextAdapter(Adapter):
    def __init__(self, channels, context_channels, device, seed):
        super().__init__(channels, 4, device, seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 1)
            self.gate = nn.Linear(2 * context_channels, 4)
            nn.init.normal_(self.gate.weight, std=1 / math.sqrt(2 * context_channels))
            nn.init.zeros_(self.gate.bias)
        self.to(device)
        self.context = None

    def forward(self, h):
        if self.context is None:
            raise ValueError('current-image context missing')
        scale = (h.square().mean(1, keepdim=True) + 1e-6).sqrt().detach()
        gate = 1 + .5 * torch.tanh(self.gate(self.context))
        a = self.U(F.gelu(self.V(h / scale)) * gate[:, :, None, None])
        delta = .1 * scale * torch.tanh(a)
        self.last = dict(relative_rms=float((delta.square().mean() / (h.square().mean() + 1e-12)).sqrt().detach()),
                         saturation=float((a.abs() > 3).float().mean().detach()),
                         gate_min=float(gate.min().detach()), gate_max=float(gate.max().detach()))
        return h + delta


def descriptor(x, source_mean, source_variance):
    """Frozen pre-first-BN feature moments; no semantic predictions or domain IDs."""
    x = x.detach().float()
    mean = x.mean((-2, -1))
    variance = x.var((-2, -1), unbiased=False)
    centered = (mean - source_mean) / (source_variance + 1e-5).sqrt()
    log_scale = .5 * torch.log((variance + 1e-5) / (source_variance + 1e-5))
    return torch.cat((centered, log_scale), 1).tanh().detach()


class Host(Base):
    def __init__(self, state, config, seed, identity, device='cuda:0', model=None):
        if config['id'] not in ('C', 'C_LR15', 'STATIC', 'VIEW', 'ANCHOR'):
            raise ValueError('unregistered condition')
        original = state if state is not None else copy.deepcopy(model.state_dict())
        super().__init__(state, config, seed, identity, device, model)
        self.mode = config['id']
        self.anchor = None
        self.context_differences = []
        self.used_context_differences = []
        if self.mode in ('C', 'C_LR15'):
            return
        n = self.native
        bn_name, bn = next((k, m) for k, m in n.model.named_modules() if isinstance(m, nn.BatchNorm2d))
        self.source_mean = original[bn_name + '.running_mean'].detach().to(self.device)[None]
        self.source_variance = original[bn_name + '.running_var'].detach().to(self.device)[None]
        if not torch.isfinite(self.source_mean).all() or not (self.source_variance >= 0).all():
            raise ValueError('checkpoint BN moments unavailable')
        saved = capture()
        try:
            self.adapter = ContextAdapter(n.model.up3.bn.num_features, bn.num_features, n.device, seed + 1000003)
        finally:
            restore(saved)
        self.handles.append(bn.register_forward_pre_hook(self._context))
        self.handles.append(n.model.up3.register_forward_hook(lambda m, a, h: self.adapter(h)))
        n.base.add_param_group({'params': list(self.adapter.parameters()), 'lr': 3e-4})
        n.opt.param_groups = [n.base.param_groups[0]]
        n.handles[2].remove()
        n.handles[2] = n.base.register_step_pre_hook(self._adapter_before_adam)

    def _context(self, module, args):
        d = descriptor(args[0], self.source_mean, self.source_variance)
        if self.anchor is None:
            self.anchor = d.clone()
        used = self.anchor if self.mode == 'ANCHOR' else (torch.zeros_like(d) if self.mode == 'STATIC' else d)
        self.adapter.context = used
        self.context_differences.append(float((d - self.anchor).abs().mean()))
        origin = torch.zeros_like(self.anchor) if self.mode == 'STATIC' else self.anchor
        self.used_context_differences.append(float((used - origin).abs().mean()))

    def _prepare_prototypes(self, x):
        # One condition shared through six weak views, strong view and final readout.
        self.anchor = None
        self.context_differences = []
        self.used_context_differences = []
        self.current_proto = {}

    def _lr(self, optimizer, args, kwargs):
        # Fixed C learning rates, never compound the multiplier across arrivals.
        optimizer.param_groups[0]['lr'] = 1.5e-4 if self.mode == 'C_LR15' else 1e-4
        if self.adapter is not None:
            optimizer.param_groups[1]['lr'] = 3e-4
        self.diag['BN_gradient_norm'] = float(torch.stack([p.grad.detach().norm() for p in self.native.params]).norm())
        self.diag['BN_lr'] = optimizer.param_groups[0]['lr']

    def step(self, x):
        z, t = super().step(x)
        if self.adapter is not None:
            if len(self.context_differences) != 8:
                raise ValueError('expected six weak, one strong and one readout context')
            t['diagnostics'].update(strong_context_distance=self.context_differences[6],
                                    strong_used_context_distance=self.used_context_differences[6],
                                    context_views=8,
                                    adapter_trainable_parameters=sum(p.numel() for p in self.adapter.parameters()))
        return z, t
