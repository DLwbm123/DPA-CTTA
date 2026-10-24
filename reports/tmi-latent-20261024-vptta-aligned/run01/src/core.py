"""Independent source-supervised latent inference; no comparator modules or weights."""
import os, json, time, signal, random, csv, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.spatial import cKDTree
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(os.environ['WORK_ROOT'])
CFG = json.loads((ROOT / 'config.private.json').read_text())
STOP = False
COUNTS = {'encoder_images': 0, 'renderer_images': 0, 'head_images': 0,
          'outer_updates': 0, 'latent_gradient_calls': 0, 'target_domain_images': 0}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def log(value):
    print(json.dumps({'unix': time.time(), **value}, ensure_ascii=False), flush=True)


def stopped(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stopped)
signal.signal(signal.SIGINT, stopped)


def norm(c):
    return nn.GroupNorm(min(8, c), c)


class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.a = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.na = norm(cout)
        self.b = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.nb = norm(cout)
        self.skip = (nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), norm(cout))
                     if cin != cout or stride != 1 else nn.Identity())

    def forward(self, x):
        return F.relu(self.nb(self.b(F.relu(self.na(self.a(x))))) + self.skip(x))


def stack(cin, cout, n, stride):
    return nn.Sequential(Block(cin, cout, stride), *[Block(cout, cout) for _ in range(n - 1)])


class Up(nn.Module):
    def __init__(self, cin, skip, cout):
        super().__init__()
        self.block = Block(cin + skip, cout)

    def forward(self, x, y):
        return self.block(torch.cat([F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False), y], 1))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), norm(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        self.e1 = stack(64, 64, 3, 1)
        self.e2 = stack(64, 128, 4, 2)
        self.e3 = stack(128, 256, 6, 2)
        self.e4 = stack(256, 512, 3, 2)
        self.c4 = nn.Conv2d(64, 16, 1)
        self.c8 = nn.Conv2d(128, 32, 1)
        self.c16 = nn.Conv2d(768, 64, 1)
        self.style = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 32))
        self.relative_style = nn.Linear(32, 4)
        self.hu8, self.hu4 = Up(64, 32, 48), Up(48, 16, 32)
        self.distance = nn.Conv2d(32, 2, 3, 1, 1)
        self.ru8, self.ru4 = Up(64, 32, 64), Up(64, 16, 32)
        self.mod16, self.mod8, self.mod4 = nn.Linear(32, 128), nn.Linear(32, 128), nn.Linear(32, 64)
        self.rgb = nn.Sequential(nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(), nn.Conv2d(32, 3, 3, 1, 1))

    def encode(self, x):
        COUNTS['encoder_images'] += len(x)
        a = self.e1(self.stem(x)); b = self.e2(a); c = self.e3(b); d = self.e4(c)
        cc = torch.cat([c, F.interpolate(d, size=c.shape[-2:], mode='bilinear', align_corners=False)], 1)
        return [self.c4(a), self.c8(b), self.c16(cc)], self.style(a.mean((-2, -1)))

    def head(self, c, size):
        COUNTS['head_images'] += len(c[0])
        d = torch.tanh(self.distance(self.hu4(self.hu8(c[2], c[1]), c[0])))
        return F.interpolate(d, size=size, mode='bilinear', align_corners=False)

    @staticmethod
    def mod(x, s, layer):
        gain, bias = layer(s).chunk(2, 1)
        return x * (1 + .2 * torch.tanh(gain[:, :, None, None])) + .2 * torch.tanh(bias[:, :, None, None])

    def render(self, c, s, size):
        COUNTS['renderer_images'] += len(s)
        a = self.mod(c[2], s, self.mod16)
        b = self.mod(self.ru8(a, c[1]), s, self.mod8)
        d = self.mod(self.ru4(b, c[0]), s, self.mod4)
        return torch.sigmoid(F.interpolate(self.rgb(d), size=size, mode='bilinear', align_corners=False))


def segloss(d, y, dt):
    z = -16 * d
    p = z.sigmoid()
    dice = (2 * (p * y).sum((-2, -1)) + 1) / (p.sum((-2, -1)) + y.sum((-2, -1)) + 1)
    w = 1 + 4 * torch.exp(-8 * dt.abs())
    sdf = (F.smooth_l1_loss(d, dt, reduction='none') * w).mean()
    inclusion = F.relu(p[:, 1] - p[:, 0]).mean()
    return F.binary_cross_entropy_with_logits(z, y) + 1 - dice.mean() + .2 * sdf + .01 * inclusion


def obs(x, r):
    return (x - r).abs().mean() + .1 * (F.avg_pool2d(x, 4) - F.avg_pool2d(r, 4)).abs().mean()


class Prior:
    def __init__(self, means=None, var=None, weights=None, rate=None):
        self.means = torch.zeros(4, 32, device='cuda') if means is None else means.clone()
        self.var = torch.ones_like(self.means) if var is None else var.clone()
        self.weights = torch.ones(4, device='cuda') / 4 if weights is None else weights.clone()
        self.rate = CFG['mixture_rate'] if rate is None else rate
        self.floor = self.var.clone().clamp_min(.01) * .25

    def copy(self):
        p = Prior(self.means, self.var, self.weights, self.rate)
        p.floor = self.floor.clone()
        return p

    def component_logp(self, s):
        v = self.var.clamp_min(.0025)
        return -.5 * (((s[:, None] - self.means[None]).square() / v[None]) + v[None].log() + math.log(2 * math.pi)).sum(-1)

    def nll(self, s):
        return -torch.logsumexp(self.component_logp(s) + self.weights.clamp_min(1e-6).log()[None], 1).mean() / 32

    @torch.no_grad()
    def update(self, s):
        assert len(s) == 1, 'Streaming updates are strictly single-case'
        r = torch.softmax(self.component_logp(s) + self.weights.log()[None], 1)[0]
        a = self.rate * r[:, None]
        delta = s[0] - self.means
        self.means.add_(a * delta)
        self.var = ((1 - a) * self.var + a * delta.square()).clamp_min(self.floor)
        # The broad component remains source-anchored; only three narrow components adapt.
        self.means[3] = self.anchor_mean
        self.var[3] = self.anchor_var
        self.weights[:3] = (1 - self.rate) * self.weights[:3] + self.rate * r[:3]
        self.weights[:3] *= .9 / self.weights[:3].sum().clamp_min(1e-6)
        self.weights[3] = .1

    def anchor(self):
        self.anchor_mean = self.means[3].clone(); self.anchor_var = self.var[3].clone()
        return self

    def receipt(self):
        return {'components': 4, 'style_dimension': 32, 'variance_min': self.var.min().item(),
                'weight_min': self.weights.min().item(), 'rate': self.rate}


def objective(model, x, c, s, c0, scale, prior):
    trust = sum(((v - ref) / sc).square().mean() for v, ref, sc in zip(c, c0, scale)) / 3
    return obs(x, model.render(c, s, x.shape[-2:])) + CFG['lambda_content'] * trust + CFG['lambda_prior'] * prior.nll(s)


def infer(model, x, c0, s0, prior, differentiable=False):
    """No label argument. Network weights are never updated here."""
    c = list(c0) if differentiable else [v.detach().requires_grad_() for v in c0]
    s = s0 if differentiable else s0.detach().requires_grad_()
    ref = [v.detach() for v in c0]
    scales = [v.detach().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(.1) for v in c0]
    trace = []
    for step in range(3):
        params = [s] if step == 0 else c + [s]
        value = objective(model, x, c, s, ref, scales, prior)
        grads = torch.autograd.grad(value, params, create_graph=differentiable)
        COUNTS['latent_gradient_calls'] += 1
        if not all(torch.isfinite(g).all() for g in grads):
            raise FloatingPointError('Nonfinite latent gradient')
        deltas = []
        for j, g in enumerate(grads):
            dims = tuple(range(1, g.ndim))
            rms = g.detach().square().mean(dims, keepdim=True).sqrt().clamp_min(1e-8)
            factor = CFG['style_lr'] if j == len(grads) - 1 else CFG['content_lr'] * scales[j]
            deltas.append(factor * g / rms)
        accepted = False
        for shrink in [1., .5]:
            cand = [p - shrink * dv for p, dv in zip(params, deltas)]
            if step == 0:
                cc, ss = c, cand[0]
            else:
                cc = [torch.maximum(torch.minimum(v, r + CFG['content_radius'] * sc), r - CFG['content_radius'] * sc)
                      for v, r, sc in zip(cand[:3], ref, scales)]
                ss = cand[3]
            # The acceptance decision is discrete in both training and deployment.
            with torch.no_grad():
                check = objective(model, x, cc, ss, ref, scales, prior).item()
            if math.isfinite(check) and check <= value.detach().item() + 1e-7:
                c, s = cc, ss
                accepted = True
                break
        trace.append({'step': step, 'before': value.detach().item(), 'after': check, 'accepted': accepted})
        if not accepted:
            break
        if not differentiable:
            c = [v.detach().requires_grad_() for v in c]
            s = s.detach().requires_grad_()
    return c, s, trace


def paired(x):
    n = len(x)
    pars = torch.empty(n, 4, device=x.device).uniform_(-.2, .2)
    gamma = pars[:, 3].exp()[:, None, None, None]
    gains = pars[:, :3].exp()[:, :, None, None]
    shade = F.interpolate(torch.randn(n, 1, 4, 4, device=x.device) * .04, size=x.shape[-2:], mode='bilinear', align_corners=False)
    return (x.clamp_min(.001).pow(gamma) * gains * (1 + shade)).clamp(0, 1), pars


def calibrate(model, fit):
    styles = []
    # Source fit only; no development image contributes to the prior.
    for i in range(0, len(fit), 4):
        x = torch.stack([v[0] for v in fit[i:i + 4]]).cuda()
        styles.append(model.encode(x)[1])
    s = torch.cat(styles)
    centers = s[torch.tensor([0, len(s) // 3, 2 * len(s) // 3], device=s.device)].clone()
    for _ in range(8):
        ids = torch.cdist(s, centers).argmin(1)
        for k in range(3):
            if (ids == k).any(): centers[k] = s[ids == k].mean(0)
    gv = s.var(0, unbiased=False).clamp_min(.04)
    vs = []
    for k in range(3):
        sub = s[ids == k]
        vs.append(sub.var(0, unbiased=False).clamp_min(gv * .25) if len(sub) > 1 else gv)
    return Prior(torch.cat([centers, s.mean(0, keepdim=True)]), torch.stack(vs + [4 * gv]),
                 torch.tensor([.3, .3, .3, .1], device='cuda')).anchor()


