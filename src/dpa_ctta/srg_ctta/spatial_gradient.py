"""Spatial logit direction intervention, independently defined from CAREM."""
import math
import torch
from .method import Host as AffineHost, entropy, l2, normalize_parameters

KINDS = ('TANGENT', 'RADIAL', 'RANDOM_ORTH', 'NORM_ONLY', 'ENT_NORM')


def spatial_gradient(z, g, kind, seed):
    if kind not in KINDS or z.ndim != 4 or z.shape[:2] != (1, 2):
        raise ValueError('one image with independent OD/OC logits required')
    a, b = z.detach().double(), g.detach().double()
    axes = (-2, -1)
    energy = a.square().sum(axes, keepdim=True)
    radial = torch.where(energy > 0, (a*b).sum(axes, keepdim=True)/energy.clamp_min(1e-300), 0)*a
    tangent = b-radial
    raw_norm, tangent_norm = b.norm(), tangent.norm()
    if not torch.isfinite(raw_norm) or raw_norm <= 1e-20:
        raise ValueError('zero or nonfinite original gradient')
    fraction = float(tangent_norm/raw_norm)
    if kind != 'ENT_NORM' and fraction <= 1e-6:
        raise ValueError('spatial tangent numerically degenerate; no fallback')
    if kind == 'TANGENT':
        out = tangent
    elif kind == 'RADIAL':
        out = radial
    elif kind == 'RANDOM_ORTH':
        gen = torch.Generator(device='cpu').manual_seed(seed)
        direction = torch.randn(z.shape, generator=gen, dtype=torch.float64).to(a.device)
        out = b-(b*direction).sum(axes, keepdim=True)/direction.square().sum(axes, keepdim=True)*direction
    else:
        out = b
    norm = out.norm()
    if not torch.isfinite(norm) or norm <= 1e-20:
        raise ValueError('zero or nonfinite projected gradient')
    # Match output-gradient magnitude across the four direction controls, then
    # independently match the actual FP32 parameter step in Host.step.
    if kind != 'ENT_NORM':
        out = out*(tangent_norm/norm)
    out = out.to(g)
    written = out.double()
    cosine = float((written*b).sum()/(written.norm()*raw_norm))
    relative_radial = float(((written*a).sum(axes).abs() /
        (written.square().sum(axes).sqrt()*energy.squeeze(-1).squeeze(-1).sqrt()).clamp_min(1e-300)).max())
    if kind == 'TANGENT' and relative_radial > 5e-6:
        raise ValueError('written spatial tangent orthogonality failed')
    return out, dict(tangent_fraction=fraction, output_gradient_raw_norm=float(raw_norm),
        output_gradient_written_norm=float(written.norm()), output_gradient_cosine=cosine,
        output_radial_cosine_max=relative_radial)


class Host(AffineHost):
    def __init__(self, base, kind):
        if kind not in KINDS:
            raise ValueError('unknown spatial-gradient control')
        super().__init__(base, 'ENT_NORM')
        self.kind = kind

    def step(self, x, xw, xs, z0, seed):
        self.opt.zero_grad(set_to_none=True)
        z = self.forward(x)
        pre = z.detach().clone()
        loss = (entropy(z.sigmoid())*math.log(2)).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite base entropy')
        original = torch.autograd.grad(loss, z, retain_graph=True)[0]
        projected, audit = spatial_gradient(z, original, self.kind, seed)
        z.backward(projected)
        if any(p.grad is None for p in self.params) or not all(bool(torch.isfinite(p.grad).all()) for p in self.params):
            raise ValueError('invalid affine gradients')
        before = [p.detach().clone() for p in self.params]
        self.opt.step()
        with torch.no_grad():
            raw, actual, cosine = normalize_parameters(self.params, before)
            post = self.forward(x).detach()
        if raw <= 1e-12:
            raise ValueError('zero parameter proposal')
        self.total_steps += 1
        self.segment_steps += 1
        names, buffers = dict(self.model.named_parameters()), dict(self.model.named_buffers())
        if any(names[n]._version != v for n, v in self.fixed.items()):
            raise ValueError('frozen parameter changed')
        if any(not torch.equal(v, buffers[n]) for n, v in self.buffers.items()):
            raise ValueError('frozen BN buffers changed')
        if any(int(self.opt.state[p]['step']) != self.segment_steps for p in self.params):
            raise ValueError('Adam age differs')
        delta = (post-pre).double()
        observed_radial = float((delta*pre.double()).sum() /
            (delta.norm()*pre.double().norm()).clamp_min(1e-300))
        log = dict(loss=float(loss.detach()), entropy=float(loss.detach()), ranking=0.,
            anchor=0., anchor_gradient_norm=0., raw_step_norm=raw, actual_step_norm=actual,
            direction_cosine=cosine, norm_matched=abs(actual-.01)<=5e-7,
            zero_proposal=False, gate_fraction=1., observed_delta_radial_cosine=observed_radial, **audit)
        return pre, post, torch.ones_like(pre, dtype=torch.bool), pre, dict(steps=[log],
            total_affine_norm=l2([p-v for p,v in zip(self.params,self.initial)]),
            optimizer_age=self.segment_steps, forwards=2, optimizer_steps=1,
            loss_backward_calls=1, output_gradient_calls=1, diagnostic_anchor_gradient_calls=0)
