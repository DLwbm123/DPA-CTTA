"""Single upstream-frozen BN intervention; no state or parameter writes."""
import torch
from torch.nn import functional as F

ARMS = ('SOURCE', 'ADABN', 'VAR_ONLY', 'RB_BN')


def moments(h):
    if h.ndim != 4 or h.shape[0] != 1 or not torch.isfinite(h).all():
        raise ValueError('one finite BCHW input required')
    return torch.var_mean(h.double(), dim=(0, 2, 3), correction=0)


def installed(bn, mean, var, arm):
    if arm not in ARMS or not isinstance(bn, torch.nn.BatchNorm2d) or bn.training:
        raise ValueError('fixed standard eval BN required')
    ms, vs = bn.running_mean.double(), bn.running_var.double()
    if mean.shape != ms.shape or var.shape != vs.shape or not torch.isfinite(mean).all() or not torch.isfinite(var).all() or (var < 0).any() or (vs < 0).any():
        raise ValueError('invalid moments')
    if arm == 'SOURCE': return ms, vs
    if arm == 'ADABN': return mean, var
    if arm == 'VAR_ONLY': return ms, var
    ss, st = (vs + bn.eps).sqrt(), (var + bn.eps).sqrt()
    return mean - st / ss * (mean - ms), var


def output(bn, h, mean, var, arm):
    mu, v = installed(bn, mean, var, arm)
    return F.batch_norm(h, mu.to(h), v.to(h), bn.weight, bn.bias, False, 0., bn.eps)


def check_transport(bn, h, source, variance, balanced, mean, var):
    gamma = bn.weight.double()
    ss, st = (bn.running_var.double()+bn.eps).sqrt(), (var+bn.eps).sqrt()
    q = gamma*(mean-bn.running_mean.double())*(1/st-1/ss)
    obs = lambda y: y.double().mean((0,2,3))
    a, b, c = obs(source), obs(variance), obs(balanced)
    scale = max(1., float(a.abs().max()), float(b.abs().max()), float(c.abs().max()), float(q.abs().max()))
    e1 = float((b-a-q).abs().max()); e2 = float((c-b+q).abs().max()); e3 = float((c-a).abs().max())
    if not torch.isfinite(q).all() or max(e1,e2,e3) > 5e-5*scale:
        raise ValueError('executed local transport identity outside frozen tolerance')
    return dict(predicted_q_rms=float(q.square().mean().sqrt()),variance_mean_shift_rms=float((b-a).square().mean().sqrt()),rb_mean_shift_rms=float((c-a).square().mean().sqrt()),variance_q_max_error=e1,compensation_max_error=e2,rb_frozen_mean_max_error=e3,identity_scale=scale,channels=q.numel())
