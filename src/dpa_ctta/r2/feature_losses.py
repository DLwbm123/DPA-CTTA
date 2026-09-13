"""Same-image losses: feature sum, token mean; region mean is the caller's job."""
import torch
from ..r1.streaming_pca import projection_residual


def unit(x):
    return x/x.norm(dim=1,keepdim=True).clamp_min(1e-6)


def paired_loss(fs, f0, U, full=False):
    if fs.ndim != 2 or fs.shape != f0.shape or fs.shape[1] != 32: raise ValueError('matched [tokens,32] features required')
    if U.ndim != 2 or U.shape[0] != 32 or not 1 <= U.shape[1] <= 8: raise ValueError('ready rank 1..8 required')
    if not len(fs): return fs.sum()*0
    delta = unit(fs)-unit(f0.detach().to(fs))
    if full: return delta.square().sum(1).mean()
    basis = U.detach().to(fs); residual = delta-(delta@basis)@basis.T
    return (32/(32-basis.shape[1]))*residual.square().sum(1).mean()


@torch.no_grad()
def energies(fs, f0, center, U):
    vs, v0 = unit(fs.detach()), unit(f0.detach().to(fs))
    b, c = U.detach().to(fs), center.detach().to(fs)
    residual = lambda x: x-(x@b)@b.T
    energy = lambda x: float(x.square().sum(1).mean())
    return dict(original_absolute_energy=energy(residual(v0-c)), strong_absolute_energy=energy(residual(vs-c)),
                delta_energy=energy(vs-v0), complement_delta_energy=energy(residual(vs-v0)))
