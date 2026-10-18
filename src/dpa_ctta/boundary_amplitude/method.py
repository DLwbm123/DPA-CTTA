"""Preserve a frozen update direction and match F's actual parameter norm."""
import torch


def match(direction, reference):
    if direction.shape != reference.shape or direction.ndim != 1:
        raise ValueError('matching one-dimensional adaptation states required')
    if not torch.isfinite(direction).all() or not torch.isfinite(reference).all():
        raise ValueError('finite adaptation states required')
    raw_norm = direction.double().norm()
    target = reference.double().norm()
    if target > .050001:
        raise ValueError('F reference exceeds the frozen state ball')
    if target == 0:
        result = torch.zeros_like(direction)
    elif raw_norm <= 1e-12:
        raise ValueError('nonzero F radius cannot be matched from a zero direction')
    else:
        result = (direction.double()*(target/raw_norm)).to(direction)
    actual = result.double().norm()
    if not torch.isclose(actual, target, rtol=1e-5, atol=1e-8):
        raise ValueError('realized parameter amplitude mismatch')
    cosine = None
    if actual > 0 and raw_norm > 0:
        cosine = float(torch.dot(direction.double(), result.double())/(raw_norm*actual))
        if cosine < .999999:
            raise ValueError('normalization changed update direction')
    return result, dict(raw_norm=float(raw_norm), target_norm=float(target), actual_norm=float(actual),
                        norm_absolute_error=float(abs(actual-target)), direction_cosine=cosine,
                        multiplier=float(target/raw_norm) if raw_norm > 1e-12 else None)


def random_direction(dimensions, device):
    generator = torch.Generator().manual_seed(20261119)
    return torch.randn(dimensions, generator=generator).to(device)
