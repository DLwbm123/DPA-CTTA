"""Full-BN-vector matching in the recorded Adam metric."""
import torch

NULL_SEEDS = (1103, 2207, 3301, 4409)
NORM_RTOL = 0.005
DOT_COS_ATOL = 0.0005


def basis(columns):
    columns = columns.double()
    norms = columns.norm(dim=0)
    columns = columns[:, norms > 1e-30] / norms[norms > 1e-30]
    if not columns.shape[1]:
        return columns
    u, s, _ = torch.linalg.svd(columns, full_matrices=False)
    return u[:, s > s[0]*1e-10]


@torch.no_grad()
def proposals(native, joint, metric, base_gradient, lso_gradient):
    if any(x.shape != native.shape for x in (joint, metric, base_gradient, lso_gradient)):
        raise ValueError('matching full-vector shapes required')
    if any(not torch.isfinite(x).all() for x in (native, joint, metric, base_gradient, lso_gradient)) or (metric <= 0).any():
        raise ValueError('finite vectors and positive Adam metric required')
    # CPU double construction keeps the private null RNG separate from model RNG.
    root = metric.detach().double().cpu().sqrt()
    w, v = root*native.double().cpu(), root*joint.double().cpu()
    h0, hs = base_gradient.double().cpu()/root, lso_gradient.double().cpu()/root
    q = basis(torch.stack((w, h0, hs), dim=1))
    parallel = q@(q.T@v)
    residual = v-parallel
    if v.norm() > w.norm()*(1+1e-5)+1e-12:
        raise ValueError('fixed joint filter must contract in the Adam metric')
    ratio = float(v.norm()/w.norm()) if w.norm() else 1.
    candidates = {'radial': (w*ratio)/root}
    degenerate = bool(residual.norm() <= 1e-6*v.norm().clamp_min(1e-30))
    for i, seed in enumerate(NULL_SEEDS):
        generator = torch.Generator(device='cpu').manual_seed(seed)
        direction = torch.randn(w.shape, generator=generator, dtype=torch.float64)
        direction -= q@(q.T@direction)
        direction -= q@(q.T@direction)
        if direction.norm() <= 1e-12:
            raise ValueError('no null direction in eligible BN space')
        candidate = v if degenerate else parallel+residual.norm()*direction/direction.norm()
        for h in (h0, hs):
            if abs(float(h@(candidate-v))) > 1e-10*float(h.norm()*v.norm())+1e-20:
                raise ValueError('double-precision derivative matching failed')
        if abs(float(candidate.norm()-v.norm())) > 1e-10*float(v.norm())+1e-20:
            raise ValueError('double-precision norm matching failed')
        if abs(float((candidate-w).norm()-(v-w).norm())) > 1e-10*float(v.norm())+1e-20:
            raise ValueError('double-precision modification matching failed')
        candidates[f'null{i}'] = joint.detach().double().cpu().clone() if degenerate else candidate/root
    return candidates, dict(matched_basis_rank=q.shape[1], residual_fraction=float(residual.norm()/v.norm().clamp_min(1e-30)),
                           degenerate=degenerate, radial_scale=ratio)


@torch.no_grad()
def check_committed(native, joint, actual, metric, base_gradient, lso_gradient, radial=False):
    root = metric.double().sqrt()
    w, v, z = root*native.double(), root*joint.double(), root*actual.double()
    norm_error = float(abs(z.norm()-v.norm())/v.norm().clamp_min(1e-30))
    change_error = float(abs((z-w).norm()-(v-w).norm())/(v-w).norm().clamp_min(1e-30))
    dots = [float(abs((g.double()/root)@(z-v))/((g.double()/root).norm()*v.norm()).clamp_min(1e-30))
            for g in (base_gradient, lso_gradient)]
    values = dict(norm_relative_error=norm_error, modification_relative_error=change_error,
                  base_derivative_cosine_error=dots[0], lso_derivative_cosine_error=dots[1])
    if norm_error > NORM_RTOL or not radial and (change_error > NORM_RTOL or max(dots) > DOT_COS_ATOL):
        error = ValueError('representable-write geometry exceeds frozen tolerance')
        error.diagnostic = values
        raise error
    return values
