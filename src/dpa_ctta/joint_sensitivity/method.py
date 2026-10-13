"""Two acquisition probes, two structure VJPs, one filtered native Adam step."""
import copy
import math
import torch
from torch.nn import functional as F
from ..b1_host import finite, official
from ..host_diagnostic import rng, restore
from ..hosts.vptta import model_input_from_pixels
from ..r20_model_only_search.method import boundary
from ..r36_incremental_modules.method import smooth, structure
from ..r37_cw_orientation.method import Host as BaseHost, candidates as base_candidates


def candidates():
    c, w, _, lso = base_candidates()
    return [c, w, lso, dict(lso, id='CW_LSO_JS', joint_strength=1.)]


def photo_probe(pixels, index):
    if index not in (0, 1):
        raise ValueError('two fixed acquisition probes')
    z = torch.logit(pixels.clamp(1e-6, 1-1e-6))
    z = 1.10*z if index == 0 else z+z.new_tensor([.05, 0., -.05]).view(1, 3, 1, 1)
    return torch.where((pixels == 0) | (pixels == 1), pixels, z.sigmoid())


def field(logits):
    return smooth(F.avg_pool2d(logits.sigmoid(), 4))


def gradient_norm(x):
    k = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])/8
    padded = F.pad(x, (1,)*4, mode='replicate')
    gx = F.conv2d(padded, k[None, None].repeat(2, 1, 1, 1), groups=2)
    gy = F.conv2d(padded, k.T[None, None].repeat(2, 1, 1, 1), groups=2)
    return (gx.square()+gy.square()).sqrt().clamp_min(1e-3)


class Sensor:
    def __init__(self, original, predictions):
        q = predictions.mean(0).detach()
        variance = (predictions-q).square().mean(0)
        _, magnitude = structure(smooth(F.avg_pool2d(q, 4)))
        band = F.max_pool2d(boundary(q >= .5, 3).float(), 4)
        nested = ~((q[:, 1:2] >= .5) & (q[:, 0:1] < .5))
        valid = F.avg_pool2d(nested.float(), 4) == 1
        reliability = magnitude/magnitude.amax((-2, -1), keepdim=True).clamp_min(1e-6)
        evidence = band*(1-4*F.avg_pool2d(variance, 4)).clamp_min(0)*reliability*valid*(magnitude > 2e-6)
        self.weight = (evidence/(band.sum((-2, -1), keepdim=True)+1e-8)).sqrt().detach()
        reference = field(original)
        self.reference = reference.detach()
        # Match the traced convolution arithmetic before detaching the reference.
        self.orientation = structure(reference)[0].detach()
        self.denominator = gradient_norm(self.reference).detach()
        # A private CPU generator leaves augmentation and stream RNG untouched.
        gen = torch.Generator(device='cpu').manual_seed(0)
        self.signs = (2*torch.randint(2, (*self.reference.shape[:2], 3, *self.reference.shape[-2:]), generator=gen)-1).to(original)
        self.evidence_mean = evidence.mean((-2, -1)).flatten().tolist()

    def __call__(self, logits):
        f = field(logits)
        displacement = 4*(f-self.reference)/self.denominator
        angle = 4*(structure(f)[0]-self.orientation)
        response = self.signs[:, :, 0]*displacement+(self.signs[:, :, 1:]*angle).sum(2)
        return (response*self.weight).sum((0, 2, 3))


@torch.no_grad()
def filter_update(delta, metric, acquisition, anatomy, strength=1., shared_span=False, descent_gradient=None):
    """Preserve the H-orthogonal complement; small eigensystems are 2x2 or 4x4."""
    if not math.isfinite(strength) or strength < 0:
        raise ValueError('finite nonnegative joint strength')
    if delta.ndim != 1 or metric.shape != delta.shape or acquisition.shape != (len(delta), 2) or anatomy.shape != acquisition.shape:
        raise ValueError('BN vector and two-column sensitivity matrices required')
    finite([delta, metric, acquisition, anatomy])
    if (metric <= 0).any():
        raise ValueError('positive Adam metric required')
    if descent_gradient is not None:
        if not shared_span or descent_gradient.shape != delta.shape:
            raise ValueError('descent constraint requires shared span and matching loss gradient')
        finite(descent_gradient)
        proposal_dot = float(descent_gradient.double() @ delta.double())
        if not math.isfinite(proposal_dot):
            raise ValueError('finite loss directional derivative required')
        if strength and proposal_dot >= 0:
            return delta.clone(), dict(rank=0, descent_fallback='non_descent_proposal',
                                      descent_proposal_dot=proposal_dot, descent_actual_dot=proposal_dot)
    if strength == 0:
        return delta.clone(), dict(rank=0)
    root = metric.double().sqrt()
    x, y = acquisition.double()/root[:, None], anatomy.double()/root[:, None]
    span = x
    if shared_span:
        if x.square().sum() == 0 or y.square().sum() == 0:
            return delta.clone(), dict(rank=0)
        # Normalize the two blocks so rank selection cannot erase the smaller one.
        span = torch.cat((x/x.norm(), y/y.norm()), dim=1)
    values, vectors = torch.linalg.eigh(span.T@span)
    keep = values > max(1e-12, 1e-6*float(values[-1]))
    rank = int(keep.sum())
    if not rank:
        return delta.clone(), dict(rank=0)
    q = (span@vectors[:, keep])/values[keep].sqrt()
    qx, qy = q.T@x, q.T@y
    a = qx@qx.T/x.square().sum()
    s = qy@qy.T/y.square().sum() if y.square().sum() > 0 else torch.zeros_like(a)
    eye = torch.eye(rank, device=delta.device, dtype=torch.float64)
    eigen, basis = torch.linalg.eigh(a+.01*eye)
    inverse_root = (basis*eigen.rsqrt())@basis.T
    t = inverse_root@s@inverse_root
    t = (t+t.T)/2
    k = torch.linalg.solve(eye+strength*t, eye)
    w = root*delta.double()
    result = ((w+q@(k-eye)@(q.T@w))/root).to(delta)
    descent = {}
    if descent_gradient is not None:
        va = q.T@w
        v0 = k@va
        c = q.T@(descent_gradient.double()/root)
        violation = float(c@(v0-va))
        multiplier = 0.
        if violation > 0:
            kc = k@c
            denominator = float(c@kc)
            if denominator <= 0:
                return delta.clone(), dict(rank=0, descent_fallback='degenerate_constraint',
                                          descent_proposal_dot=proposal_dot, descent_actual_dot=proposal_dot)
            multiplier = violation/denominator
            result = ((w+q@(v0-multiplier*kc-va))/root).to(delta)
        descent = dict(descent_constraint_active=violation > 0,
                       descent_unconstrained_gap=violation, descent_multiplier=multiplier,
                       descent_proposal_dot=proposal_dot,
                       descent_filtered_dot=float(descent_gradient.double()@result.double()))
    finite(result)
    diagnostic = dict(rank=rank, filter_eigenvalues=torch.linalg.eigvalsh(k).tolist(),
                      proposal_metric_norm=float(w.norm()), filtered_metric_norm=float((root*result.double()).norm()))
    if shared_span:
        diagnostic.update(acquisition_capture_fraction=float(qx.square().sum()/x.square().sum()),
                          anatomy_capture_fraction=float(qy.square().sum()/y.square().sum()),
                          proposal_span_energy_fraction=float((q.T@w).square().sum()/w.square().sum()) if w.square().sum() > 0 else 0.,
                          relative_metric_update_change=float((root*result.double()-w).norm()/w.norm()) if w.norm() > 0 else 0.)
    diagnostic.update(descent)
    return result, diagnostic


def flatten(tensors):
    return torch.cat([t.detach().reshape(-1) for t in tensors])


def descent_tolerance(proposal):
    return 1e-5*max(abs(proposal), 1e-10)


@torch.no_grad()
def commit_descent(params, before, filtered, gradient, diagnostic):
    """Audit the representable displacement; native Adam is the exact safe fallback."""
    proposal_dot = float(gradient.double()@flatten([p-old for p, old in zip(params, before)]).double())
    native = [p.detach().clone() for p in params] if diagnostic['rank'] else None
    if diagnostic['rank']:
        offset = 0
        for p, old in zip(params, before):
            p.copy_(old+filtered[offset:offset+p.numel()].view_as(p)); offset += p.numel()
    candidate_dot = float(gradient.double()@flatten([p-old for p, old in zip(params, before)]).double())
    tolerance = descent_tolerance(proposal_dot)
    repaired = native is not None and (not math.isfinite(candidate_dot) or
                                       proposal_dot < 0 and candidate_dot-proposal_dot > tolerance)
    if repaired:
        for p, proposed in zip(params, native):
            p.copy_(proposed)
        diagnostic['descent_fallback'] = 'native_adam_after_commit_violation'
    actual_dot = float(gradient.double()@flatten([p-old for p, old in zip(params, before)]).double())
    candidate_gap = candidate_dot-proposal_dot
    diagnostic.update(descent_proposal_dot=proposal_dot,
                      descent_candidate_actual_dot=candidate_dot if math.isfinite(candidate_dot) else repr(candidate_dot),
                      descent_candidate_committed_gap=candidate_gap if math.isfinite(candidate_gap) else repr(candidate_gap),
                      descent_commit_repaired=repaired,
                      descent_actual_dot=actual_dot, descent_committed_gap=actual_dot-proposal_dot,
                      descent_tolerance=tolerance)


class Host(BaseHost):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.strength = float(self.config.get('joint_strength', 0.))
        self.basis = self.config.get('joint_basis', 'acquisition')
        if self.basis not in ('acquisition', 'shared'):
            raise ValueError('registered joint basis required')
        self.descent = self.config.get('joint_descent_constraint', False)
        if not isinstance(self.descent, bool) or self.descent and self.basis != 'shared':
            raise ValueError('boolean descent constraint requires shared joint basis')
        lso = base_candidates()[3]
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError('finite nonnegative joint strength')
        if self.strength and (any(self.config.get(k) != v for k, v in lso.items() if k != 'id')
                              or self.modules != {'W': lso['params']}):
            raise ValueError('enabled joint sensitivity requires unchanged CW_LSO')

    @torch.enable_grad()
    def step(self, pixels):
        if not self.strength:
            return super().step(pixels)
        x = model_input_from_pixels(pixels, 'fundus')
        if pixels.shape != (1, 3, 512, 512) or pixels.requires_grad:
            raise ValueError('one current detached image required')
        n = self.native
        restore(n.rng)
        counts_before = n.counts.copy()
        before = [p.detach().clone() for p in n.params]
        n.last = {}; self.diag = {}; self.weak = []; self.collect = True
        native_x = x.to(self.device)
        try:
            z0 = n.model(native_x)[0]
            rotate = official().Rotate_and_Flip()
            with torch.no_grad():
                for index in range(5):
                    n.model(rotate(native_x, index))
        finally:
            self.collect = False
        predictions = torch.stack(self.weak).sigmoid()
        q = predictions.mean(0).to(z0).detach()
        sensor = Sensor(z0, predictions.to(z0))
        self.diag['structure_evidence_mean'] = sensor.evidence_mean
        scores = sensor(z0)
        b = [flatten(torch.autograd.grad(scores[c], n.params, retain_graph=c == 0)) for c in range(2)]
        del scores, z0
        bsum = (b[0]+b[1])/math.sqrt(2)
        acquisition = []
        saved = rng()
        try:
            for index in range(2):
                image = model_input_from_pixels(photo_probe(pixels, index), 'fundus').to(self.device)
                logits = n.model(image)[0]
                response = sensor(logits).sum()/math.sqrt(2)
                acquisition.append(flatten(torch.autograd.grad(response, n.params))-bsum)
                del logits, response
        finally:
            restore(saved)
        n.base.zero_grad()
        strong = official().augmentation_strong_style({'data': x.numpy().copy()})
        strong = official().normalize_image_to_0_1(torch.from_numpy(strong).float().to(self.device))
        loss = self._criterion(n.model(strong)[0], q)
        finite(loss); n.last['cal_consis_loss'] = float(loss.detach())
        loss.backward()
        loss_gradient = flatten([p.grad for p in n.params]) if self.descent else None
        n.base.step()
        delta = flatten([p-old for p, old in zip(n.params, before)])
        beta2 = n.base.param_groups[0]['betas'][1]
        eps = n.base.param_groups[0]['eps']
        metric = flatten([(n.base.state[p]['exp_avg_sq']/(1-beta2**int(n.base.state[p]['step']))).sqrt()+eps for p in n.params])
        arguments = (delta, metric, torch.stack(acquisition, 1), torch.stack(b, 1), self.strength)
        if self.basis == 'shared':
            if self.descent:
                filtered, diagnostic = filter_update(*arguments, shared_span=True, descent_gradient=loss_gradient)
            else:
                filtered, diagnostic = filter_update(*arguments, shared_span=True)
        else:
            filtered, diagnostic = filter_update(*arguments)
        with torch.no_grad():
            # Rank-zero fallback keeps the actual proposal bitwise, without subtraction/addition.
            if self.descent:
                commit_descent(n.params, before, filtered, loss_gradient, diagnostic)
            elif diagnostic['rank']:
                offset = 0
                for p, old in zip(n.params, before):
                    p.copy_(old+filtered[offset:offset+p.numel()].view_as(p)); offset += p.numel()
            logits = n.model(native_x)[0].detach()
        finite([n.base.state, n.params, self.diag])
        n.steps += 1; n.rng = rng(); self.visits += 1
        if any(int(n.base.state[p]['step']) != n.steps for p in n.params):
            raise ValueError('Adam step drift')
        if any(p._version != version or p.grad is not None for p, version in n.frozen.values()):
            raise ValueError('non-BN parameter changed')
        counts = {key: value-counts_before[key] for key, value in n.counts.items()}
        if counts != dict(forwards=10, backwards=5, base_adam=1, perturb=0, restore=0):
            raise ValueError('joint sensitivity physical counters')
        self.diag.update(joint=diagnostic, BN_update_norm=float(flatten([p-old for p, old in zip(n.params, before)]).norm()))
        native_detail = dict(counts=counts, adam_step=n.steps, lr=float(n.base.param_groups[0]['lr']), diagnostics=n.last.copy())
        return logits.float().cpu(), dict(visit=self.visits, native=[native_detail], diagnostics=copy.deepcopy(self.diag), state_committed=True)
