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


def load_data():
    dr = Path(CFG['data_root'])
    rows = list(csv.DictReader((dr / 'RIM_ONE_r3_train.csv').open()))
    assert len(rows) == 99
    def group(r): return Path(r['image']).stem.rsplit('-', 1)[0]
    groups = sorted(set(map(group, rows))); random.Random(10223).shuffle(groups)
    devgroups = set(groups[:round(.2 * len(groups))])
    split = {'fit': [r for r in rows if group(r) not in devgroups], 'dev': [r for r in rows if group(r) in devgroups]}
    assert len(split['fit']) == 79 and len(split['dev']) == 20
    assert not set(map(group, split['fit'])) & set(map(group, split['dev']))
    write(ROOT / 'split.private.json', split)
    def read(r):
        size = CFG['size']
        x = np.asarray(Image.open(dr / r['image']).convert('RGB').resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32)
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
        mask = np.asarray(Image.open(dr / r['mask']).convert('L').resize((size, size), Image.Resampling.NEAREST))
        assert set(np.unique(mask)) <= {0, 128, 255}, 'Unexpected mask convention'
        y = np.stack([mask < 255, mask == 0]).astype(np.float32)
        dt = np.stack([(distance_transform_edt(1 - z) - distance_transform_edt(z)).clip(-16, 16) / 16 for z in y]).astype(np.float32)
        return torch.tensor(x.transpose(2, 0, 1).copy()), torch.tensor(y), torch.tensor(dt)
    return [read(r) for r in split['fit']], [read(r) for r in split['dev']]


def batch(data, rng, n):
    ix = rng.sample(range(len(data)), n)
    tensors = [torch.stack([data[i][j] for i in ix]).cuda() for j in range(3)]
    for dim in [-1, -2]:
        if rng.random() < .5: tensors = [v.flip(dim) for v in tensors]
    return tensors


@torch.no_grad()
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


def corrupt(x, kind):
    if kind == 'clean': return x
    if kind == 'gamma': return x.clamp_min(1e-6).pow(1.52)
    if kind == 'color': return (x * x.new_tensor([1.2, .88, .84])[None, :, None, None]).clamp(0, 1)
    if kind == 'blur': return F.avg_pool2d(x, 5, 1, 2)
    raise ValueError(kind)


def metrics(d, y):
    pred = (d.detach().cpu().numpy()[0] < 0)
    gt = y.numpy().astype(bool)
    dice = [float(200 * (a & b).sum() / max(int(a.sum() + b.sum()), 1)) for a, b in zip(pred, gt)]
    out = {'dice': float(np.mean(dice)), 'od': dice[0], 'oc': dice[1], 'empty_od': not bool(pred[0].any()),
           'empty_oc': not bool(pred[1].any()), 'cup_outside': float((pred[1] & ~pred[0]).sum() / max(int(pred[1].sum()), 1))}
    def ratio(m):
        def height(a):
            rr = np.where(a.any(1))[0]
            return int(rr[-1] - rr[0] + 1) if len(rr) else 0
        return height(m[1]) / max(height(m[0]), 1)
    out['vcdr_error'] = abs(ratio(pred) - ratio(gt))
    for name, a, b in zip(['od', 'oc'], pred, gt):
        aa, bb = np.argwhere(a ^ binary_erosion(a)), np.argwhere(b ^ binary_erosion(b))
        if len(aa) and len(bb):
            dist = np.concatenate([cKDTree(aa).query(bb)[0], cKDTree(bb).query(aa)[0]])
            out[name + '_hd95'] = float(np.quantile(dist, .95)); out[name + '_assd'] = float(dist.mean())
        else:
            out[name + '_hd95'] = None; out[name + '_assd'] = None
    return out


def evaluate(model, dev, prior, out, adapt=True):
    model.eval()
    for p in model.parameters(): p.requires_grad_(False)
    rows = []
    ids = list(range(len(dev))); random.Random(8821).shuffle(ids)
    persistent = prior.copy().anchor()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    for kind in ['clean', 'gamma', 'color', 'blur']:
        for i in ids:
            if STOP: raise InterruptedError('Budget stop during readout')
            xx, yy, _ = dev[i]; x = corrupt(xx[None].cuda(), kind)
            with torch.no_grad():
                c0, s0 = model.encode(x)
                z0 = model.head(c0, x.shape[-2:])
            zero = metrics(z0, yy)
            rec = {'content': i, 'condition': kind, 'ZERO': zero}
            if adapt:
                for arm, current in [('INDEPENDENT', prior.copy().anchor()), ('CONTINUAL', persistent)]:
                    with torch.enable_grad():
                        c, s, trace = infer(model, x, c0, s0, current)
                    with torch.no_grad():
                        dd = model.head(c, x.shape[-2:])
                        mm = metrics(dd, yy)
                        if arm == 'CONTINUAL' and any(t['accepted'] for t in trace): current.update(s.detach())
                    rec[arm] = mm; rec[arm + '_trace'] = trace
            rows.append(rec)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items()), 'Evaluation changed model weights'
    write(out / 'detailed.private.json', rows)
    summary = {}
    for arm in ['ZERO', 'INDEPENDENT', 'CONTINUAL'] if adapt else ['ZERO']:
        summary[arm] = {k: float(np.mean([r[arm][k] for r in rows])) for k in ['dice', 'od', 'oc', 'vcdr_error']}
    summary['coverage'] = len(rows)
    summary['network_unchanged'] = True
    summary['prior_final'] = persistent.receipt()
    write(out / 'metrics.json', summary)
    for p in model.parameters(): p.requires_grad_(True)
    return summary


def selfcheck(model):
    model.train()
    x = torch.rand(1, 3, 64, 64, device='cuda')
    c, s = model.encode(x)
    assert [list(v.shape[1:]) for v in c] == [[16, 16, 16], [32, 8, 8], [64, 4, 4]]
    y = (torch.rand(1, 2, 64, 64, device='cuda') > .5).float()
    d = torch.zeros_like(y)
    p = Prior().anchor()
    state = [v.detach().clone() for v in model.parameters()]
    cc, ss, trace = infer(model, x, c, s, p, True)
    loss = segloss(model.head(cc, (64, 64)), y, d)
    loss.backward()
    assert model.rgb[-1].weight.grad is not None and model.rgb[-1].weight.grad.isfinite().all(), 'No executed-update meta gradient'
    assert all(torch.equal(a, b.detach()) for a, b in zip(state, model.parameters()))
    model.zero_grad(set_to_none=True)
    c = [v.detach() for v in c]; s = s.detach()
    cc, ss, trace = infer(model, x, c, s, p)
    assert all(not t['accepted'] or t['after'] <= t['before'] + 1e-7 for t in trace)
    p.update(ss.detach())
    assert torch.isfinite(p.means).all() and p.var.min() > 0
    return {'shape': True, 'second_order_gradient': True, 'no_parameter_mutation': True, 'labels_absent_from_infer': True,
            'prior_finite': True, 'synthetic_only': True, 'counts': COUNTS.copy()}


def checkpoint(model, opt, path, phase, step):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'phase': phase, 'step': step}, tmp)
    tmp.replace(path)


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(771)
    model = Model().cuda()
    write(ROOT / 'SELF_CHECK.json', selfcheck(model))
    del model
    fit, dev = load_data()
    write(ROOT / 'DATA_RECEIPT.json', {'fit': len(fit), 'dev': len(dev), 'resolution': CFG['size'], 'target_reads': 0,
          'medical_initialization': 'random, no historical medical checkpoint', 'patient_identity': 'UNKNOWN',
          'development_reuse': True, 'counts_before_training': COUNTS.copy()})
    allseeds = []
    for seed in CFG['seeds']:
        if STOP: break
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model = Model().cuda()
        rng = random.Random(seed)
        prior = Prior().anchor()
        run = ROOT / 'runs' / str(seed)
        run.mkdir(parents=True, exist_ok=True)
        write(run / 'MODEL.json', {'parameters': sum(p.numel() for p in model.parameters()), 'seed': seed})
        for phase, steps, lr, bs in [('warm', CFG['warm_steps'], CFG['lr'], CFG['batch']), ('meta', CFG['meta_steps'], CFG['meta_lr'], CFG['meta_batch'])]:
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.0001)
            for step in range(1, steps + 1):
                if STOP: break
                x, y, dt = batch(fit, rng, bs)
                xb, params = paired(x)
                a, sa = model.encode(x); b, sb = model.encode(xb)
                da = model.head(a, x.shape[-2:]); db = model.head(b, x.shape[-2:])
                med = (segloss(da, y, dt) + segloss(db, y, dt)) / 2
                reconstruction = (obs(x, model.render(a, sa, x.shape[-2:])) + obs(xb, model.render(b, sb, x.shape[-2:]))) / 2
                crossed = (obs(xb, model.render(a, sb, x.shape[-2:])) + obs(x, model.render(b, sa, x.shape[-2:]))) / 2
                invariant = sum(F.smooth_l1_loss(aa, bb) for aa, bb in zip(a, b)) / 3
                relative = F.mse_loss(model.relative_style(sb - sa), params)
                loss = med + .5 * reconstruction + .25 * crossed + .05 * invariant + .05 * relative
                outer = None
                clean_harm = None
                if phase == 'meta':
                    cc, ss, trace = infer(model, xb, b, sb, prior, True)
                    outer = segloss(model.head(cc, x.shape[-2:]), y, dt)
                    clean_c, clean_s, clean_trace = infer(model, x, a, sa, prior, True)
                    clean_outer = segloss(model.head(clean_c, x.shape[-2:]), y, dt)
                    clean_harm = F.relu(clean_outer - segloss(da, y, dt).detach())
                    loss = loss + .5 * outer + .1 * clean_harm
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                opt.step(); COUNTS['outer_updates'] += 1
                if phase == 'meta' and any(t['accepted'] for t in trace):
                    prior.update(ss[:1].detach())
                if step == 1 or step % 20 == 0:
                    value = {'seed': seed, 'phase': phase, 'step': step, 'loss': loss.item(), 'medical': med.item(),
                             'reconstruction': reconstruction.item(), 'grad_norm': gn.item(), 'outer': None if outer is None else outer.item(),
                             'clean_harm': None if clean_harm is None else clean_harm.item()}
                    write(ROOT / 'RUN_STATE.json', {**value, 'counts': COUNTS.copy(), 'updated_unix': time.time()})
                    log(value)
                if step % 500 == 0:
                    checkpoint(model, opt, run / 'latest.pt', phase, step)
            checkpoint(model, opt, run / 'latest.pt', phase, step)
            if STOP: break
            model.eval(); prior = calibrate(model, fit)
            write(run / (phase + '_source_prior.json'), prior.receipt())
            if phase == 'warm':
                warm = evaluate(model, dev, prior, run / 'warm_readout', False)
                log({'seed': seed, 'warm_zero': warm['ZERO']})
        if STOP:
            write(run / 'completion.json', {'status': 'STOPPED', 'phase': phase, 'step': step})
            break
        model.eval(); prior = calibrate(model, fit)
        zero = evaluate(model, dev, prior, run / 'final_qualification', False)
        qualified = zero['ZERO']['dice'] >= CFG['gate_min_source_dice']
        if qualified:
            result = evaluate(model, dev, prior, run / 'final_readout', True)
        else:
            result = zero
        record = {'seed': seed, 'status': 'COMPLETE' if qualified else 'SOURCE_NOT_QUALIFIED', 'steps': CFG['warm_steps'] + CFG['meta_steps'], 'metrics': result}
        write(run / 'completion.json', record)
        allseeds.append(record)
        log({'seed_finished': record})
        del model, opt
        torch.cuda.empty_cache()
    write(ROOT / 'TRAINING_FINISHED.json', {'status': 'STOPPED' if STOP else 'COMPLETE', 'seeds': allseeds,
          'counts': COUNTS, 'peak_cuda_allocated': torch.cuda.max_memory_allocated(), 'ended_unix': time.time()})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        write(ROOT / 'FAILURE.json', {'type': type(exc).__name__, 'message': str(exc), 'unix': time.time(), 'counts': COUNTS})
        raise
