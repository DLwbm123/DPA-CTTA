"""Source-trained positive latent metric; frozen generative segmentation network."""
import os, json, time, random, math, importlib.util
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

R = Path(os.environ['WORK_ROOT'])
C = json.loads((R / 'config.private.json').read_text())
P = Path(C['parent_root'])
spec = importlib.util.spec_from_file_location('base', P / 'src/train.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class Metric(nn.Module):
    def __init__(self, coupled=True):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(3, 16, 16))
        self.coupled = coupled

    def matrix(self, index):
        raw = self.raw[index]
        if not self.coupled: raw = torch.diag_embed(raw.diagonal())
        b = torch.eye(16, device=raw.device) + .5 * raw.tanh()
        bound = (b.abs().sum(0).max() * b.abs().sum(1).max()).sqrt()
        b = b / (bound / 3).clamp_min(1)
        return .1 * torch.eye(16, device=raw.device) + .9 * b.T @ b

    def forward(self, g, index):
        b, c, h, w = g.shape
        q = g.reshape(b, c // 16, 16, h, w)
        return torch.einsum('ij,bgjhw->bgihw', self.matrix(index), q).reshape_as(g)


def infer(model, metric, x, c0, s0, prior, training=False):
    """No labels. Three proposals; positive metric, original J and acceptance."""
    c = [v.detach().requires_grad_() for v in c0]
    s = s0.detach().requires_grad_()
    refs = [v.detach() for v in c0]
    scales = [v.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(.1) for v in refs]
    trace = []
    for step in range(3):
        params = [s] if step == 0 else c + [s]
        value = m.objective(model, x, c, s, refs, scales, prior)
        grads = torch.autograd.grad(value, params, create_graph=training)
        m.COUNTS['latent_gradient_calls'] += 1
        if not all(torch.isfinite(g).all() for g in grads): raise FloatingPointError('Latent gradient')
        deltas = []
        for j, g in enumerate(grads):
            rms = g.detach().square().mean(tuple(range(1, g.ndim)), keepdim=True).sqrt().clamp_min(1e-8)
            factor = C['style_lr'] if j == len(grads) - 1 else C['content_lr'] * scales[j]
            direction = metric(g, j) if step and j < 3 and metric is not None else g
            deltas.append(factor * direction / rms)
        accepted = False
        for shrink in [1., .5]:
            candidate = [p - shrink * dv for p, dv in zip(params, deltas)]
            cc = c if step == 0 else [torch.maximum(torch.minimum(v, r + C['content_radius'] * sc), r - C['content_radius'] * sc)
                                     for v, r, sc in zip(candidate[:3], refs, scales)]
            ss = candidate[0] if step == 0 else candidate[3]
            with torch.no_grad(): after = m.objective(model, x, cc, ss, refs, scales, prior).item()
            if math.isfinite(after) and after <= value.detach().item() + 1e-7:
                c, s = cc, ss; accepted = True; break
        trace.append({'step': step, 'before': value.detach().item(), 'after': after, 'accepted': accepted})
        if not accepted: break
        if not training:
            c = [v.detach().requires_grad_() for v in c]; s = s.detach().requires_grad_()
    return c, s, trace


def selfcheck():
    torch.manual_seed(771)
    model = m.Model().cuda().eval()
    for p in model.parameters(): p.requires_grad_(False)
    metric = Metric().cuda()
    with torch.no_grad():
        metric.raw[0, 0, 1] = .2
        p = metric.matrix(0)
        assert (p[0, 1].abs() > 0) and torch.linalg.eigvalsh(p).min() >= .09999
        assert torch.linalg.eigvalsh(p).max() <= 8.20001
        metric.raw.zero_()
        diagonal = Metric(False).cuda()
        assert all(torch.equal(metric.matrix(i), diagonal.matrix(i)) for i in range(3))
    x = torch.rand(1, 3, 64, 64, device='cuda')
    with torch.no_grad(): c, s = model.encode(x)
    prior = m.Prior().anchor()
    a, sa, ta = m.infer(model, x, c, s, prior)
    b, sb, tb = infer(model, metric, x, c, s, prior)
    delta = [float((u - v).abs().max()) for u, v in zip(a + [sa], b + [sb])]
    m.write(R / 'IDENTITY_CHECK.json', {'max_abs_differences': delta, 'atol': 3e-6, 'rtol': 3e-5,
            'reason': 'CUDA bilinear interpolation backward can be nondeterministic; identity is numerical, not bitwise'})
    assert all(torch.allclose(u, v, atol=3e-6, rtol=3e-5) for u, v in zip(a + [sa], b + [sb])), 'Identity metric differs'
    bb, _, trace = infer(model, metric, x, c, s, prior, True)
    y = torch.zeros(1, 2, 64, 64, device='cuda'); dt = torch.ones_like(y)
    loss = m.segloss(model.head(bb, (64, 64)), y, dt)
    loss.backward()
    grads = [p.grad for p in metric.parameters() if p.grad is not None]
    assert grads and all(g.isfinite().all() for g in grads) and sum(g.abs().sum().item() for g in grads) > 0
    assert all(p.grad is None for p in model.parameters())
    assert all(not t['accepted'] or t['after'] <= t['before'] + 1e-7 for t in trace)
    m.write(R / 'SELF_CHECK.json', {'synthetic_only': True, 'identity_matches_original': True,
            'executed_path_metric_gradient_nonzero': True, 'frozen_network_no_grad': True,
            'positive_metric_eigenvalue_bounds': [.1, 8.2], 'cross_channel_coupling_nonzero': True, 'counts': m.COUNTS.copy()})


def save(metric, opt, rng, seed, step, arm):
    path = R / 'runs' / str(seed) / (arm + '_metric.pt'); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'metric': metric.state_dict(), 'optimizer': opt.state_dict(), 'completed_episode': step,
                'rng': rng.getstate(), 'cuda_rng': torch.cuda.get_rng_state(), 'torch_rng': torch.get_rng_state(),
                'counts': m.COUNTS.copy()}, path.with_suffix('.tmp'))
    path.with_suffix('.tmp').replace(path)


def evaluate(model, metrics, dev, prior, run):
    rows = []; ids = list(range(len(dev))); random.Random(8821).shuffle(ids)
    for kind in ['clean', 'gamma', 'color', 'blur']:
        for i in ids:
            if m.STOP: raise InterruptedError('Budget stop during final readout')
            xx, yy, _ = dev[i]; x = m.corrupt(xx[None].cuda(), kind)
            with torch.no_grad(): c0, s0 = model.encode(x); d0 = model.head(c0, (256, 256))
            rec = {'content': i, 'condition': kind, 'ZERO': m.metrics(d0, yy)}
            for arm, current in [('FIXED3', None), *metrics.items()]:
                c, s, trace = infer(model, current, x, c0, s0, prior)
                with torch.no_grad(): d = model.head(c, (256, 256))
                rec[arm] = m.metrics(d, yy); rec[arm + '_trace'] = trace
            rows.append(rec)
    m.write(run / 'detailed.private.json', rows)
    out = {arm: {k: float(np.mean([r[arm][k] for r in rows])) for k in ['dice', 'od', 'oc', 'vcdr_error']}
           for arm in ['ZERO', 'FIXED3', 'DIAGONAL3', 'COUPLED3']}
    out['coverage'] = len(rows); m.write(run / 'metrics.json', out)
    return out


def main():
    torch.set_num_threads(4); torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert not (R / "runs").exists(), "No automatic replay of partially completed run"
    selfcheck()
    # The loader fixes the same split; dev is held in memory but never used during fitting.
    fit, dev = m.load_data()
    m.write(R / 'DATA_RECEIPT.json', {'fit': len(fit), 'dev_final_only': len(dev), 'target_reads': 0,
            'patient_identity': 'UNKNOWN', 'dev_reused': True, 'base_checkpoint_copied': False})
    records = []
    for seed in C['seeds']:
        torch.manual_seed(seed); rng = random.Random(seed + 30000)
        model = m.Model().cuda().eval()
        ck = torch.load(P / f'runs/{seed}/latest.pt', map_location='cpu', weights_only=False)
        assert ck['phase'] == 'meta' and ck['step'] == 500
        model.load_state_dict(ck['model'], strict=True); del ck
        for p in model.parameters(): p.requires_grad_(False)
        before = [p.detach().clone() for p in model.parameters()]
        prior = m.calibrate(model, fit)
        metrics = {'DIAGONAL3': Metric(False).cuda(), 'COUPLED3': Metric(True).cuda()}
        opts = {arm: torch.optim.Adam(mm.parameters(), lr=C['metric_lr']) for arm, mm in metrics.items()}
        run = R / 'runs' / str(seed); run.mkdir(parents=True, exist_ok=True)
        completed = 0; updates = {arm: 0 for arm in metrics}; skipped = {arm: 0 for arm in metrics}
        for step in range(1, C['episodes'] + 1):
            if m.STOP: break
            x, y, dt = m.batch(fit, rng, C['metric_batch']); xb, _ = m.paired(x)
            with torch.no_grad():
                ca, sa = model.encode(x); cb, sb = model.encode(xb)
                clean_zero = m.segloss(model.head(ca, (256, 256)), y, dt)
            losses = {}
            for arm, metric in metrics.items():
                cc, _, _ = infer(model, metric, xb, cb, sb, prior, True)
                clean_c, _, _ = infer(model, metric, x, ca, sa, prior, True)
                shifted = m.segloss(model.head(cc, (256, 256)), y, dt)
                clean = m.segloss(model.head(clean_c, (256, 256)), y, dt)
                loss = shifted + .1 * F.relu(clean - clean_zero)
                if not loss.isfinite(): raise FloatingPointError('Training loss')
                opts[arm].zero_grad(set_to_none=True)
                if loss.requires_grad: loss.backward()
                active = [p for p in metric.parameters() if p.grad is not None]
                if active:
                    torch.nn.utils.clip_grad_norm_(active, 5., error_if_nonfinite=True); opts[arm].step()
                    updates[arm] += 1; m.COUNTS['outer_updates'] += 1
                else: skipped[arm] += 1
                losses[arm] = loss.item()
                del cc, clean_c, shifted, clean, loss
            completed = step
            if step == 1 or step % 20 == 0:
                state = {'seed': seed, 'completed_episode': step, 'optimizer_updates': updates.copy(),
                         'no_metric_dependency': skipped.copy(), 'losses': losses,
                         'counts': m.COUNTS.copy(), 'updated_unix': time.time()}
                m.write(R / 'RUN_STATE.json', state); m.log(state)
            if step % 100 == 0:
                for arm, metric in metrics.items(): save(metric, opts[arm], rng, seed, completed, arm)
        for arm, metric in metrics.items(): save(metric, opts[arm], rng, seed, completed, arm)
        assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before)), 'Base network changed'
        if m.STOP: break
        for mm in metrics.values():
            mm.eval()
            for p in mm.parameters(): p.requires_grad_(False)
        metric_before = {arm: [p.detach().clone() for p in mm.parameters()] for arm, mm in metrics.items()}
        result = evaluate(model, metrics, dev, prior, run)
        assert all(torch.equal(p, b) for p, b in zip(model.parameters(), before)), 'Readout changed base'
        assert all(torch.equal(p, b) for arm, mm in metrics.items() for p, b in zip(mm.parameters(), metric_before[arm])), 'Readout changed metric'
        matrices = {arm: [mm.matrix(i).detach().cpu().tolist() for i in range(3)] for arm, mm in metrics.items()}
        m.write(run / 'matrices.private.json', matrices)
        receipt = {arm: [{'eig_min': torch.linalg.eigvalsh(mm.matrix(i)).min().item(),
                          'eig_max': torch.linalg.eigvalsh(mm.matrix(i)).max().item(),
                          'off_diagonal_norm': (mm.matrix(i) - torch.diag_embed(mm.matrix(i).diagonal())).norm().item()} for i in range(3)]
                   for arm, mm in metrics.items()}
        record = {'seed': seed, 'episodes_per_arm': completed, 'optimizer_updates': updates, 'skipped': skipped,
                  'frozen_base_unchanged': True, 'metric_unchanged_during_readout': True,
                  'allocated_metric_parameters': 768, 'unmasked_raw_parameters': {'DIAGONAL3': 48, 'COUPLED3': 768},
                  'matrix_receipt': receipt, 'metrics': result}
        m.write(run / 'completion.json', record); records.append(record)
        del model, metrics, opts, metric_before, before; torch.cuda.empty_cache()
    m.write(R / 'TRAINING_FINISHED.json', {'status': 'STOPPED' if m.STOP else 'COMPLETE', 'seeds': records,
            'counts': m.COUNTS.copy(), 'ended_unix': time.time(), 'peak_cuda_allocated': torch.cuda.max_memory_allocated()})


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        m.write(R / 'FAILURE.json', {'type': type(exc).__name__, 'message': str(exc), 'counts': m.COUNTS,
                                  'unix': time.time()})
        raise
