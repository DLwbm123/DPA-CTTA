"""Finite source-fit diagnosis. Label-assisted candidates are diagnostic only."""
import os, json, random, time, importlib.util
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
import torch
import torch.nn.functional as F

R = Path(os.environ['WORK_ROOT'])
C = json.loads((R / 'config.private.json').read_text())
P = Path(C['parent_root'])
spec = importlib.util.spec_from_file_location('parent', P / 'src/train.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False


def cosine(a, b):
    dot = sum((x * y).sum() for x, y in zip(a, b))
    na = sum(x.square().sum() for x in a).sqrt()
    nb = sum(x.square().sum() for x in b).sqrt()
    return (dot / (na * nb).clamp_min(1e-20)).item()


def direction(grads, scales):
    return [-C['content_lr'] * sc * g / g.square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(1e-8)
            for g, sc in zip(grads, scales)]


def single_task(d, y, dt, index):
    dd, yy, tt = d[:, index:index + 1], y[:, index:index + 1], dt[:, index:index + 1]
    pp = (-16 * dd).sigmoid()
    di = (2 * (pp * yy).sum() + 1) / (pp.sum() + yy.sum() + 1)
    sdf = (F.smooth_l1_loss(dd, tt, reduction='none') * (1 + 4 * torch.exp(-8 * tt.abs()))).mean()
    return F.binary_cross_entropy_with_logits(-16 * dd, yy) + 1 - di + .2 * sdf


def main():
    split = json.loads((P / 'split.private.json').read_text())
    assert len(split['fit']) == 79 and len(split['dev']) == 20
    dr = Path(C['data_root']); fit = []
    for row in split['fit']:
        x = np.asarray(Image.open(dr / row['image']).convert('RGB').resize((256, 256), Image.Resampling.BICUBIC), dtype=np.float32)
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
        z = np.asarray(Image.open(dr / row['mask']).convert('L').resize((256, 256), Image.Resampling.NEAREST))
        y = np.stack([z < 255, z == 0]).astype(np.float32)
        dt = np.stack([(distance_transform_edt(1 - q) - distance_transform_edt(q)).clip(-16, 16) / 16 for q in y]).astype(np.float32)
        fit.append((torch.tensor(x.transpose(2, 0, 1).copy()), torch.tensor(y), torch.tensor(dt)))
    ids = random.Random(C['selection_seed']).sample(range(79), C['fit_probes'])
    m.write(R / 'SOURCE_RECEIPT.json', {'fit_only': 79, 'probed_fit_contents': len(ids), 'dev_reads': 0,
            'target_reads': 0, 'training_updates': 0, 'checkpoint_copies': 0})
    allrows = []
    for seed in C['seeds']:
        torch.manual_seed(seed)
        model = m.Model().cuda().eval()
        ck = torch.load(P / f'runs/{seed}/latest.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model'], strict=True); del ck
        for parameter in model.parameters(): parameter.requires_grad_(False)
        prior = m.calibrate(model, fit)
        for ordinal, i in enumerate(ids):
            if m.STOP: raise InterruptedError('Diagnostic budget stop')
            xx, yy, dt = fit[i]; x = xx[None].cuda(); y = yy[None].cuda(); target_distance = dt[None].cuda()
            torch.manual_seed(seed * 1000 + i)
            xb, _ = m.paired(x)
            with torch.no_grad():
                ca, sa = model.encode(x); cb, sb = model.encode(xb)
                render = model.render(cb, sb, (256, 256))
                swapped = model.render(cb, sa, (256, 256))
                cross = model.render(ca, sb, (256, 256))
                d0 = model.head(cb, (256, 256))
                zero = m.metrics(d0, yy)
            c = [v.detach().requires_grad_() for v in cb]
            scales = [v.detach().square().mean((1, 2, 3), keepdim=True).sqrt().clamp_min(.1) for v in c]
            j = m.objective(model, xb, c, sb, [v.detach() for v in c], scales, prior)
            gj = torch.autograd.grad(j, c)
            dj = direction(gj, scales)
            d = model.head(c, (256, 256))
            losses = {'joint': m.segloss(d, y, target_distance),
                      'od': single_task(d, y, target_distance, 0),
                      'oc': single_task(d, y, target_distance, 1)}
            gradients = {name: torch.autograd.grad(value, c, retain_graph=True) for name, value in losses.items()}
            alignment = {name: {'cosine': cosine(gj, gm),
                         'preconditioned_directional_derivative': sum((g * v).sum().item() for g, v in zip(gm, dj))}
                         for name, gm in gradients.items()}
            oracle_delta = direction(gradients['joint'], scales)
            with torch.no_grad():
                oracle_c = [torch.maximum(torch.minimum(v.detach() + dv, v.detach() + C['content_radius'] * sc),
                                          v.detach() - C['content_radius'] * sc)
                            for v, dv, sc in zip(c, oracle_delta, scales)]
                oracle = m.metrics(model.head(oracle_c, (256, 256)), yy)
                oracle_task_change = (m.segloss(model.head(oracle_c, (256, 256)), y, target_distance) - losses['joint']).item()
            adapted_c, adapted_s, trace = m.infer(model, xb, cb, sb, prior.copy().anchor())
            with torch.no_grad(): adapted = m.metrics(model.head(adapted_c, (256, 256)), yy)
            image_change = (xb - x).abs().mean().item()
            rec = {'seed': seed, 'fit_index': i, 'ordinal': ordinal, 'alignment': alignment,
                   'zero': zero, 'adapted': adapted, 'source_label_candidate_not_deployable': oracle,
                   'source_label_candidate_task_delta': oracle_task_change, 'trace': trace,
                   'style_renderer_change_ratio': (render - swapped).abs().mean().item() / max(image_change, 1e-8),
                   'style_swap_observation_gain': m.obs(xb, swapped).item() - m.obs(xb, render).item(),
                   'cross_observation_error': m.obs(xb, cross).item(), 'observation_error': m.obs(xb, render).item(),
                   'content_difference': sum((a - b).square().mean().item() for a, b in zip(ca, cb)) / 3}
            allrows.append(rec)
            m.write(R / 'detailed.private.json', allrows)
            assert all(not p.requires_grad for p in model.parameters())
            m.write(R / 'RUN_STATE.json', {'seed': seed, 'completed_probes': len(allrows), 'planned_probes': 24,
                    'counts': m.COUNTS.copy(), 'updated_unix': time.time()})
        del model; torch.cuda.empty_cache()
    m.write(R / 'detailed.private.json', allrows)
    report = {'scope': 'Frozen source-fit diagnosis, no development or target images; label-assisted candidate is not a deployable method',
              'seeds': [], 'counts': m.COUNTS.copy()}
    for seed in C['seeds']:
        rows = [r for r in allrows if r['seed'] == seed]
        item = {'seed': seed, 'probes': len(rows), 'alignment': {}}
        for task in ['joint', 'od', 'oc']:
            item['alignment'][task] = {'mean_cosine': float(np.mean([r['alignment'][task]['cosine'] for r in rows])),
                    'non_descent_fraction': float(np.mean([r['alignment'][task]['preconditioned_directional_derivative'] >= 0 for r in rows]))}
        for name in ['style_renderer_change_ratio', 'style_swap_observation_gain', 'cross_observation_error', 'observation_error', 'content_difference']:
            item[name] = {'mean': float(np.mean([r[name] for r in rows])), 'median': float(np.median([r[name] for r in rows]))}
        item['actual_infer_dice_delta'] = float(np.mean([r['adapted']['dice'] - r['zero']['dice'] for r in rows]))
        item['source_label_candidate_dice_delta'] = float(np.mean([r['source_label_candidate_not_deployable']['dice'] - r['zero']['dice'] for r in rows]))
        item['source_label_candidate_loss_delta'] = float(np.mean([r['source_label_candidate_task_delta'] for r in rows]))
        report['seeds'].append(item)
    assert m.COUNTS['outer_updates'] == 0 and m.COUNTS['target_domain_images'] == 0
    m.write(R / 'REPORT.json', report)
    m.write(R / 'TRAINING_FINISHED.json', {'status': 'COMPLETE', 'scope': 'diagnostic only, no training',
            'completed_probes': len(allrows), 'counts': m.COUNTS, 'ended_unix': time.time()})


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        m.write(R / 'FAILURE.json', {'type': type(exc).__name__, 'message': str(exc), 'counts': m.COUNTS, 'unix': time.time()})
        raise
