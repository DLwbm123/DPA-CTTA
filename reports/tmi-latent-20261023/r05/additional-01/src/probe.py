"""Fixed, source-only likelihood ranking; no neural model or adaptation claim."""
import os, json, signal, time, random, math
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion, binary_dilation

R = Path(os.environ['WORK_ROOT'])
C = json.loads((R / 'config.private.json').read_text())
COUNTS = {'source_images_read': 0, 'source_masks_read': 0, 'dev_images_read': 0, 'target_images_read': 0,
          'neural_model_calls': 0, 'network_training_updates': 0, 'synthetic_solves': 0,
          'calibration_solves': 0, 'probe_solves': 0, 'completed_probe_views': 0}
STOP = False


def stop(*_):
    global STOP
    STOP = True


signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)


def write(name, value):
    p = R / name; tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)); tmp.replace(p)


def check_stop():
    if STOP: raise InterruptedError('Budget stop; no automatic probe replay')


def grids(size):
    u, v = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    return np.stack([np.ones_like(u), u, v, u*u, u*v, v*v], -1).reshape(-1, 6), ((np.indices((size, size)).sum(0) % 2) == 0).ravel()


def gate(mask):
    distance = distance_transform_edt(~mask) - distance_transform_edt(mask)
    return 1 / (1 + np.exp(np.clip(distance, -40, 40)))


def candidates(y):
    od, oc = y
    return {'TRUE': y, 'OD_SHRINK': np.stack([binary_erosion(od, iterations=2) | oc, oc]),
            'OD_EXPAND': np.stack([binary_dilation(od, iterations=2), oc]),
            'OC_SHRINK': np.stack([od, binary_erosion(oc, iterations=2)]),
            'OC_EXPAND': np.stack([od, binary_dilation(oc, iterations=2) & od])}


def basis(q, masks=None):
    return q if masks is None else np.column_stack([q, gate(masks[0]).ravel(), gate(masks[1]).ravel()])


def fit(x, design, even, stage):
    COUNTS[stage + '_solves'] += 1
    xf = x.reshape(-1, 3)
    a, b = design[even], xf[even]
    penalty = np.eye(a.shape[1]) * C['ridge']; penalty[0, 0] = 0
    coef = np.linalg.solve(a.T @ a / len(a) + penalty, a.T @ b / len(a))
    residual = xf[~even] - design[~even] @ coef
    assert np.isfinite(residual).all()
    return residual


def score(residual, variance):
    return float(.5 * (residual**2 / variance + np.log(2 * math.pi * variance)).mean())


def severity(y, wrong, head):
    a, b = y[head], wrong[head]
    return 100 - 200 * (a & b).sum() / max(int(a.sum() + b.sum()), 1)


def selfcheck():
    n = 64; yy, xx = np.indices((n, n))
    y = np.stack([(xx-32)**2 + (yy-32)**2 < 24**2, (xx-32)**2 + (yy-32)**2 < 11**2])
    q, even = grids(n); truth = basis(q, y)
    beta = np.zeros((8, 3)); beta[0] = [.2, .2, .2]; beta[1] = [.02, -.01, .01]
    beta[6] = [.2, .15, .1]; beta[7] = [.25, .2, .3]
    x = (truth @ beta).reshape(n, n, 3)
    scores = {name: score(fit(x, basis(q, m), even, 'synthetic'), np.ones(3) * .0001) for name, m in candidates(y).items()}
    assert all(value > scores['TRUE'] for name, value in scores.items() if name != 'TRUE')
    null = score(fit(x, q, even, 'synthetic'), np.ones(3)*.0001)
    assert all(np.array_equal(basis(q, m)[:, :6], q) for m in candidates(y).values())
    assert all(not np.array_equal(basis(q, m)[:, 6:], truth[:, 6:]) for name, m in candidates(y).items() if name != 'TRUE')
    write('SELF_CHECK.json', {'synthetic_only': True, 'known_anatomy_model_ranks_truth_first': True,
                             'anatomy_free_control_candidate_independent_by_construction': True, 'counts': COUNTS.copy()})


def read(row):
    dr = Path(C['data_root']); n = C['size']
    x = np.asarray(Image.open(dr / row['image']).convert('RGB').resize((n, n), Image.Resampling.BICUBIC), dtype=np.float64)
    x = (x - x.min()) / max(float(x.max() - x.min()), 1e-6)
    z = np.asarray(Image.open(dr / row['mask']).convert('L').resize((n, n), Image.Resampling.NEAREST))
    assert set(np.unique(z)) <= {0, 128, 255}
    COUNTS['source_images_read'] += 1; COUNTS['source_masks_read'] += 1
    return x, np.stack([z < 255, z == 0])


def aggregate(rows, qualification):
    out = {'scope': 'Source-only finite anatomy likelihood ranking, no segmentation adaptation or target validation',
           'qualified_contents': sum(qualification.values()), 'planned_contents': 12, 'views': len(rows), 'heads': {}, 'negative_control_candidate_independent_by_construction': True}
    passing = len(qualification) == 12 and all(qualification.values()) and len(rows) == 24
    for head in ['OD', 'OC']:
        names = [head + '_SHRINK', head + '_EXPAND']
        content = [float(np.mean([r['scores'][name] - r['scores']['TRUE'] for r in rows if r['content'] == i for name in names])) for i in sorted(qualification)]
        by_view = {view: float(np.mean([r['scores'][name] - r['scores']['TRUE'] for r in rows if r['view'] == view for name in names])) for view in ['original','photometric']}
        win = sum(v > 1e-6 for v in content)
        okay = win >= 9 and float(np.mean(content)) > 1e-6 and all(v > 1e-6 for v in by_view.values())
        out['heads'][head] = {'mean_margin': float(np.mean(content)), 'median_margin': float(np.median(content)),
                'q05_margin': float(np.quantile(content, .05)), 'content_win_tie_loss': [win, sum(abs(v) <= 1e-6 for v in content), sum(v < -1e-6 for v in content)],
                'view_mean_margin': by_view,
                'direction_mean_margin': {name: float(np.mean([r['scores'][name] - r['scores']['TRUE'] for r in rows])) for name in names},
                'ranking_gate': bool(okay)}
        passing = passing and okay
    out['truth_rank_aggregate'] = {str(k): sum(r['truth_rank'] == k for r in rows) for k in range(1, 6)}
    out['observability_gate'] = bool(passing)
    out['decision'] = 'Retain only for a separately controlled method study; not efficacy evidence' if passing else 'Reject this quadratic-illumination plus region-mean observation; no degree/noise/threshold search'
    out['counts'] = COUNTS.copy()
    return out


def main():
    assert not (R / 'detailed.private.json').exists(), 'Refuse automatic replay'
    selfcheck()
    split = json.loads((Path(C['parent_root']) / 'split.private.json').read_text())
    assert len(split['fit']) == 79 and len(split['dev']) == 20
    probe = random.Random(C['selection_seed']).sample(range(79), C['probe_count'])
    calibration = [i for i in range(79) if i not in probe]
    write('DATA_RECEIPT.json', {'source_fit_count': 79, 'probe_contents': 12, 'calibration_contents': 67,
            'dev_reads': 0, 'target_reads': 0, 'patient_identity': 'UNKNOWN', 'neural_weights_loaded_or_copied': False})
    q, even = grids(C['size']); data = {}
    for i, row in enumerate(split['fit']):
        check_stop(); data[i] = read(row)
    sse = np.zeros(3); pixels = 0
    for i in calibration:
        check_stop(); x, y = data[i]
        residual = fit(x, basis(q, y), even, 'calibration')
        sse += (residual**2).sum(0); pixels += len(residual)
    variance = (sse / pixels).clip(min=.0001)
    write('CALIBRATION.json', {'contents': 67, 'variance_rgb': variance.tolist(), 'fixed_across_candidates': True})
    rows = []; qualification = {}
    for i in probe:
        check_stop(); original, y = data[i]; cs = candidates(y)
        qualification[i] = all(m[0].any() and m[1].any() and not (m[1] & ~m[0]).any()
                               and severity(y, m, 0 if name.startswith('OD') else 1) >= .25
                               for name, m in cs.items() if name != 'TRUE')
        for view, x in [('original', original), ('photometric', (original**1.25 * np.array([1.08,.96,.94])).clip(0,1))]:
            check_stop()
            scores = {name: score(fit(x, basis(q, m), even, 'probe'), variance) for name, m in cs.items()}
            null = score(fit(x, q, even, 'probe'), variance)
            rows.append({'content': i, 'view': view, 'qualified': qualification[i], 'scores': scores,
                         'null_score_all_candidates': null, 'truth_rank': 1 + sum(v < scores['TRUE'] - 1e-10 for k, v in scores.items() if k != 'TRUE'),
                         'counterfactual_head_dice_drop_pp': {name: severity(y, m, 0 if name.startswith('OD') else 1) for name, m in cs.items() if name != 'TRUE'}})
            COUNTS['completed_probe_views'] += 1
            write('detailed.private.json', rows)
            write('RUN_STATE.json', {'completed_contents': len(qualification), 'completed_views': len(rows), 'planned_views': 24,
                                     'counts': COUNTS.copy(), 'updated_unix': time.time()})
    report = aggregate(rows, qualification)
    write('REPORT.json', report)
    write('TRAINING_FINISHED.json', {'status':'COMPLETE', 'scope':'diagnostic only, no neural training',
            'counts': COUNTS.copy(), 'observability_gate': report['observability_gate'], 'ended_unix': time.time()})


if __name__ == '__main__':
    try: main()
    except BaseException as exc:
        write('FAILURE.json', {'type':type(exc).__name__, 'message':str(exc), 'counts':COUNTS, 'unix':time.time()})
        raise
