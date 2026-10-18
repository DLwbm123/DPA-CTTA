"""CPU-only final labels, after successful generation and worker retirement."""
from pathlib import Path
import csv
import time

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

from .run import ARMS, MODES, SEED, read, save


def metric(pred, gt, base):
    values = []
    for p, y, b in zip(pred, gt, base):
        n = p.sum()+y.sum()
        ps = p ^ binary_erosion(p, border_value=0)
        ys = y ^ binary_erosion(y, border_value=0)
        assd = hd95 = None
        if p.any() and y.any():
            distances = np.concatenate((distance_transform_edt(~ys)[ps], distance_transform_edt(~ps)[ys]))
            assd = float(distances.mean());hd95 = float(np.percentile(distances, 95))
        boundary = distance_transform_edt(~ys) <= 3 if ys.any() else np.zeros_like(y)
        corrected = (b != y) & (p == y)
        spoiled = (b == y) & (p != y)
        regions = {}
        for name, region in [('all', np.ones_like(y)), ('foreground', y), ('boundary3px', boundary)]:
            denominator = int(region.sum())
            regions[name] = dict(denominator=denominator,
                 corrected_fraction=float((corrected & region).sum()/denominator) if denominator else None,
                 spoiled_fraction=float((spoiled & region).sum()/denominator) if denominator else None)
        values.append(dict(dice_percent=float(200*(p & y).sum()/n) if n else 100.,
                           assd_pixel=assd, hd95_pixel=hd95,
                           pred_empty=not bool(p.any()), gt_empty=not bool(y.any()), regions=regions))
    def ratio(mask):
        heights = []
        for m in mask:
            yy = np.nonzero(m)[0]
            heights.append(int(yy.max()-yy.min()+1) if len(yy) else 0)
        return heights[1]/heights[0] if heights[0] else None
    p_ratio, y_ratio = ratio(pred), ratio(gt)
    area_ratio = float(pred[1].sum()/pred[0].sum()) if pred[0].any() else None
    return values, dict(predicted_vcdr=p_ratio, true_vcdr=y_ratio,
                        vcdr_abs_error=abs(p_ratio-y_ratio) if p_ratio is not None and y_ratio is not None else None,
                        predicted_area_ratio=area_ratio)


def interval(delta):
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, 8, (2000, 8))
    bootstrap = delta[indices].mean(tuple(range(1, delta.ndim+1)))
    return np.quantile(bootstrap, [.025, .975]).tolist()


def csvout(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def score(c):
    from ..source_io import read_mask
    root = Path(c['output_root']);started = time.time()
    state = read(root/'RUN_STATE.json');retired = read(root/'processes/worker.json')
    if __import__('os').environ.get('CUDA_VISIBLE_DEVICES') != '' or retired['active'] or retired['exit_code'] != 0:
        raise ValueError('successful retired GPU worker and CPU-only scorer required')
    if state['status'] != 'GPU_COMPLETE' or state['predictions'] != 736 or state['training_episodes'] != 1000:
        raise ValueError('fixed matrix not complete')
    records = read(root/'private/predictions.json');panel = read(root/'scorer/panel.json')
    if len(records) != 736 or len(panel) != 8:
        raise ValueError('final scorer coverage')
    keys = [(r['pair'], r['condition'], r['arm'], r['mode']) for r in records]
    if len(set(keys)) != 736 or [r['index'] for r in records] != list(range(736)):
        raise ValueError('prediction identity coverage')
    expected = {(p, k, a, m) for p in range(8) for k in range(4) for a in ARMS
                for m in (('DIRECT',) if a in ('B0', 'D') else MODES)}
    if set(keys) != expected:
        raise ValueError('wrong fixed methods/readouts')
    masks = {}
    for r, key in zip(records, keys):
        raw = (root/'predictions'/f'{r["index"]:03d}.bits').read_bytes()
        if len(raw) != 65536:
            raise ValueError('prediction size mismatch')
        masks[key] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(2, 512, 512).astype(bool)
    labels = [read_mask(r['mask_path'], 'fundus', r['image_size']).numpy()[0].astype(bool) for r in panel]
    dice = np.zeros((8, 4, len(ARMS), 3, 2));surface = np.full_like(dice, np.nan);hd95 = surface.copy()
    empty = np.zeros_like(dice, dtype=bool);ratios = np.full(dice.shape[:-1], np.nan)
    rows = []
    for r, key in zip(records, keys):
        if time.time()-started > c['cpu_cap_seconds']-10:
            raise TimeoutError('CPU scorer cap')
        p, k, a, mode = key
        mm, rr = metric(masks[key], labels[p], masks[p, k, 'B0', 'DIRECT'])
        ai = ARMS.index(a)
        mi = range(3) if mode == 'DIRECT' else [MODES.index(mode)]
        for m in mi:
            for channel in range(2):
                value = mm[channel]
                dice[p, k, ai, m, channel] = value['dice_percent']
                surface[p, k, ai, m, channel] = value['assd_pixel'] if value['assd_pixel'] is not None else np.nan
                hd95[p, k, ai, m, channel] = value['hd95_pixel'] if value['hd95_pixel'] is not None else np.nan
                empty[p, k, ai, m, channel] = value['pred_empty']
            ratios[p, k, ai, m] = rr['vcdr_abs_error'] if rr['vcdr_abs_error'] is not None else np.nan
        rows.append(dict(**r, metrics=mm, ratios=rr))
    save(root/'scorer/individual_metrics.private.json', rows)
    np.save(root/'scorer/dice.private.npy', dice)
    absolute = [];contrasts = [];cells = []
    def finite_mean(array):
        return float(np.nanmean(array)) if np.isfinite(array).any() else None
    for ai, a in enumerate(ARMS):
        for mi, mode in enumerate(MODES):
            if a in ('B0', 'D') and mi:
                continue
            absolute.append(dict(arm=a, mode='DIRECT' if a in ('B0', 'D') else mode,
                 macro_dice_percent=float(dice[:, :, ai, mi].mean()),
                 assd_pixel=finite_mean(surface[:, :, ai, mi]), hd95_pixel=finite_mean(hd95[:, :, ai, mi]),
                 surface_defined=int(np.isfinite(surface[:, :, ai, mi]).sum()), surface_total=64,
                 empty_predictions=int(empty[:, :, ai, mi].sum()), vcdr_mae=finite_mean(ratios[:, :, ai, mi]),
                 vcdr_defined=int(np.isfinite(ratios[:, :, ai, mi]).sum()), vcdr_total=32))
            for k in range(4):
                for ch, channel in enumerate(('OD', 'OC')):
                    cells.append(dict(arm=a, mode='DIRECT' if a in ('B0', 'D') else mode, condition=f'C{k}', channel=channel,
                                      macro_dice_percent=float(dice[:, k, ai, mi, ch].mean())))
    x = ARMS.index('X');f = ARMS.index('F');b = ARMS.index('B0')
    for control in ('F', 'P', 'S', 'D', 'H', 'C_A', 'C_FULL', 'B0'):
        for mi, mode in enumerate(MODES):
            delta = dice[:, :, x, mi]-dice[:, :, ARMS.index(control), mi]
            contrasts.append(dict(comparison='X-'+control, mode=mode, delta_pp=float(delta.mean()),
                                  ci95_pp=interval(delta), condition_delta_pp=delta.mean((0, 2)).tolist(),
                                  worst_content_condition_channel_pp=float(delta.min())))
    primary = next(r for r in contrasts if r['comparison'] == 'X-F' and r['mode'] == 'CROSS_MATCH')
    delta_base = dice[:, :, x, 1]-dice[:, :, b, 1]
    secondary = {a: float((dice[:, :, x, 1]-dice[:, :, ARMS.index(a), 1]).mean()) for a in ('P', 'S', 'D', 'H')}
    original_delta = float(delta_base[:, 0].mean())
    new_empty = int((empty[:, :, x, 1] & ~empty[:, :, b, 1]).sum())
    surface_coverage = int(np.isfinite(surface[:, :, x, 1]).sum()) >= int(np.isfinite(surface[:, :, b, 1]).sum())
    ratio_coverage = int(np.isfinite(ratios[:, :, x, 1]).sum()) >= int(np.isfinite(ratios[:, :, b, 1]).sum())
    structure = (finite_mean(surface[:, :, x, 1]) is not None and finite_mean(surface[:, :, b, 1]) is not None
                 and finite_mean(surface[:, :, x, 1]) <= finite_mean(surface[:, :, b, 1])
                 and finite_mean(ratios[:, :, x, 1]) is not None and finite_mean(ratios[:, :, b, 1]) is not None
                 and finite_mean(ratios[:, :, x, 1]) <= finite_mean(ratios[:, :, b, 1]))
    gates = dict(primary_mean=primary['delta_pp'] >= .5, primary_ci_lower=primary['ci95_pp'][0] > 0,
                 secondary_all_positive=all(v > 0 for v in secondary.values()),
                 original_no_harm=original_delta >= -.2, worst_no_harm=float(delta_base.min()) >= -1,
                 no_new_empty=new_empty == 0, surface_coverage=surface_coverage,
                 vcdr_coverage=ratio_coverage, structure_direction=bool(structure))
    transitions = []
    for a in ARMS:
        aa = [r for r in rows if r['arm'] == a and r['mode'] in ('DIRECT', 'CROSS_MATCH')]
        for ch, channel in enumerate(('OD', 'OC')):
            for region in ('all', 'foreground', 'boundary3px'):
                rr = [r['metrics'][ch]['regions'][region] for r in aa]
                for measure in ('corrected_fraction', 'spoiled_fraction'):
                    valid = [r[measure] for r in rr if r[measure] is not None]
                    transitions.append(dict(arm=a, channel=channel, region=region, measure=measure,
                                            macro_fraction=float(np.mean(valid)) if valid else None,
                                            defined=len(valid), total=len(rr)))
    result = dict(status='COMPLETE', evidence='CONTENT_DISJOINT_SOURCE_DEVELOPMENT_ONLY',
         patient_independence=False, checkpoint_pretraining_membership='UNKNOWN', independent_pair_clusters=8,
         deployment_readouts=736, training_episodes=1000, final_source_label_files=8,
         primary=primary, secondary_mean_pp=secondary, original_X_minus_B0_pp=original_delta,
         worst_X_minus_B0_pp=float(delta_base.min()), new_empty_predictions=new_empty,
         gates=gates, advance_development_gate=all(gates.values()),
         absolute=absolute, contrasts=contrasts, interval_scope='Descriptive content-pair bootstrap; unknown patient dependence; one training seed',
         cpu_seconds=time.time()-started, rdc=False, engram=False, no_automatic_next_experiment=True)
    save(root/'public/RESULTS.json', result)
    csvout(root/'public/absolute.csv', absolute);csvout(root/'public/condition_channel.csv', cells)
    csvout(root/'public/transitions.csv', transitions)
    csvout(root/'public/contrasts.csv', [dict(comparison=r['comparison'],mode=r['mode'],delta_pp=r['delta_pp'],
           ci95_lower_pp=r['ci95_pp'][0], ci95_upper_pp=r['ci95_pp'][1],
           worst_content_condition_channel_pp=r['worst_content_condition_channel_pp']) for r in contrasts])
    decision = 'PASS: only the fixed development gate passed; no CTTA or clinical efficacy claim.' if all(gates.values()) else 'STOP: the fixed development gate did not pass; no tuning, expansion, or next experiment.'
    text = '\n'.join(['# R78 source-development boundary objective pilot', '', decision, '',
         f"Primary CROSS_MATCH X-F: {primary['delta_pp']:+.6f} Dice pp; descriptive 95% interval {primary['ci95_pp']}.", '',
         'All 1,000 outer episodes and 736 final readouts completed. RDC and ENGRAM were disabled.', '',
         'This is a content-disjoint source-development experiment on synthetic appearance changes. Patient linkage and backbone pretraining membership remain unknown. Eight content-pair clusters and one training seed do not establish independent-patient, hospital, or CTTA generalization.', '',
         'Source labels were used to train and select the five heads. Final labels were read by this independent CPU scorer after the GPU worker retired. Strong controls have the same source-label visit budget, not equal FLOPs.', '',
         'C_FULL preserves the actual existing CW_LSO C host: six weak views, one strong view, all-BN Adam; no G-host entropy perturbation. C_A uses the same W/LSO objective with the fixed 512-dimensional SGD update.', '',
         'Surface distances are pixels; undefined surfaces and vCDR remain missing with explicit denominators. vCDR and area ratio are separate. Every method, input condition and anatomical channel is included in the accompanying tables.', '',
         'Source images, masks, model weights, individual outputs, identifiers, and official third-party archive files are excluded from public delivery.', ''])
    (root/'public/REPORT.md').write_text(text)
    save(root/'SCORE_COMPLETE.json', dict(status='COMPLETE', predictions=736, cpu_seconds=time.time()-started))
