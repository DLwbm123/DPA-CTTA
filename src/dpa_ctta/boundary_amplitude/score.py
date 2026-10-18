"""Score every fixed output only after the label-free GPU worker retires."""
from pathlib import Path
import os
import time

import numpy as np

from .run import ARMS, MODES, read, save
from ..r78_cross_patient.score import metric, interval, csvout


def score(c):
    from ..source_io import read_mask
    root = Path(c['output_root']);started = time.time()
    retirement = read(root/'processes/worker.json');state = read(root/'RUN_STATE.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or retirement['active'] or retirement['exit_code']:
        raise ValueError('retired successful GPU worker and CPU-only scoring required')
    if state['status'] != 'GPU_COMPLETE' or state['predictions'] != 1024 or state['baseline_parity'] != 352:
        raise ValueError('complete prediction/parity coverage required')
    if read(root/'MATCHING_AUDIT.json')['status'] != 'PASS':
        raise ValueError('actual amplitude matching failed')
    records = read(root/'private/predictions.json');panel = read(root/'scorer/panel.json')
    keys = [(r['pair'], r['condition'], r['arm'], r['mode']) for r in records]
    expected = {(p, k, a, m) for p in range(8) for k in range(4) for a in ARMS
                for m in (('DIRECT',) if a in ('B0', 'D') else MODES)}
    if len(records) != 1024 or len(set(keys)) != 1024 or set(keys) != expected or len(panel) != 8:
        raise ValueError('fixed matrix coverage')
    masks = {}
    for r, key in zip(records, keys):
        raw = (root/'predictions'/f'{r["index"]:04d}.bits').read_bytes()
        if len(raw) != 65536:
            raise ValueError('invalid packed prediction')
        masks[key] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(2, 512, 512).astype(bool)
    labels = [read_mask(r['mask_path'], 'fundus', r['image_size']).numpy()[0].astype(bool) for r in panel]
    shape = (8, 4, len(ARMS), 3, 2)
    values = np.zeros(shape);surface = np.full(shape, np.nan);hd95 = surface.copy()
    empty = np.zeros(shape, dtype=bool);vcdr = np.full(shape[:-1], np.nan)
    individual = [];groups = {}
    for r, key in zip(records, keys):
        if time.time()-started > c['cpu_cap_seconds']-10:
            raise TimeoutError('CPU scorer cap')
        p, k, arm, mode = key;ai = ARMS.index(arm)
        metrics, ratios = metric(masks[key], labels[p], masks[p, k, 'B0', 'DIRECT'])
        for mi in (range(3) if mode == 'DIRECT' else [MODES.index(mode)]):
            values[p, k, ai, mi] = [m['dice_percent'] for m in metrics]
            surface[p, k, ai, mi] = [m['assd_pixel'] if m['assd_pixel'] is not None else np.nan for m in metrics]
            hd95[p, k, ai, mi] = [m['hd95_pixel'] if m['hd95_pixel'] is not None else np.nan for m in metrics]
            empty[p, k, ai, mi] = [m['pred_empty'] for m in metrics]
            vcdr[p, k, ai, mi] = ratios['vcdr_abs_error'] if ratios['vcdr_abs_error'] is not None else np.nan
        individual.append(dict(**r, metrics=metrics, ratios=ratios))
        groups.setdefault((arm, mode), []).append(r['diagnostics'])
    save(root/'scorer/individual_metrics.private.json', individual)
    np.save(root/'scorer/dice.private.npy', values)
    def mean(x):
        return float(np.nanmean(x)) if np.isfinite(x).any() else None
    absolute = [];cells = [];diagnostics = [];transitions = []
    for ai, arm in enumerate(ARMS):
        for mi, mode in enumerate(MODES):
            if arm in ('B0', 'D') and mi:
                continue
            label = 'DIRECT' if arm in ('B0', 'D') else mode
            absolute.append(dict(arm=arm, mode=label, macro_dice_percent=float(values[:, :, ai, mi].mean()),
                 assd_pixel=mean(surface[:, :, ai, mi]), hd95_pixel=mean(hd95[:, :, ai, mi]),
                 surface_defined=int(np.isfinite(surface[:, :, ai, mi]).sum()), surface_total=64,
                 vcdr_mae=mean(vcdr[:, :, ai, mi]), vcdr_defined=int(np.isfinite(vcdr[:, :, ai, mi]).sum()),
                 vcdr_total=32, empty_predictions=int(empty[:, :, ai, mi].sum())))
            for k in range(4):
                for ch, channel in enumerate(('OD', 'OC')):
                    cells.append(dict(arm=arm, mode=label, condition=f'C{k}', channel=channel,
                                      macro_dice_percent=float(values[:, k, ai, mi, ch].mean())))
            dd = groups[arm, label];out = dict(arm=arm, mode=label)
            for field in ('raw_norm', 'target_norm', 'actual_norm', 'norm_absolute_error', 'direction_cosine', 'multiplier',
                          'norm', 'BN_update_norm', 'logit_rms', 'probability_abs_mean', 'changed_pixels'):
                numbers = [d[field] for d in dd if d.get(field) is not None]
                if numbers:
                    out.update({field+'_mean': float(np.mean(numbers)), field+'_min': float(np.min(numbers)), field+'_max': float(np.max(numbers))})
            diagnostics.append(out)
        rr = [r for r in individual if r['arm'] == arm and r['mode'] in ('DIRECT', 'CROSS_MATCH')]
        for ch, channel in enumerate(('OD', 'OC')):
            for region in ('all', 'foreground', 'boundary3px'):
                for measure in ('corrected_fraction', 'spoiled_fraction'):
                    array = [r['metrics'][ch]['regions'][region][measure] for r in rr]
                    valid = [v for v in array if v is not None]
                    transitions.append(dict(arm=arm, channel=channel, region=region, measure=measure,
                         macro_fraction=float(np.mean(valid)) if valid else None, defined=len(valid), total=len(array)))
    x = ARMS.index('X_MATCH');b = ARMS.index('B0');contrasts = []
    for control in ARMS:
        if control == 'X_MATCH':
            continue
        for mi, mode in enumerate(MODES):
            delta = values[:, :, x, mi]-values[:, :, ARMS.index(control), mi]
            contrasts.append(dict(comparison='X_MATCH-'+control, mode=mode, delta_pp=float(delta.mean()),
                 ci95_pp=interval(delta), condition_delta_pp=delta.mean((0, 2)).tolist(),
                 worst_content_condition_channel_pp=float(delta.min()), negative_cells=int((delta < 0).sum()), total_cells=64))
    primary = next(r for r in contrasts if r['comparison'] == 'X_MATCH-F' and r['mode'] == 'CROSS_MATCH')
    scaling = next(r for r in contrasts if r['comparison'] == 'X_MATCH-X_RAW' and r['mode'] == 'CROSS_MATCH')
    delta_b = values[:, :, x, 1]-values[:, :, b, 1]
    secondary = {a: float((values[:, :, x, 1]-values[:, :, ARMS.index(a), 1]).mean())
                 for a in ('P_MATCH', 'S_MATCH', 'H_MATCH', 'C_A_MATCH', 'X_NEG', 'RANDOM_MATCH', 'D')}
    gates = dict(primary_mean=primary['delta_pp'] >= .5, primary_ci=primary['ci95_pp'][0] > 0,
         all_strong_directions_positive=all(v > 0 for v in secondary.values()),
         original_no_harm=float(delta_b[:, 0].mean()) >= -.2, worst_no_harm=float(delta_b.min()) >= -1,
         no_new_empty=not bool((empty[:, :, x, 1] & ~empty[:, :, b, 1]).any()),
         surface_coverage=int(np.isfinite(surface[:, :, x, 1]).sum()) >= int(np.isfinite(surface[:, :, b, 1]).sum()),
         vcdr_coverage=int(np.isfinite(vcdr[:, :, x, 1]).sum()) >= int(np.isfinite(vcdr[:, :, b, 1]).sum()))
    gates['structure_direction'] = (mean(surface[:, :, x, 1]) is not None and mean(vcdr[:, :, x, 1]) is not None
         and mean(surface[:, :, b, 1]) is not None and mean(vcdr[:, :, b, 1]) is not None
         and mean(surface[:, :, x, 1]) <= mean(surface[:, :, b, 1]) and mean(vcdr[:, :, x, 1]) <= mean(vcdr[:, :, b, 1]))
    result = dict(status='COMPLETE', final_readouts=1024, baseline_parity=352, actual_matched_states=672,
         new_training=False, heads='FROZEN_R78_SELECTED_HEADS', patient_independence=False, query_development_exposed=True,
         primary=primary, scaling_comparison=scaling, secondary_mean_pp=secondary, gates=gates,
         advance_development_gate=all(gates.values()), local_useful_direction_suppressed_by_small_step=scaling['ci95_pp'][0] > 0,
         absolute=absolute, contrasts=contrasts, matching=read(root/'MATCHING_AUDIT.json'),
         original_X_MATCH_minus_B0_pp=float(delta_b[:, 0].mean()), worst_X_MATCH_minus_B0_pp=float(delta_b.min()),
         interval_scope='Descriptive bootstrap of eight previously exposed content pairs; patient dependence unknown',
         parameter_norm_matched=True, function_space_matched=False, rdc=False, engram=False,
         final_label_files_scorer_only=8, cpu_seconds=time.time()-started)
    save(root/'public/RESULTS.json', result);save(root/'public/DIAGNOSTICS.json', diagnostics)
    for name, rows in [('absolute', absolute), ('condition_channel', cells), ('transitions', transitions)]:
        csvout(root/'public'/f'{name}.csv', rows)
    csvout(root/'public/contrasts.csv', [dict(comparison=r['comparison'], mode=r['mode'], delta_pp=r['delta_pp'],
         ci95_lower_pp=r['ci95_pp'][0], ci95_upper_pp=r['ci95_pp'][1],
         worst_content_condition_channel_pp=r['worst_content_condition_channel_pp']) for r in contrasts])
    save(root/'SCORE_COMPLETE.json', dict(status='COMPLETE', final_readouts=1024, cpu_seconds=time.time()-started))
