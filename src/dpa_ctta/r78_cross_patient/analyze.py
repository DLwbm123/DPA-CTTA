"""Reproduce aggregate diagnostics from completed private outputs, without labels."""
import json
import os
from pathlib import Path
import time

import numpy as np

from .run import ARMS, TRAIN, read, save
from .score import csvout, interval


def analyze(root):
    root = Path(root);started = time.time()
    ledger = read(root/'RESOURCE_LEDGER.json')
    if not ledger['all_workers_retired'] or any(a['exit_code'] for a in ledger['attempts']):
        raise ValueError('completed, retired workers required')
    rows = read(root/'private/predictions.json')
    if len(rows) != 736:
        raise ValueError('complete output matrix required')
    def mask(row):
        raw = (root/'predictions'/f'{row["index"]:03d}.bits').read_bytes()
        if len(raw) != 65536:
            raise ValueError('prediction bytes')
        return np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(2, 512, 512).astype(bool)
    baseline = {(r['pair'], r['condition']): mask(r) for r in rows if r['arm'] == 'B0'}
    groups = {}
    for row in rows:
        d = row['diagnostics']
        g = groups.setdefault((row['arm'], row['mode']), dict(norm=[], projected=[], empty=[], delta=[], change=[], any_change=[]))
        if 'norm' in d:
            g['norm'].append(d['norm'])
        elif 'BN_update_norm' in d:
            g['norm'].append(d['BN_update_norm'])
        for key, field in [('projected', 'projected'), ('empty', 'empty'), ('delta', 'delta_mean')]:
            if field in d:
                g[key].append(d[field])
        changed = mask(row) != baseline[row['pair'], row['condition']]
        g['change'].append(changed.mean((1, 2)));g['any_change'].append(bool(changed.any()))
    diagnostics = []
    for (arm, mode), g in groups.items():
        out = dict(arm=arm, mode=mode, mean_changed_pixel_fraction_OD=float(np.mean(g['change'], axis=0)[0]),
                   mean_changed_pixel_fraction_OC=float(np.mean(g['change'], axis=0)[1]),
                   fraction_readouts_with_any_change=float(np.mean(g['any_change'])))
        for key, values in g.items():
            if key not in ('change', 'any_change') and values:
                out.update({key+'_mean': float(np.mean(values)), key+'_min': float(np.min(values)), key+'_max': float(np.max(values))})
        diagnostics.append(out)
    save(root/'public/DIAGNOSTICS.json', dict(rows=diagnostics, cpu_seconds=time.time()-started))
    curves = []
    for arm in TRAIN:
        training = read(root/'private'/f'training_{arm}.json')['steps']
        if len(training) != 200:
            raise ValueError('complete training history required')
        for i in range(0, 200, 25):
            block = training[i:i+25]
            curves.append(dict(arm=arm, episode_start=i+1, episode_end=i+25,
                 mean_supervised_loss=float(np.mean([x['loss'] for x in block])),
                 mean_gradient_norm=float(np.mean([x['gradient_norm'] for x in block]))))
    csvout(root/'public/TRAINING_CURVES.csv', curves)
    values = np.load(root/'scorer/dice.private.npy')
    if values.shape != (8, 4, 9, 3, 2):
        raise ValueError('fixed score matrix required')
    contrasts = {}
    for arm in ('X', 'F', 'D', 'C_FULL'):
        delta = values[:, :, ARMS.index(arm), 1]-values[:, :, ARMS.index('B0'), 1]
        contrasts[arm] = dict(mean_pp=float(delta.mean()), ci95_pp=interval(delta),
             condition_delta_pp=delta.mean((0, 2)).tolist(), channel_delta_pp=delta.mean((0, 1)).tolist(),
             worst_content_condition_channel_pp=float(delta.min()), negative_content_condition_channels=int((delta < 0).sum()),
             total_content_condition_channels=int(delta.size))
    contrasts['mode_differences_pp'] = {arm: dict(
         self_minus_match=float((values[:, :, ARMS.index(arm), 0]-values[:, :, ARMS.index(arm), 1]).mean()),
         match_minus_mismatch=float((values[:, :, ARMS.index(arm), 1]-values[:, :, ARMS.index(arm), 2]).mean()))
         for arm in ('X', 'F', 'C_FULL')}
    contrasts['cpu_seconds'] = time.time()-started
    save(root/'public/SUPPLEMENTARY_COMPARISONS.json', contrasts)


if __name__ == '__main__':
    analyze(read(os.environ['DIAG_CONFIG'])['output_root'])
