"""Post-hoc error accounting of saved outputs; never generates a new prediction."""
import json
import os
from pathlib import Path
import time
import numpy as np
from ..r51_same_state.run import read, save
from ..r51_same_state.score import csvout

NAMES = ('LOCAL', 'SHUFFLE', 'SIGN0', 'SIGN1', 'SIGN2', 'SIGN3')


def transitions(before, after, truth):
    if before.shape != after.shape or before.shape != truth.shape or any(x.dtype != bool for x in (before, after, truth)):
        raise ValueError('aligned boolean masks required')
    added = after & ~before; removed = before & ~after
    result = dict(add_help=int((added & truth).sum()), add_harm=int((added & ~truth).sum()),
                  remove_help=int((removed & ~truth).sum()), remove_harm=int((removed & truth).sum()),
                  baseline_FP=int((before & ~truth).sum()), baseline_FN=int((~before & truth).sum()),
                  area_change=int(after.sum())-int(before.sum()))
    assert result['area_change'] == result['add_help']+result['add_harm']-result['remove_help']-result['remove_harm']
    assert int((after != truth).sum())-int((before != truth).sum()) == result['add_harm']+result['remove_harm']-result['add_help']-result['remove_help']
    return result


def self_check():
    before = np.array([0, 0, 1, 1], bool); after = ~before; truth = np.array([1, 0, 0, 1], bool)
    r = transitions(before, after, truth)
    assert all(r[k] == 1 for k in ('add_help', 'add_harm', 'remove_help', 'remove_harm'))
    assert transitions(before, before, truth)['area_change'] == 0
    r = transitions(np.zeros(4, bool), np.ones(4, bool), np.ones(4, bool))
    assert r['add_help'] == 4 and r['add_harm'] == 0


def main():
    start = time.monotonic(); self_check()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU only')
    c = read(os.environ['DIAG_CONFIG']); root = Path(c['input_root']); out = Path(c['output_root'])
    if read(root/'RUN_STATE.json')['status'] != 'COMPLETE': raise ValueError('completed input run required')
    for mode in ('worker', 'score'):
        p = read(root/'processes'/f'{mode}.json')
        if p['active'] or p['exit_code'] != 0: raise ValueError('successful retirement required')
    from ..r7_target_screen.runner import TargetReader, file_identity
    panel = read(root/'scorer/panel.json'); scores = np.load(root/'scorer/scores.private.npy')
    if len(panel) != 48 or scores.shape != (48, 2, 7, 2): raise ValueError('fixed input coverage')
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask'); groups = {}
    for i, row in enumerate(panel):
        if time.monotonic()-start > 115: raise TimeoutError('120s analysis budget')
        truth = reader.read(row).numpy()[0].astype(bool)
        for order in (0, 1):
            raw = (root/'predictions'/f'{order*48+i:03d}.bits').read_bytes()
            if len(raw) != 7*65536: raise ValueError('prediction byte coverage')
            masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(7, 2, 512, 512).astype(bool)
            for b, name in enumerate(NAMES, 1):
                for ch, channel in enumerate(('OD', 'OC')):
                    r = transitions(masks[0, ch], masks[b, ch], truth[ch])
                    r['dice_delta_pp'] = float(scores[i, order, b, ch]-scores[i, order, 0, ch])
                    groups.setdefault((order, row['domain'], channel, name), []).append(r)
    for path, (_, identity) in reader.used.items():
        if file_identity(path) != identity: raise ValueError('label file metadata changed')
    rows = []
    for (order, domain, channel, branch), records in sorted(groups.items()):
        totals = {k:sum(r[k] for r in records) for k in records[0] if k != 'dice_delta_pp'}
        adds = totals['add_help']+totals['add_harm']; removes = totals['remove_help']+totals['remove_harm']
        rows.append(dict(order=order, domain=domain, channel=channel, branch=branch, n_contents=len(records),
            **totals, pooled_add_correct_fraction=totals['add_help']/adds if adds else None,
            pooled_remove_correct_fraction=totals['remove_help']/removes if removes else None,
            dice_delta_pp=float(np.mean([r['dice_delta_pp'] for r in records])),
            helped_cases=sum(r['dice_delta_pp']>0 for r in records), harmed_cases=sum(r['dice_delta_pp']<0 for r in records),
            unchanged_cases=sum(r['dice_delta_pp']==0 for r in records)))
    if len(rows) != 96 or any(r['n_contents'] != 12 for r in rows): raise ValueError('aggregate coverage')
    csvout(out/'TRANSITIONS.csv', rows)
    save(out/'COMPLETION.json', dict(status='COMPLETE', post_hoc=True, contents=48, pairs=96,
        comparisons=576, channel_comparisons=1152, aggregate_cells=96, new_predictions=0, model_forwards=0,
        gpu_seconds=0, cpu_wall_seconds=time.monotonic()-start, self_check='PASS',
        interpretation='Pixel fractions are pooled, not patient-level probabilities; no oracle policy or new efficacy test.'))


if __name__ == '__main__': main()
