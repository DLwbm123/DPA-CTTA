"""Aggregate existing frozen confusion counts without reading images or labels."""
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from ..r10_12h_core.run import read, save
from ..r20_model_only_search.report import csvout


def rates(tp, fp, fn):
    actual = tp + fn
    return dict(fp_per_gt=fp / actual if actual else None,
                fn_per_gt=fn / actual if actual else None,
                area_ratio=(tp + fp) / actual if actual else None)


def main():
    assert rates(6, 3, 4) == dict(fp_per_gt=.3, fn_per_gt=.4, area_ratio=.9)
    assert rates(0, 0, 0)['fp_per_gt'] is None
    c = read(os.environ['RUN_CONFIG']); root = Path(c['output_root']); src = Path(c['r52_root'])
    baseline = c.get('baseline', 'VPTTA'); assert baseline in ('C', 'VPTTA')
    started = time.time(); state = dict(status='RUNNING', started=started, GPU_seconds=0, new_label_reads=0)
    save(root/'RUN_STATE.json', state)
    try:
        assert read(src/'RUN_STATE.json')['status'] == 'COMPLETE'
        groups = defaultdict(lambda: [0, 0, 0, 0]); observations = 0
        for seed in (20260907, 17011, 29009):
            for order in (0, 1):
                jid = f's{seed}_o{order}'
                paths = [src/'scorer/reused'/f'CW_LSO_{jid}.private.jsonl', src/('scorer/reused' if baseline == 'C' else 'scores/final')/f'{baseline}_{jid}.private.jsonl']
                streams = [[json.loads(line) for line in p.read_text().splitlines()] for p in paths]
                assert len(streams[0]) == len(streams[1]) == 1951
                for a, b in zip(*streams):
                    assert (a['content'], a['visit'], a['subset'], a['domain']) == (b['content'], b['visit'], b['subset'], b['domain'])
                    if a['subset'] != 'remaining_dev': continue
                    observations += 1
                    for method, row in zip(('CW_LSO', baseline), (a, b)):
                        assert row['seed'] == seed and row['order'] == order
                        for m in row['metrics']:
                            tp, fp, fn = m['TP'], m['FP'], m['FN']
                            assert min(tp, fp, fn) >= 0 and tp + fp == m['pred_pixels'] and tp + fn == m['gt_pixels']
                            g = groups[(method, seed, order, row['domain'], m['channel'])]
                            for i, value in enumerate((1, tp, fp, fn)): g[i] += value
        assert observations == 10170
        rows = []
        for (method, seed, order, domain, channel), (n, tp, fp, fn) in sorted(groups.items()):
            rows.append(dict(method=method, seed=seed, order=order, domain=domain, channel=channel, images=n, TP=tp, FP=fp, FN=fn, **rates(tp, fp, fn)))
        csvout(root/'public/CONFUSION_AGGREGATES.csv', rows)
        index = {(r['method'], r['seed'], r['order'], r['domain'], r['channel']): r for r in rows}; paired = []
        for r in rows:
            if r['method'] != 'CW_LSO': continue
            b = index[(baseline, r['seed'], r['order'], r['domain'], r['channel'])]
            paired.append(dict(seed=r['seed'], order=r['order'], domain=r['domain'], channel=r['channel'], fp_excess=r['fp_per_gt']-b['fp_per_gt'], fn_excess=r['fn_per_gt']-b['fn_per_gt'], area_ratio_cw=r['area_ratio'], area_ratio_baseline=b['area_ratio'], baseline=baseline))
        csvout(root/'public/PAIRED_ERROR.csv', paired)
        focus = [r for r in paired if r['domain']=='REFUGE_Valid' and r['channel']=='OC']; assert len(focus)==6
        excess = [r['fp_excess']-r['fn_excess'] for r in focus]
        support = mean(excess) >= .05 and sum(x>0 for x in excess)>=5
        decision = dict(oversegmentation_hypothesis_supported=support, mean_fp_minus_fn_excess=mean(excess), positive_pairs=sum(x>0 for x in excess), independent_validation=False, adaptation_runs=0)
        save(root/'public/DECISION.json', decision)
        (root/'public/REPORT.md').write_text('# '+c.get('diagnostic_id', 'R54')+' frozen confusion-count diagnostic\n\nBaseline: '+baseline+'\n\n'+json.dumps(decision, indent=2)+'\n\nAll domains, channels, seeds and orders are retained in the tables. Rates are ratios of pooled pixel counts within each cell, not mean per-image rates or Dice. This is descriptive development evidence; no causal mechanism, deployed improvement or independent validation is established. No images, masks or new labels were read.\n')
        (root/'public/PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())
        state.update(status='COMPLETE', ended=time.time(), wall_seconds=time.time()-started, paired_images=observations, aggregate_rows=len(rows))
        save(root/'RUN_STATE.json', state); save(root/'public/RUN_STATE.json', state)
    except BaseException as e:
        state.update(status='FAILED', reason=str(e), ended=time.time()); save(root/'RUN_STATE.json', state); raise


if __name__ == '__main__': main()
