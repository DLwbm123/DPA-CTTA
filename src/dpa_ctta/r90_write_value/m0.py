"""Frozen mask-only M0; no model construction or GPU calls."""
import os
import time
START = time.time()
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import csv
import gzip
import json
import math
import signal
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi

SEEDS = (20260907, 17011, 29009)
ARMS = ('SOURCE_ONLY', 'P0', 'NATIVE', 'VV')
METRICS = ('Dice', 'FP', 'FN', 'foreground', 'ASSD_px', 'HD95_px', 'vCDR_abs_error',
           'nesting_pixel_percent', 'nesting_present_percent', 'pred_empty', 'gt_empty')
WIDTH = 131072
EIGHT = np.ones((3, 3), dtype=bool)
FOUR = ndi.generate_binary_structure(2, 1)
DIAGONAL = math.sqrt(2) * 511


def load(p):
    return json.loads(p.read_text())


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def post(mask):
    result = np.zeros_like(mask, dtype=bool)
    for ch in (0, 1):
        labels, n = ndi.label(mask[ch], EIGHT)
        if n:
            count = np.bincount(labels.ravel()); count[0] = 0
            result[ch] = ndi.binary_fill_holes(labels == count.argmax(), structure=FOUR)
    result[1] &= result[0]
    return result


def boundary(mask):
    return mask & ~ndi.binary_erosion(mask, structure=EIGHT, border_value=0)


def vcdr(mask):
    heights = []
    for ch in (0, 1):
        rows = np.flatnonzero(mask[ch].any(axis=1))
        heights.append(int(rows[-1]-rows[0]+1) if len(rows) else 0)
    return heights[1]/heights[0] if heights[0] else None


def metrics(mask, truth, truth_boundary, truth_distance):
    result = np.zeros((2, len(METRICS)), dtype=np.float64)
    pred_ratio, true_ratio = vcdr(mask), vcdr(truth)
    ratio_error = abs(pred_ratio-true_ratio) if pred_ratio is not None and true_ratio is not None else np.nan
    outside = mask[1] & ~mask[0]
    for ch in (0, 1):
        a, b = mask[ch], truth[ch]
        na, nb = int(a.sum()), int(b.sum())
        dice = 200*int((a & b).sum())/(na+nb) if na+nb else 100.
        if not na and not nb:
            assd = hd = 0.
        elif not na or not nb:
            assd = hd = DIAGONAL
        else:
            edge = boundary(a)
            distances = np.concatenate((truth_distance[ch][edge], ndi.distance_transform_edt(~edge)[truth_boundary[ch]]))
            assd, hd = float(distances.mean()), float(np.percentile(distances, 95))
        result[ch] = [dice, 100*(a & ~b).mean(), 100*(~a & b).mean(), 100*a.mean(), assd, hd,
                      ratio_error, 100*outside.mean(), 100*bool(outside.any()), 100*(na == 0), 100*(nb == 0)]
    return result


def mean(x):
    finite = x[np.isfinite(x)]
    return float(finite.mean()) if finite.size else None


def csvout(p, rows):
    with p.open('x') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for row in rows:
            w.writerow({k: 'NA' if v is None else v for k, v in row.items()})


def interval(delta, domains):
    rng = np.random.default_rng(20260907)
    ids = [np.flatnonzero(domains == d) for d in sorted(set(domains))]
    draws = [np.mean([delta[rng.choice(i, len(i), replace=True)].mean() for i in ids]) for _ in range(2000)]
    return np.quantile(draws, (.025, .975)).tolist()


def self_check():
    m = np.zeros((2, 9, 9), bool); m[0, 1:4, 1:4] = True; m[0, 5:8, 5:8] = True
    m[1, 5:7, 5:7] = True
    p = post(m)
    assert p[0, 1:4, 1:4].all() and not p[0, 5:8, 5:8].any() and not p[1].any()
    assert not (p[1] & ~p[0]).any() and np.array_equal(post(p), p)
    z = np.zeros((2, 512, 512), bool)
    got = metrics(z, z, z, z.astype(float))
    assert np.all(got[:, 0] == 100) and np.all(got[:, 4:6] == 0) and np.isnan(got[:, 6]).all()
    a = z.copy(); a[0, 2:5, 2:5] = True
    assert vcdr(a) == 0 and vcdr(z) is None
    got = metrics(a, z, z, z.astype(float))
    assert got[0, 0] == 0 and got[0, 4] == DIAGONAL


def run(c):
    self_check()
    root, source = Path(c['output_root']), Path(c['source_root'])
    public, private = root/'public', root/'private'
    stage_deadline = min(START+1700, c['deadline']-180)
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('M0 independent alarm')))
    signal.alarm(max(1, int(stage_deadline-time.time())))
    def guard():
        if time.time() >= stage_deadline:
            raise TimeoutError('shared or M0 deadline')
        usage = sum(max(p.stat().st_size, p.stat().st_blocks*512) for p in root.rglob('*') if not p.is_symlink())
        if usage >= c['output_cap_bytes']-c['recovery_reserve_bytes']:
            raise RuntimeError('actual allocated output cap')
        return usage
    def state(status, **more):
        save(root/'RUN_STATE.json', dict(stage='M0', status=status, started=START, shared_started=c['started'],
                                       shared_deadline=c['deadline'], **more))
    state('ASSET_AUDIT')
    panels = [load(source/'scorer'/f'screen_o{o}.json') for o in (0, 1)]
    index = {row['image_sha256']: i for i, row in enumerate(panels[0])}
    assert len(index) == 192 and set(index) == {r['image_sha256'] for r in panels[1]}
    names = sorted({r['domain'] for r in panels[0]})
    domains = np.array([names.index(r['domain']) for r in panels[0]])
    files = [('static', source/'jobs/source_only/predictions.bits', None, None)]
    files += [(f'{arm}_s{seed}_o{order}', source/'jobs'/f'{prefix}_s{seed}_o{order}'/'predictions.bits', arm, (seed, order))
              for seed in SEEDS for order in (0, 1) for arm, prefix in [('NATIVE', 'baseline_NATIVE'), ('VV', 'replay_VV')]]
    assert all(p.stat().st_size == 192*WIDTH for _, p, _, _ in files)
    save(public/'ASSET_AUDIT.json', dict(status='PASS', contents=192, orders=2, seeds=3, binary_files=13,
         expected_bytes_each=192*WIDTH, probability_outputs='NOT_AVAILABLE', matched_reset='NOT_AVAILABLE',
         matched_fixed_multiview_static='NOT_AVAILABLE', coordinates='same source reader and 512x512 output grid',
         new_image_reads=0, new_model_calls=0, same_hardware_main_controls=True))
    state('SEALING_STATIC_CONTROL')
    for label, path, _, _ in files:
        guard()
        with path.open('rb') as f, gzip.GzipFile(filename=str(private/(label+'.post.bits.gz')), mode='xb', compresslevel=1, mtime=0) as g:
            for i in range(192):
                frame = f.read(WIDTH); assert len(frame) == WIDTH
                mask = np.unpackbits(np.frombuffer(frame, dtype=np.uint8)).reshape(2, 2, 512, 512).astype(bool)
                g.write(np.packbits(np.stack([post(x) for x in mask])).tobytes())
                if i % 16 == 0:
                    guard()
            assert not f.read(1)
    save(public/'SEALED.json', dict(status='ALL_13_TRANSFORMED_STREAMS_SEALED', sealed_at=time.time(),
         label_reads_before_seal=0, post_rule='8-LCC first raster tie; 4-background fill; OC intersect OD',
         new_model_calls=0, new_GPU_updates=0))
    state('CPU_SCORING')
    from dpa_ctta.r7_target_screen.runner import TargetReader
    config = load(source/'private/config.json')
    reader = TargetReader(config['target_root'], 256*1024**2, 'mask')
    truths, edges, distances = [], [], []
    for row in panels[0]:
        guard(); t = reader.read(row).numpy()[0].astype(bool)
        truths.append(t); e = np.stack([boundary(x) for x in t]); edges.append(e)
        distances.append(np.stack([ndi.distance_transform_edt(~x).astype(np.float32) for x in e]))
    # axes: content, anchor, arm, PRE/K, raw/post, channel, metric.
    values = np.full((192, 6, 4, 2, 2, 2, len(METRICS)), np.nan)
    old = np.load(source/'scorer/values.private.npz')
    checked = 0
    for label, path, arm, stream in files:
        order = stream[1] if stream else 0
        anchor = SEEDS.index(stream[0])*2+order if stream else None
        with path.open('rb') as raw, gzip.open(private/(label+'.post.bits.gz'), 'rb') as pp:
            for visit, row in enumerate(panels[order]):
                i = index[row['image_sha256']]
                blocks = [f.read(WIDTH) for f in (raw, pp)]
                assert all(len(b) == WIDTH for b in blocks)
                for variant, block in enumerate(blocks):
                    masks = np.unpackbits(np.frombuffer(block, dtype=np.uint8)).reshape(2, 2, 512, 512).astype(bool)
                    for output, mask in enumerate(masks):
                        got = metrics(mask, truths[i], edges[i], distances[i])
                        if arm is None:
                            values[i, :, output, :, variant] = got
                            ref = old['source'][i] if output == 0 else old['p0'][i, 0]
                        else:
                            ai = ARMS.index(arm); values[i, anchor, ai, output, variant] = got
                            ref = old['native'][i, anchor, output] if arm == 'NATIVE' else old['values'][i, anchor, 0, output]
                        if variant == 0:
                            assert np.allclose(got[:, 0], ref, atol=1e-10, rtol=0), 'old raw Dice parity'
                            checked += 1
                if visit % 16 == 0:
                    usage = guard(); state('CPU_SCORING', stream=label, visit=visit+1, root_bytes=usage)
            assert not raw.read(1) and not pp.read(1)
    reader.after_check()
    assert np.isfinite(values[..., :6]).all() and np.isfinite(values[..., 7:]).all()
    np.savez_compressed(private/'m0_values.private.npz', values=values, domains=domains)
    state('AGGREGATING')
    summary = []
    for ai, arm in enumerate(ARMS):
        for output, phase in enumerate(('PRE', 'K')):
            if ai < 2 and output == 0:
                continue
            for variant, kind in enumerate(('raw', 'post')):
                for j in range(6):
                    for di in range(4):
                        for ch, head in enumerate(('OD', 'OC')):
                            for mi, metric in enumerate(METRICS):
                                v = values[domains == di, j, ai, output, variant, ch, mi]
                                summary.append(dict(arm=arm, phase='STATIC' if ai < 2 else phase, variant=kind,
                                     seed=SEEDS[j//2], order=j%2, domain=f'D{di}', channel=head, metric=metric,
                                     mean=mean(v), n=len(v), n_NA=int((~np.isfinite(v)).sum())))
    csvout(public/'ALL_SCORES.csv', summary)
    pairs, pair_cells, content, temporal = [], [], [], []
    for variant, kind in enumerate(('raw', 'post')):
        for a, b in [('VV', 'P0'), ('VV', 'NATIVE'), ('NATIVE', 'P0')]:
            ai, bi = ARMS.index(a), ARMS.index(b)
            for output, phase in enumerate(('PRE', 'K')):
                delta = values[:, :, ai, output, variant] - values[:, :, bi, output, variant]
                contrasts(public, f'{a}-{b}/{kind}/{phase}', delta, domains, pairs, pair_cells, content)
        for ai, arm in enumerate(ARMS):
            delta = values[:, :, ai, 1, 1]-values[:, :, ai, 1, 0]
            if kind == 'post':
                contrasts(public, f'{arm}/post-raw/K', delta, domains, pairs, pair_cells, content)
        for ai, arm in enumerate(('NATIVE', 'VV'), 2):
            h = values[:, :, ai, 0, variant]-values[:, :, 1, 0, variant]
            current = values[:, :, ai, 1, variant]-values[:, :, ai, 0, variant]
            total = values[:, :, ai, 1, variant]-values[:, :, 1, 0, variant]
            assert np.allclose((h+current)[..., 0], total[..., 0], atol=1e-10)
            for label, data in [('H', h), ('C', current), ('T', total)]:
                for di in range(4):
                    for ch, head in enumerate(('OD', 'OC')):
                        temporal.append(dict(arm=arm, variant=kind, term=label, domain=f'D{di}', channel=head,
                                             Dice_delta_pp=mean(data[domains == di, ..., ch, 0])))
    csvout(public/'CONTRASTS.csv', pairs); csvout(public/'ALL_CONTRAST_CELLS.csv', pair_cells)
    csvout(public/'CONTENT_DISTRIBUTIONS.csv', content); csvout(public/'H_C_T.csv', temporal)
    gates = {}
    for b in ('P0', 'NATIVE'):
        delta = values[:, :, 3, 1, 1, :, 0]-values[:, :, ARMS.index(b), 1, 1, :, 0]
        anchors = np.mean([delta[domains == di].mean((0, 2)) for di in range(4)], axis=0)
        ci = interval(delta, domains)
        ok = anchors.mean() >= .3 and ci[0] > 0 and (anchors > 0).sum() >= 5
        ok = ok and all(anchors[o::2].mean() > 0 for o in (0, 1)) and all(anchors[s*2:s*2+2].mean() > 0 for s in range(3))
        gates[b] = dict(macro_delta_pp=float(anchors.mean()), ci95_pp=ci, positive_anchors=int((anchors > 0).sum()),
                        anchors=anchors.tolist(), passed=bool(ok))
    delta = values[:, :, 3, 1, 1, :, 0]-values[:, :, 1, 1, 1, :, 0]
    d1 = delta[domains == 1].mean(0); d3 = delta[domains == 3, :, 1]
    d3_anchor = d3.mean(0); neg_contents = int((d3.mean(1) < 0).sum())
    protect = bool((d1 > 0).all())
    residual = d3.mean() <= -.1 and (d3_anchor < 0).sum() >= 5 and neg_contents > len(d3)/2
    passed = gates['P0']['passed'] and gates['NATIVE']['passed'] and protect and residual
    save(public/'M0_GATE.json', dict(status='PASS' if passed else 'NOT_PASSED', comparisons=gates,
          D1_both_heads_all_anchors_positive=protect, D1_anchor_head_deltas=d1.tolist(),
          residual_D3_OC_delta_pp=float(d3.mean()), residual_D3_OC_negative_anchors=int((d3_anchor < 0).sum()),
          residual_D3_OC_negative_contents=neg_contents, residual_D3_contents=len(d3), residual_problem_passed=bool(residual),
          authorize_M1=bool(passed), authorize_M2=False, independent_validation=False))
    final_usage = guard()
    save(public/'COMPLETION.json', dict(status='COMPLETE', started_epoch=START, ended_epoch=time.time(),
         M0_script_wall_seconds=time.time()-START, shared_started=c['started'], shared_deadline=c['deadline'],
         new_model_calls=0, new_GPU_updates=0, new_image_reads=0, label_contents=192,
         raw_Dice_parity_checks=checked, all_transformed_outputs_sealed_before_labels=True,
         actual_conservative_root_bytes=final_usage, self_checks='PASS', M0_passed=bool(passed)))
    state('COMPLETE', M0_passed=bool(passed))
    print(json.dumps(load(public/'M0_GATE.json')), flush=True)


def contrasts(public, name, delta, domains, summary, cells, contents):
    for mi, metric in enumerate(METRICS):
        data = delta[..., mi]
        anchors = np.array([mean(np.array([mean(data[domains == di, j]) for di in range(4)], dtype=float)) for j in range(6)], dtype=float)
        summary.append(dict(contrast=name, metric=metric, macro_delta=mean(anchors),
                            positive_anchors=int((anchors > 0).sum()), negative_anchors=int((anchors < 0).sum())))
        for di in range(4):
            for ch, head in enumerate(('OD', 'OC')):
                ds = data[domains == di, :, ch]
                for j in range(6):
                    cells.append(dict(contrast=name, metric=metric, domain=f'D{di}', channel=head,
                                 seed=SEEDS[j//2], order=j%2, mean_delta=mean(ds[:, j]), n_NA=int((~np.isfinite(ds[:, j])).sum())))
                counts = np.isfinite(ds).sum(1)
                a = np.divide(np.nansum(ds, axis=1), counts, out=np.full(len(ds), np.nan), where=counts > 0)
                finite = a[np.isfinite(a)]
                contents.append(dict(contrast=name, metric=metric, domain=f'D{di}', channel=head, n_contents=len(ds),
                    n_NA=int((~np.isfinite(a)).sum()), mean_delta=mean(a), median_delta=float(np.median(finite)) if len(finite) else None,
                    p10=float(np.quantile(finite, .1)) if len(finite) else None, p90=float(np.quantile(finite, .9)) if len(finite) else None,
                    minimum=float(finite.min()) if len(finite) else None, maximum=float(finite.max()) if len(finite) else None,
                    positive_contents=int((a > 0).sum()), negative_contents=int((a < 0).sum())))


if __name__ == '__main__':
    cfg = load(Path(os.environ['CONFIG']))
    try:
        run(cfg)
    except BaseException as exc:
        save(Path(cfg['output_root'])/'M0_FAILURE.json', dict(status='NOT_EVALUABLE', error=type(exc).__name__,
             reason=str(exc), started=START, ended=time.time()))
        raise
