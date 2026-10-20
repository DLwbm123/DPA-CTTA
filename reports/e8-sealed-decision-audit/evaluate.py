import itertools
from pathlib import Path
from fractions import Fraction
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion, distance_transform_edt
from core import read, save, unpack, components, aggregate, auc, ARMS
from metric import metric


def compare_frames(frames, utilities, control, weighted, subset=None):
    delta = np.full((8, 4), np.nan)
    for f in frames:
        i, k = f['case'], f['condition']
        rows = [r for r in f['blocks'] if r['common'] and (subset is None or subset(r))]
        u = [utilities[i, k, r['block_id']] for r in rows]
        a = auc([r['scores']['CROP_Q'] for r in rows], u, weighted)
        b = auc([r['scores'][control] for r in rows], u, weighted)
        if a is not None and b is not None:
            delta[i, k] = a - b
    return delta


def run(c):
    root, old = Path(c['root']), Path(c['sealed_root'])
    rec = read(root / 'processes/select.json')
    from runtime import identity
    assert rec['exit_code'] == 0 and not rec['active'] and identity(rec['pid']) != rec['start_ticks']
    frames = read(old / 'private/scores.json')
    decisions = read(root / 'private/DECISIONS.json')
    utility = read(old / 'scorer/utility.private.json')
    utilities = {(r['case'], r['condition'], r['block_id']): float(Fraction(r['utility_numerator'], r['utility_denominator'])) for r in utility}
    assert len(utility) == len(utilities) == 3028
    panels = read(old / 'scorer/panel.json')
    labels = []
    for item in panels:
        with Image.open(item['mask_path']) as im:
            assert list(im.size) == item['image_size']
            y = np.asarray(im.convert('L').resize((512, 512), Image.Resampling.NEAREST))
        labels.append(np.stack([y < 255, y == 0]))
    arrays = {}
    details = []
    coverage = {a: dict(surface_defined=0, vcdr_defined=0, empty_heads=0, nesting_pixels=0,
                        selected_blocks=0, selected_pixels=0, zero_edit_frames=0) for a in ARMS}
    def put(arm, key, i, k, value):
        arrays.setdefault((arm, key), np.full((8, 4), np.nan))[i, k] = value if value is not None else np.nan
    count = 0
    for f, choice in zip(frames, decisions):
        i, k = f['case'], f['condition']
        assert (i, k) == (choice['case'], choice['condition'])
        s = unpack(old / 'inputs' / f'{i}_{k}_SOURCE.bits')
        b = unpack(old / 'inputs' / f'{i}_{k}_TOPO_SMOOTH.bits')
        y = labels[i]
        groups = components(s, b)
        byid = {r['block_id']: r for r in f['blocks']}
        for arm in ARMS:
            pred = s.copy() if arm == 'SOURCE' else b.copy()
            ids = choice['decisions'].get(arm, [])
            edit = np.isin(groups, ids) if ids else np.zeros_like(groups, bool)
            if arm != 'SOURCE':
                pred[:, edit] = s[:, edit]
                assert np.array_equal(pred[0], b[0])
            values, ratios = metric(pred, y, s)
            for ch, name in enumerate(('OD', 'OC')):
                put(arm, name + '_dice', i, k, values[ch]['dice_percent'])
            put(arm, 'ALL_dice', i, k, np.mean([v['dice_percent'] for v in values]))
            for key, field in [('ASSD', 'assd_pixel'), ('HD95', 'hd95_pixel')]:
                v = [v[field] for v in values if v[field] is not None]
                put(arm, key, i, k, float(np.mean(v)) if len(v) == 2 else None)
            put(arm, 'vCDR', i, k, ratios['vcdr_abs_error'])
            cov = coverage[arm]
            cov['surface_defined'] += sum(v['assd_pixel'] is not None for v in values)
            cov['vcdr_defined'] += ratios['vcdr_abs_error'] is not None
            cov['empty_heads'] += sum(v['pred_empty'] for v in values)
            cov['nesting_pixels'] += int((pred[1] & ~pred[0]).sum())
            cov['selected_blocks'] += len(ids)
            pixels = int((pred[1] != b[1]).sum())
            cov['selected_pixels'] += pixels
            cov['zero_edit_frames'] += pixels == 0
            put(arm, 'selected_pixels', i, k, pixels)
            put(arm, 'selected_blocks', i, k, len(ids))
            p, g, t = int(b[1].sum()), int(y[1].sum()), int((b[1] & y[1]).sum())
            dp, dt = int(pred[1].sum()) - p, int((pred[1] & y[1]).sum()) - t
            db = Fraction(2 * t, p + g) if p + g else Fraction(1)
            da = Fraction(2 * (t + dt), p + g + dp) if p + g + dp else Fraction(1)
            assert abs(100 * float(da) - values[1]['dice_percent']) < 1e-12
            if p + g and p + g + dp:
                assert da - db == Fraction(2 * dt - db * dp, p + g + dp)
            for base_name, base in [('B', b), ('SOURCE', s)]:
                corrected = (base[1] != y[1]) & (pred[1] == y[1])
                harmed = (base[1] == y[1]) & (pred[1] != y[1])
                surface = y[1] ^ binary_erosion(y[1], border_value=0)
                boundary = distance_transform_edt(~surface) <= 3 if surface.any() else np.zeros_like(surface)
                for name, region in [('all', np.ones_like(surface)), ('foreground', y[1]), ('background', ~y[1]), ('boundary3px', boundary)]:
                    den = int(region.sum())
                    net = 100 * (int((corrected & region).sum()) - int((harmed & region).sum())) / den if den else None
                    put(arm, 'OC_net_' + name + '_vs_' + base_name + '_pp', i, k, net)
            selected_u = [utilities[i, k, j] for j in ids]
            for sign, test in [('positive', lambda v: v > 0), ('negative', lambda v: v < 0), ('zero', lambda v: v == 0)]:
                put(arm, 'selected_' + sign + '_utility_blocks', i, k, sum(test(u) for u in selected_u))
            details.append(dict(case=i, condition=k, arm=arm, selected_blocks=len(ids), selected_pixels=pixels,
                                joint_oc_delta_pp=100 * float(da - db)))
            count += 1
    assert count == 320
    public = dict(posthoc_exploratory_only=True, e8_qualification_changed=False, new_model_calls=0,
                  arms=list(ARMS), unique_inputs=32, union_mask_evaluations=count,
                  coverage_unique_inputs_not_repeated_weight_positions=coverage,
                  absolute=[dict(arm=a, metric=m, **aggregate(v)) for (a, m), v in arrays.items()])
    comparisons = []
    for arm in ARMS:
        if arm not in ('B', 'SOURCE'):
            comparisons += [(arm, 'B'), (arm, 'SOURCE')]
    comparisons += [('CROP_POS10', a) for a in ('LOGIT_POS10', 'GEOM_POS10', 'LOGIT_MATCH', 'GEOM_MATCH', 'MARGIN_MATCH', 'CYCLE_MATCH')]
    metrics = ['OC_dice', 'ALL_dice', 'OD_dice', 'ASSD', 'HD95', 'vCDR',
               'OC_net_foreground_vs_B_pp', 'OC_net_boundary3px_vs_B_pp']
    public['contrasts'] = [dict(comparison=a + '-' + b, metric=m, **aggregate(arrays[a, m] - arrays[b, m]))
                           for a, b in comparisons for m in metrics]
    original = read(old / 'public/RESULTS.json')
    pair_audits, attribution = [], []
    for control in ('GLOBAL_Q', 'CROP_LOGIT_Q', 'GEOMETRY_ONLY'):
        d = compare_frames(frames, utilities, control, True)
        result = aggregate(d)
        reference = next(r for r in original['contrasts'] if r['comparison'] == 'CROP_Q-' + control and r['weight'] == 'UTILITY')
        assert abs(result['mean'] - reference['mean']) <= 1e-12
        content = np.average(d, axis=1, weights=[2, 1, 1, 1])
        loo = [(content.sum() - v) / 7 for v in content]
        pair_audits.append(dict(comparison='CROP_Q-' + control, weight='UTILITY', **result,
                               leave_one_content_out_min=float(min(loo)), leave_one_content_out_max=float(max(loo)),
                               leave_one_content_out_positive=int(sum(v > 0 for v in loo)), leave_one_content_out_total=8))
        contribution = {key: np.full((8, 4), np.nan) for key in ('HIGH_POS_HIGH_NEG', 'HIGH_POS_LOW_NEG', 'LOW_POS_HIGH_NEG', 'LOW_POS_LOW_NEG')}
        for frame in frames:
            i, k = frame['case'], frame['condition']
            rows = [r for r in frame['blocks'] if r['common']]
            u = np.array([utilities[i, k, r['block_id']] for r in rows])
            pos, neg = u > 0, u < 0
            wp, wn = np.abs(u[pos]), np.abs(u[neg])
            weight = wp[:, None] * wn[None, :]
            hp, hn = wp >= np.quantile(wp, .75), wn >= np.quantile(wn, .75)
            def credit(name):
                v = np.array([r['scores'][name] for r in rows])
                return (v[pos, None] > v[None, neg]) + .5 * (v[pos, None] == v[None, neg])
            diff = (credit('CROP_Q') - credit(control)) * weight / weight.sum()
            for key in contribution:
                p = hp if key.startswith('HIGH_POS') else ~hp
                n = hn if key.endswith('HIGH_NEG') else ~hn
                contribution[key][i, k] = diff[np.ix_(p, n)].sum()
            assert abs(sum(v[i, k] for v in contribution.values()) - d[i, k]) < 1e-12
        attribution += [dict(comparison='CROP_Q-' + control, pair_cell=key, **aggregate(v)) for key, v in contribution.items()]
    public['weighted_pair_audits'] = pair_audits
    public['weighted_pair_difference_contributions'] = attribution
    strata = []
    for direction, size, distance in itertools.product(('EXPAND', 'SHRINK', 'MIXED'), ('1', '2_4', 'GE5'), ('LE1', 'LE3', 'GT3')):
        subset = lambda r: (r['direction'], r['size_bucket'], r['distance_bucket']) == (direction, size, distance)
        for control, weighted in itertools.product(('CROP_LOGIT_Q', 'GEOMETRY_ONLY'), (False, True)):
            d = compare_frames(frames, utilities, control, weighted, subset)
            strata.append(dict(direction_source_to_B=direction, size_bucket=size, distance_bucket=distance,
                               comparison='CROP_Q-' + control, weight='UTILITY' if weighted else 'BLOCK', **aggregate(d)))
    assert len(strata) == 108
    public['fixed_strata_contrasts'] = strata
    save(root / 'private/FRAME_METRICS.json', details)
    save(root / 'private/CONTENT_METRICS.json', {a + ':' + m: [[float(x) if np.isfinite(x) else None for x in row] for row in v] for (a, m), v in arrays.items()})
    save(root / 'public/RESULTS.json', public)
    save(root / 'public/EVALUATION_COMPLETE.json', dict(status='COMPLETE', registered_gt_files=8,
         utility_rows=3028, input_frames=32, joint_masks=320, fixed_strata_rows=108,
         weighted_reference_checks=3, selection_exit0_and_identity_retired=True, model_calls=0,
         exact_union_dice_checked=True, synthetic_or_runtime_gt_feedback_to_selection=False))
