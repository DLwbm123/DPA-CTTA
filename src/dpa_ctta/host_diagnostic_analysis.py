"""Frozen descriptive paired analysis. Private identities never enter public output."""
import json
import math
from pathlib import Path

import numpy as np

from .source_pilot import SEGMENTS, validate_arm


def distribution(values):
    a = np.asarray(values, dtype=np.float64)
    if not np.isfinite(a).all():
        raise ValueError('nonfinite descriptive input')
    if not len(a):
        return dict(n=0, mean=None, median=None, p10=None, p90=None, positive=0,
                    zero=0, negative=0, minimum=None, worst_decile_mean=None, tail_count=0)
    tail = max(1, math.ceil(len(a) * .1))
    return dict(n=len(a), mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .1, method='linear')),
                p90=float(np.quantile(a, .9, method='linear')),
                positive=int((a > 0).sum()), zero=int((a == 0).sum()), negative=int((a < 0).sum()),
                minimum=float(a.min()), worst_decile_mean=float(np.sort(a)[:tail].mean()), tail_count=tail)


def paired(left, right):
    if len(left) != len(right) or not left:
        raise ValueError('paired coverage mismatch')
    both = [(a, b) for a, b in zip(left, right) if a['assd'] is not None and b['assd'] is not None]
    return dict(dice_delta_pp=distribution([100 * (a['dice'] - b['dice']) for a, b in zip(left, right)]),
                assd_delta_px=distribution([a['assd'] - b['assd'] for a, b in both]),
                assd_common_valid=len(both), assd_not_jointly_defined=len(left)-len(both),
                assd_left_undefined=sum(a['assd'] is None for a in left),
                assd_right_undefined=sum(b['assd'] is None for b in right))


def channels(rows, channel):
    if channel == 'macro':
        return [dict(dice=float(np.mean([m['dice'] for m in r['metrics']])), assd=None) for r in rows]
    return [next(m for m in r['metrics'] if m['channel'] == channel) for r in rows]


def old_analysis(registration, directory):
    public, private = {}, {}
    for task, reg in registration['tasks'].items():
        arms = {a: [json.loads(line) for line in (Path(directory)/f'{task}_{a}.jsonl').read_text().splitlines()]
                for a in ('N', 'A', 'B', 'C')}
        for arm, rows in arms.items():
            validate_arm(rows, reg['expected_visits'], task, arm)
        names = ['OD', 'OC', 'macro'] if task == 'fundus' else ['polyp']
        output = public[task] = dict(comparisons={}, within_content_segment_changes={})
        grouped = private[task] = {}
        n = len(reg['selected']['query'])
        for left, right in [('A','N'), ('B','N'), ('C','N'), ('B','A'), ('C','B')]:
            comparisons = output['comparisons'][left+'-'+right] = {}
            for section in ('all', *SEGMENTS, 'clean_1_5', 'clean_6_16', 'clean_17_plus'):
                def select(rows):
                    if section == 'all': return rows
                    if section in SEGMENTS: return [r for r in rows if r['segment'] == section]
                    lo, hi = {'clean_1_5':(1,5), 'clean_6_16':(6,16), 'clean_17_plus':(17,n)}[section]
                    return [r for r in rows if r['segment']=='clean' and lo<=r['visit']<=hi]
                a, b = select(arms[left]), select(arms[right])
                comparisons[section] = {c: paired(channels(a,c),channels(b,c)) for c in names}
        for arm, rows in arms.items():
            changes = output['within_content_segment_changes'][arm] = {}
            grouped[arm] = []
            for i, registered in enumerate(reg['selected']['query']):
                records = [rows[i + j*n] for j in range(4)]
                if any(r['group_id']!=registered['group_id'] for r in records):
                    raise ValueError('cross-segment identity mismatch')
                grouped[arm].append(dict(group_id=registered['group_id'], sample_id=registered['sample_id'],
                    segments={r['segment']:r['metrics'] for r in records}))
            # Pair within the same content first, then aggregate. No independent-patient claim.
            for j, segment in enumerate(SEGMENTS[1:], 1):
                changes[segment+'-clean'] = {c: paired(channels(rows[j*n:(j+1)*n],c),channels(rows[:n],c)) for c in names}
        output['formal_records'] = sum(len(rows) for rows in arms.values())
    return public, private
