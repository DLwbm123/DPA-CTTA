"""Summarize completed private scalar traces without exporting content rows."""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


root = Path(os.environ['TASK_ROOT'])
assert json.loads((root / 'RUN_STATE.json').read_text())['status'] == 'COMPLETE'
assert not any(json.loads(p.read_text()).get('active') for p in (root / 'processes').glob('*.json'))
groups = defaultdict(lambda: defaultdict(list))
ranks = defaultdict(Counter)
counts = Counter()
for path in (root / 'scores/final').glob('*.private.jsonl'):
    for line in path.open():
        row = json.loads(line)
        condition = row['condition']
        if condition not in ('CW_LSO_JS', 'CW_LSO_JS_SHARED') or row['role'] != 'SEARCH':
            continue
        counts[condition] += 1
        joint = row['diagnostics']['joint']
        ranks[condition][joint['rank']] += 1
        for key, value in joint.items():
            if isinstance(value, list):
                for i, scalar in enumerate(value):
                    groups[condition][f'{key}.{i}'].append(scalar)
            else:
                groups[condition][key].append(value)
        proposal = joint['proposal_metric_norm']
        filtered = joint['filtered_metric_norm']
        assert proposal > 0
        groups[condition]['filtered_to_proposal_metric_norm'].append(filtered / proposal)
        groups[condition]['contraction_violation'].append(int(filtered > proposal * (1 + 1e-6)))
        for channel, value in zip(('OD', 'OC'), row['diagnostics']['structure_evidence_mean']):
            groups[condition][f'structure_evidence.{channel}'].append(value)
assert counts == {'CW_LSO_JS': 6102, 'CW_LSO_JS_SHARED': 6102}
result = {}
for condition, metrics in groups.items():
    result[condition] = {'SEARCH_steps': counts[condition], 'rank_counts': dict(ranks[condition]), 'metrics': {}}
    for key, values in metrics.items():
        x = np.asarray(values, dtype=float)
        assert np.isfinite(x).all()
        result[condition]['metrics'][key] = dict(count=len(x), mean=float(x.mean()),
            minimum=float(x.min()), p05=float(np.quantile(x, .05)),
            median=float(np.median(x)), p95=float(np.quantile(x, .95)), maximum=float(x.max()))
(root / 'public/JOINT_MECHANISM_SUMMARY.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result))
