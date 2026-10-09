"""Post-hoc position audit using already-scored R30 data; no new label reads."""
import csv
import json
import os
import statistics as st
from collections import defaultdict
from pathlib import Path


def bins(position, length):
    if not 1 <= position <= length:
        raise ValueError('invalid domain position')
    return ['ALL', 'Q'+str(min(3, (position-1)*4//length)+1)] + (['FIRST64'] if position <= 64 else [])


def run(root, out):
    pos = {}; sequences = {}; layout = []
    for order in (0, 1):
        groups = defaultdict(list)
        for visit, row in enumerate(json.loads((root/f'scorer/FULL_o{order}.json').read_text()), 1):
            groups[row['domain']].append((visit, row['image_sha256']))
        for domain, items in groups.items():
            sequences[order, domain] = [key for _, key in items]
            assert items[-1][0]-items[0][0]+1 == len(items)
            layout.append(dict(order=order, domain=domain, start=items[0][0], end=items[-1][0], arrivals=len(items)))
            for i, (_, key) in enumerate(items, 1):
                pos[order, key] = (i, len(items))
    assert all(sequences[0, d] == sequences[1, d] for o, d in sequences)
    rows = [json.loads(s) for s in (root/'scores/B.private.jsonl').read_text().splitlines()]
    index = {(r['condition'], r['seed'], r['order'], r['content']): r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        if r['condition'] != 'C_CONT' or r['role'] not in ('SEARCH', 'SEALED_REVIEW'):
            continue
        b = index['C_EPISODIC', r['seed'], r['order'], r['content']]
        for bucket in bins(*pos[r['order'], r['content']]):
            for k, channel in enumerate(('OD', 'OC')):
                groups[r['role'], r['domain'], r['seed'], r['order'], bucket, channel].append(
                    (r['metrics'][k]['dice'], b['metrics'][k]['dice']))
    result = []
    for key, values in sorted(groups.items()):
        result.append(dict(zip(('role','domain','seed','order','position_bin','channel'), key),
                           scored_contents=len(values), C_CONT=100*st.mean(v[0] for v in values),
                           C_EPISODIC=100*st.mean(v[1] for v in values),
                           delta_pp=100*st.mean(a-b for a,b in values)))
    for name, values in [('R30_POSITION_AUDIT.csv', result), ('R30_DOMAIN_LAYOUT.csv', layout)]:
        with (out/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(values[0])); writer.writeheader(); writer.writerows(values)
    return dict(status='PASS', prior_scores_only=True, within_domain_order_identical=True,
                role='post_hoc_development_analysis', independent_confirmation=False)


if __name__ == '__main__':
    assert bins(1, 800) == ['ALL','Q1','FIRST64'] and bins(800,800) == ['ALL','Q4']
    root = Path(os.environ['AUDIT_INPUT']); out = Path(os.environ['AUDIT_OUTPUT'])
    print(json.dumps(run(root, out)))
