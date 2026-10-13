"""Audit terminal receipts and export aggregates only; private inputs stay private."""
import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

root = Path(os.environ['TASK_ROOT'])
read = lambda path: json.loads(path.read_text())
state = read(root / 'RUN_STATE.json')
jobs = read(root / 'FINAL_JOBS.json')
receipts = [read(p) for p in (root / 'attempts').glob('*.json')]
processes = [read(p) for p in (root / 'processes').glob('*.json')]
assert state['status'] == 'COMPLETE' and len(jobs) == 30
assert len(receipts) == len(processes) == 65
assert all(p['status'] == 'COMPLETE' and p['attempt'] == 0 for p in receipts)
assert all(not p['active'] and p['exit_code'] == 0 for p in processes)
watch = read(root / 'watchdog.json')
pids = [p['pid'] for p in processes] + [watch['pid'], watch['supervisor_pid']]
assert not subprocess.run(['ps', '-p', ','.join(map(str, pids)), '-o', 'pid='],
                          capture_output=True, text=True).stdout.strip()
online = [p for p in receipts if p['phase'].startswith('online_')]
score = [p for p in receipts if p['phase'].startswith('score_final_')]
assert {p['phase'][7:] for p in online} == {j['id'] for j in jobs}
assert {p['phase'][12:] for p in score} == {j['id'] for j in jobs}
assert len(online) == len(score) == 30
assert max(p['ended'] for p in online) < min(p['started'] for p in score)
for p in online:
    assert p['result']['visits'] == 1951
    joint = '_JS_SHARED' in p['phase']
    native_w = p['phase'].startswith('online_W_')
    assert p['cost']['model_forwards'] == 1951 * (10 if joint else 9 if native_w else 8)
    assert p['cost']['backward_calls'] == 1951 * (2 if native_w else 1)
    assert p['cost']['optimizer_steps'] == 1951
    assert p['cost'].get('vjp_calls', 0) == (7804 if joint else 0)
assert all(p['result']['independent_CPU'] and p['result']['rows'] == 1951 for p in score)
ledger = read(root / 'RESOURCE_LEDGER.json')
assert abs(sum(p['cost'].get('gpu_seconds', 0) for p in receipts)
           - ledger['gpu_worker_seconds']) < 1e-6
score_paths = list((root / 'scores/final').glob('*.private.jsonl'))
assert len(score_paths) == 30
groups = defaultdict(lambda: defaultdict(list))
fallbacks = defaultdict(Counter)
counts = Counter()
scope_counts = Counter()
rows_total = 0

def add(condition, scope, diagnostics):
    group = condition + ':' + scope
    scope_counts[group] += 1
    joint = diagnostics['joint']
    fallbacks[group][joint.get('descent_fallback', 'none')] += 1
    for key, value in joint.items():
        values = value if isinstance(value, list) else [value]
        for i, scalar in enumerate(values):
            if isinstance(scalar, (int, float, bool)):
                assert np.isfinite(scalar)
                name = f'{key}.{i}' if isinstance(value, list) else key
                groups[group][name].append(float(scalar))
    if 'filtered_metric_norm' in joint:
        groups[group]['candidate_norm_ratio'].append(
            joint['filtered_metric_norm'] / joint['proposal_metric_norm'])
    for channel, value in zip(('OD', 'OC'), diagnostics['structure_evidence_mean']):
        groups[group]['structure_evidence.' + channel].append(value)
    groups[group]['BN_gradient_norm'].append(diagnostics['BN_gradient_norm'])
    groups[group]['BN_update_norm'].append(diagnostics['BN_update_norm'])
    for i, regions in enumerate(diagnostics['logit_gradient_regions']):
        for j, value in enumerate(regions):
            groups[group][f'logit_gradient_regions.{i}.{j}'].append(value)
    if 'descent_proposal_dot' in joint:
        proposal = joint['descent_proposal_dot']
        tolerance = joint['descent_tolerance']
        assert tolerance == 1e-5 * max(abs(proposal), 1e-10)
        actual = joint['descent_actual_dot']
        assert np.isfinite(actual)
        if proposal < 0:
            assert actual - proposal <= tolerance
            groups[group]['final_gap_tolerance_ratio'].append((actual - proposal) / tolerance)
        if joint.get('descent_commit_repaired'):
            assert actual == proposal
            groups[group]['repair_candidate_gap_tolerance_ratio'].append(
                joint['descent_candidate_committed_gap'] / tolerance)
        if 'descent_unconstrained_gap' in joint:
            groups[group]['unconstrained_dot'].append(proposal + joint['descent_unconstrained_gap'])

for path in score_paths:
    rows = [json.loads(line) for line in path.open()]
    assert len(rows) == 1951 and {r['visit'] for r in rows} == set(range(1, 1952))
    rows_total += len(rows)
    for row in rows:
        condition = row['condition']
        if condition not in ('CW_LSO_JS_SHARED', 'CW_LSO_JS_SHARED_DESCENT'):
            continue
        if condition.endswith('_DESCENT'):
            assert 'descent_proposal_dot' in row['diagnostics']['joint']
        counts[condition, row['role']] += 1
        add(condition, row['role'], row['diagnostics'])
        if row['role'] == 'SEARCH':
            add(condition, f"SEARCH/order{row['order']}/{row['domain']}", row['diagnostics'])
assert rows_total == 58530
assert all(counts[c, 'SEARCH'] == 6102 and sum(n for (cc, _), n in counts.items() if cc == c) == 11706
           for c in ('CW_LSO_JS_SHARED', 'CW_LSO_JS_SHARED_DESCENT'))

def summarize(values):
    x = np.asarray(values, dtype=float)
    return dict(count=len(x), mean=float(x.mean()), minimum=float(x.min()),
                p05=float(np.quantile(x, .05)), median=float(np.median(x)),
                p95=float(np.quantile(x, .95)), maximum=float(x.max()))

result = {g: dict(steps=scope_counts[g], fallback_reasons=dict(fallbacks[g]),
                 metrics={k: summarize(v) for k, v in metrics.items()})
          for g, metrics in groups.items()}
audit = dict(status='COMPLETE', all_online_score_watch_supervisor_retired=True,
             unique_formal_jobs=30, independent_CPU_scores=30, scored_rows=rows_total,
             original_tolerance_unchanged=True, final_descent_violations_all_roles=0,
             attempts=65, retries=0, all_attempts_charged=True,
             online_before_scoring=True, GPU_worker_hours=ledger['gpu_worker_seconds']/3600,
             CPU_scoring_seconds=ledger['cpu_worker_seconds'],
             CPU_preparation_seconds=24.81283211708069,
             wall_seconds=ledger['wall_seconds'], allowed_physical_GPU_ids=[0, 1])
for name, output in [('COMPLETION_VERIFICATION.json', audit), ('JOINT_NUMERICAL_SUMMARY.json', result)]:
    (root / 'public' / name).write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')
print(json.dumps(audit))
