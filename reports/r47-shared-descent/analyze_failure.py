"""Read retained receipts/traces; export only anonymous execution aggregates."""
import json
import os
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

started = time.monotonic()
root = Path(os.environ['TASK_ROOT'])
receipts = [json.loads(p.read_text()) for p in (root/'attempts').glob('*.json')]
state = json.loads((root/'RUN_STATE.json').read_text())
groups = defaultdict(list)
fields = defaultdict(lambda: defaultdict(list))
fallbacks = defaultdict(Counter)
failed = []
for job, status in state['jobs'].items():
    condition = job.rsplit('_s', 1)[0]
    path = root/'target'/job/'visits.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    if status == 'COMPLETE':
        assert len(rows) == 1951
    groups[condition].append({'status': status, 'successful_visits': len(rows)})
    for row in rows:
        joint = row.get('diagnostics', {}).get('joint', {})
        if 'descent_fallback' in joint:
            fallbacks[condition][joint['descent_fallback']] += 1
        if 'descent_proposal_dot' in joint:
            proposal = joint['descent_proposal_dot']
            actual = joint['descent_actual_dot']
            fields[condition]['committed_minus_proposal'].append(actual-proposal)
            if proposal < 0:
                relative = (actual-proposal)/max(abs(proposal), 1e-10)
                assert relative <= 1e-5
                fields[condition]['committed_relative_gap_negative_proposal'].append(relative)
        for key, value in joint.items():
            if isinstance(value, (int, float)):
                fields[condition][key].append(float(value))
            elif key == 'filter_eigenvalues':
                for i, item in enumerate(value):
                    fields[condition][f'{key}_{i}'].append(float(item))
    if status == 'FAILED_NUMERICAL':
        receipt = next(x for x in receipts if x['phase'] == 'online_'+job)
        failed.append({'seed': int(job.rsplit('_s', 1)[1].split('_o')[0]),
                       'order': int(job.rsplit('_o', 1)[1]),
                       'attempt': receipt['attempt'],
                       'successful_visits': len(rows),
                       'failed_step': receipt['cost']['optimizer_steps'],
                       'gpu_seconds': receipt['cost']['gpu_seconds'],
                       'reason': receipt['failure']['reason'],
                       'failing_derivatives_persisted': False})

def summary(values):
    values = sorted(values)
    def percentile(fraction):
        position = (len(values)-1)*fraction
        index = int(position)
        return values[index]+(values[min(index+1, len(values)-1)]-values[index])*(position-index)
    return dict(count=len(values), mean=statistics.fmean(values), min=values[0],
                p05=percentile(.05), median=percentile(.5), p95=percentile(.95), max=values[-1])

output = {
    'scope': 'Execution diagnostics only; all retained successful steps before failure, not SEARCH efficacy.',
    'scientific_status': 'NOT_EVALUABLE',
    'attempt_status_counts': dict(Counter(x['status'] for x in receipts)),
    'condition_status_counts': {key: dict(Counter(x['status'] for x in rows)) for key, rows in groups.items()},
    'successful_visit_totals': {key: sum(x['successful_visits'] for x in rows) for key, rows in groups.items()},
    'failed_attempts': failed,
    'fallback_counts': {key: dict(value) for key, value in fallbacks.items()},
    'diagnostics': {key: {field: summary(values) for field, values in data.items()} for key, data in fields.items()},
    'gpu_worker_seconds': sum(x['cost'].get('gpu_seconds', 0) for x in receipts),
    'scored_jobs': len(list((root/'scores/final').glob('*.complete.json'))),
    'analysis_cpu_wall_seconds': time.monotonic()-started,
}
print(json.dumps(output, indent=2, allow_nan=False))
