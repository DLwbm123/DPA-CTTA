"""Export anonymous derivative aggregates from private numerical profile traces."""
import json
import math
import os
import statistics
from pathlib import Path

root = Path(os.environ['TASK_ROOT'])
profiles = [json.loads(p.read_text()) for p in (root/'profiles').glob('*.json')]
assert len(profiles) == 8 and all(p['status'] == 'PASS' and p['arrivals'] == 64 for p in profiles)
records = []
all_arrivals = 0
for path in (root/'private/numerical_steps').glob('*.jsonl'):
    steps = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(steps) == 64 and [p['visit'] for p in steps] == list(range(1, 65))
    all_arrivals += len(steps)
    records.extend(step['derivatives'] for step in steps if 'descent_proposal_dot' in step['derivatives'])
assert all_arrivals == 512 and len(records) == 256

def summarize(values):
    values = sorted(values)
    position = .95*(len(values)-1)
    index = int(position)
    p95 = values[index]+(values[min(index+1, len(values)-1)]-values[index])*(position-index)
    return dict(count=len(values), mean=statistics.fmean(values), min=values[0],
                median=statistics.median(values), p95=p95, max=values[-1])

fields = {}
for key in sorted({key for row in records for key in row}):
    values = [row[key] for row in records if type(row.get(key)) in (int, float)]
    if values:
        assert all(math.isfinite(value) for value in values)
        fields[key] = summarize(values)
negative = [row for row in records if row['descent_proposal_dot'] < 0]
violations = sum(row['descent_actual_dot']-row['descent_proposal_dot'] > row['descent_tolerance'] for row in negative)
repairs = [row for row in records if row.get('descent_commit_repaired')]
assert violations == 0
output = dict(status='PASS_BOUNDED_PROFILE_ONLY', formal_experiment_started=False,
              scope='Two authorized64-image prefixes on both devices; repeats are not independent subjects.',
              all_condition_arrivals=all_arrivals, repaired_arrivals=len(records),
              negative_native_proposals=len(negative), final_violations=violations,
              commit_repairs=len(repairs), commit_repair_fraction=len(repairs)/len(records),
              non_descent_fallbacks=sum(row.get('descent_fallback') == 'non_descent_proposal' for row in records),
              constraint_triggers=sum(row.get('descent_constraint_active', False) for row in records),
              repair_gap_tolerance_ratio=summarize([row['descent_candidate_committed_gap']/row['descent_tolerance'] for row in repairs]),
              repaired_returned_filter_gap_tolerance_ratio=summarize([(row['descent_filtered_dot']-row['descent_proposal_dot'])/row['descent_tolerance'] for row in repairs]),
              derivatives=fields)
(root/'public/NUMERICAL_AGGREGATES.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
print(json.dumps({key: value for key, value in output.items() if key not in ('derivatives', 'repair_gap_tolerance_ratio')}))
