"""Content-level paired aggregate analysis; no model calls."""
import json, sys
from pathlib import Path
import numpy as np

R = Path(sys.argv[1])
results = []
new_events = []
for run in sorted((R / 'runs').iterdir()):
    p = run / 'detailed.private.json'
    if not p.exists(): continue
    rows = json.loads(p.read_text())
    assert len(rows) == 80 and len({(r['content'], r['condition']) for r in rows}) == 80, 'Incomplete paired readout'
    new_events.append(sum((r['COUPLED3']['empty_od'] and not r['ZERO']['empty_od']) or (r['COUPLED3']['empty_oc'] and not r['ZERO']['empty_oc']) or (r['COUPLED3']['cup_outside'] > .01 and r['ZERO']['cup_outside'] <= .01) for r in rows))
    seed = {'seed': int(run.name), 'comparisons': {}, 'conditions': {}}
    seed['completion'] = json.loads((run / 'completion.json').read_text())
    seed['trace_diagnostics'] = {}
    for arm in ['FIXED3', 'DIAGONAL3', 'COUPLED3']:
        traces = [r[arm + '_trace'] for r in rows]
        def final_j(trace):
            accepted = [t for t in trace if t['accepted']]
            return accepted[-1]['after'] if accepted else trace[0]['before']
        jd = [final_j(t) - t[0]['before'] for t in traces]
        seed['trace_diagnostics'][arm] = {
            'mean_full_J_delta': float(np.mean(jd)),
            'J_decreased_but_dice_worse_conditions': sum(j < 0 and r[arm]['dice'] < r['ZERO']['dice'] for j, r in zip(jd, rows)),
            'accepted_by_step': [sum(any(t['step'] == k and t['accepted'] for t in trace) for trace in traces) for k in range(3)],
            'conditions': len(rows)}
    for a, b in [('COUPLED3', 'ZERO'), ('FIXED3', 'ZERO'), ('COUPLED3', 'FIXED3'), ('COUPLED3', 'DIAGONAL3'), ('DIAGONAL3', 'ZERO'), ('DIAGONAL3', 'FIXED3')]:
        name = a + '-' + b
        pair = {}
        for key in ['dice', 'od', 'oc', 'vcdr_error']:
            contents = [np.mean([r[a][key] - r[b][key] for r in rows if r['content'] == i]) for i in sorted({r['content'] for r in rows})]
            pair[key] = {'mean': float(np.mean(contents)), 'q05': float(np.quantile(contents, .05)), 'minimum': float(min(contents))}
        pair['clean'] = float(np.mean([r[a]['dice'] - r[b]['dice'] for r in rows if r['condition'] == 'clean']))
        dd = [r[a]['dice'] - r[b]['dice'] for r in rows]
        pair['condition_win_tie_loss'] = [sum(d > 0 for d in dd), sum(d == 0 for d in dd), sum(d < 0 for d in dd)]
        pair['finite_boundaries'] = {}
        for key in ['od_hd95', 'oc_hd95', 'od_assd', 'oc_assd']:
            valid = [r for r in rows if r[a][key] is not None and r[b][key] is not None]
            by_content = [np.mean([r[a][key] - r[b][key] for r in valid if r['content'] == i]) for i in sorted({r['content'] for r in valid})]
            pair['finite_boundaries'][key] = {'paired_conditions': len(valid), 'total': len(rows), 'content_mean_delta': float(np.mean(by_content)) if by_content else None}
        seed['comparisons'][name] = pair
    for arm in ['ZERO', 'FIXED3', 'DIAGONAL3', 'COUPLED3']:
        seed[arm] = {'dice': float(np.mean([r[arm]['dice'] for r in rows])),
                     'empty_od': sum(r[arm]['empty_od'] for r in rows), 'empty_oc': sum(r[arm]['empty_oc'] for r in rows),
                     'cup_outside_gt1pct': sum(r[arm]['cup_outside'] > .01 for r in rows)}
        seed['conditions'][arm] = {kind: float(np.mean([r[arm]['dice'] for r in rows if r['condition'] == kind])) for kind in ['clean', 'gamma', 'color', 'blur']}
    results.append(seed)
gate = False
mechanism = False
safety = False
coupling = False
if len(results) == 2:
    ds = [r['comparisons']['COUPLED3-ZERO'] for r in results]
    gate = all(d['dice']['mean'] > 0 and d['od']['mean'] >= 0 and d['oc']['mean'] >= 0 and d['clean'] >= -.3 and d['dice']['q05'] >= -2 for d in ds) and np.mean([d['dice']['mean'] for d in ds]) >= .2
    md = [r['comparisons']['COUPLED3-FIXED3']['dice']['mean'] for r in results]
    mechanism = all(v > 0 for v in md) and np.mean(md) >= .15
    cd = [r['comparisons']['COUPLED3-DIAGONAL3']['dice']['mean'] for r in results]
    coupling = all(v > 0 for v in cd) and np.mean(cd) >= .05
    safety = all(n == 0 for n in new_events)
report = {'scope': 'Repeated source development, not independent patients or real target validation; fixed generative base plus additional source-fit metric training',
          'results': results, 'new_safety_events_by_seed': new_events, 'adaptation_gate': bool(gate), 'metric_gate': bool(mechanism), 'coupling_gate': bool(coupling), 'safety_count_gate': bool(safety),
          'finished': json.loads((R / 'TRAINING_FINISHED.json').read_text()), 'exit': json.loads((R / 'EXIT.json').read_text())}
assert report['finished']['status'] == 'COMPLETE' and report['exit']['code'] == 0 and len(results) == 2, 'Run not complete'
report['prior_failed_synthetic_attempts'] = [json.loads(p.read_text()) for p in sorted(R.glob('FAILURE_attempt*.json'))]
report['synthetic_accounting'] = json.loads((R / 'SELF_CHECK.json').read_text())
report['real_calls_excluding_synthetic'] = {k: v - report['synthetic_accounting']['counts'][k]
        for k, v in report['finished']['counts'].items()}
report['all_attempts_calls_including_synthetic'] = {k: v + sum(a['counts'][k] for a in report['prior_failed_synthetic_attempts'])
        for k, v in report['finished']['counts'].items()}
runtime = json.loads((R / 'RUNTIME_RECEIPT.json').read_text())
elapsed = report['exit']['ended_unix'] - runtime['origin_unix']
report['budget_audit'] = {'original_clock_wall_seconds_through_exit_receipt': elapsed,
        'recorded_seconds_before_exit_storage_scan': report['exit']['seconds'],
        'within_time_budget': elapsed < runtime['hard_seconds'],
        'exit_storage_below_soft_limit': report['exit']['allocated_bytes'] < runtime['soft_stop_bytes'],
        'exit_storage_below_hard_limit': report['exit']['allocated_bytes'] < runtime['hard_storage_bytes']}
(R / 'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'seeds_evaluated': len(results), 'adaptation_gate': bool(gate), 'metric_gate': bool(mechanism), 'coupling_gate': bool(coupling), 'safety_count_gate': bool(safety)}))
