"""Content-level paired aggregate analysis; no model calls."""
import json, sys
from pathlib import Path
import numpy as np

R = Path(sys.argv[1])
results = []
for run in sorted((R / 'runs').iterdir()):
    p = run / 'final_readout' / 'detailed.private.json'
    if not p.exists(): continue
    rows = json.loads(p.read_text())
    seed = {'seed': int(run.name), 'comparisons': {}, 'conditions': {}}
    for a, b in [('CONTINUAL', 'ZERO'), ('INDEPENDENT', 'ZERO'), ('CONTINUAL', 'INDEPENDENT')]:
        name = a + '-' + b
        pair = {}
        for key in ['dice', 'od', 'oc', 'vcdr_error']:
            contents = [np.mean([r[a][key] - r[b][key] for r in rows if r['content'] == i]) for i in sorted({r['content'] for r in rows})]
            pair[key] = {'mean': float(np.mean(contents)), 'q05': float(np.quantile(contents, .05)), 'minimum': float(min(contents))}
        pair['clean'] = float(np.mean([r[a]['dice'] - r[b]['dice'] for r in rows if r['condition'] == 'clean']))
        pair['finite_boundaries'] = {}
        for key in ['od_hd95', 'oc_hd95', 'od_assd', 'oc_assd']:
            valid = [r for r in rows if r[a][key] is not None and r[b][key] is not None]
            by_content = [np.mean([r[a][key] - r[b][key] for r in valid if r['content'] == i]) for i in sorted({r['content'] for r in valid})]
            pair['finite_boundaries'][key] = {'paired_conditions': len(valid), 'total': len(rows), 'content_mean_delta': float(np.mean(by_content)) if by_content else None}
        seed['comparisons'][name] = pair
    for arm in ['ZERO', 'INDEPENDENT', 'CONTINUAL']:
        seed[arm] = {'dice': float(np.mean([r[arm]['dice'] for r in rows])),
                     'empty_od': sum(r[arm]['empty_od'] for r in rows), 'empty_oc': sum(r[arm]['empty_oc'] for r in rows),
                     'cup_outside_gt1pct': sum(r[arm]['cup_outside'] > .01 for r in rows)}
        seed['conditions'][arm] = {kind: float(np.mean([r[arm]['dice'] for r in rows if r['condition'] == kind])) for kind in ['clean', 'gamma', 'color', 'blur']}
    results.append(seed)
gate = False
memory = False
if len(results) == 2:
    deltas = [r['comparisons']['CONTINUAL-ZERO'] for r in results]
    gate = all(d['dice']['mean'] > 0 and d['oc']['mean'] >= 0 and d['clean'] >= -.3 and d['dice']['q05'] >= -2 for d in deltas) and np.mean([d['dice']['mean'] for d in deltas]) >= .2
    md = [r['comparisons']['CONTINUAL-INDEPENDENT']['dice']['mean'] for r in results]
    memory = all(v > 0 for v in md) and np.mean(md) >= .05
report = {'scope': 'Repeated source development, not independent patients or real target domain validation',
          'results': results, 'adaptation_gate': bool(gate), 'persistent_state_gate': bool(memory),
          'finished': json.loads((R / 'TRAINING_FINISHED.json').read_text()), 'exit': json.loads((R / 'EXIT.json').read_text())}
(R / 'REPORT.json').write_text(json.dumps(report, indent=2, allow_nan=False))
print(json.dumps({'seeds_evaluated': len(results), 'adaptation_gate': bool(gate), 'persistent_state_gate': bool(memory)}))
