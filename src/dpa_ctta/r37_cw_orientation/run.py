"""Reuse the fixed-matrix runtime; change only registration and reporting."""
import csv
import json
from pathlib import Path
import re
import statistics
from ..r36_incremental_modules import run as retained
from ..r20_model_only_search.report import bootstrap_pair, csvout, public as common_public
from ..r10_12h_core.run import read, save

ID = 'R37_CW_ORIENTATION'
original_report = retained.report


def public(value):
    value = common_public(value)
    excluded = {'identity', 'prediction_sha256', 'trace_sha256', 'scalar_sha256',
                'rows_sha256', 'context_sha256', 'image_sha256', 'patient_id'}
    if isinstance(value, dict):
        return {k: public(v) for k, v in value.items() if k not in excluded}
    if isinstance(value, list):
        return [public(v) for v in value]
    if isinstance(value, str):
        return re.sub(r'/remote-home/[^\s\"\']+', '[PRIVATE_PATH]', value)
    return value


def passes(pair, minimum):
    return (pair.get('status') == 'COMPLETE' and float(pair['delta_pp']) >= minimum
            and min(json.loads(pair['order_delta_pp'])) > 0
            and int(pair['positive_trajectories']) >= 5
            and float(pair['imageweighted_delta_pp']) >= 0
            and float(pair['worst_seed_averaged_cell_pp']) >= -2)


def comparison_passes(pair, attribution_control, noninferiority_margin=None):
    if pair['baseline'] == attribution_control and noninferiority_margin is not None:
        return pair['status'] == 'COMPLETE' and float(pair['delta_pp']) >= -noninferiority_margin
    return passes(pair, .1 if pair['baseline'] == attribution_control else .3)


def report(c, state):
    primary = c.get('primary', 'CW_LSO')
    attribution_control = c.get('attribution_control', 'CW')
    original_report(c, state)
    root = Path(c['output_root']); out = root/'public'
    jobs = read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    done = [j for j in jobs if (root/'scores/final'/(j['id']+'.complete.json')).exists()]
    rows = retained.scoring.load_rows(root, 'FINAL', done, final=True)
    attribution, intervals, negatives = bootstrap_pair(rows, primary, attribution_control, c['seeds'], role='SEARCH')
    for name, extra in [('PAIRED_SUMMARY.csv', [attribution]),
                        ('CONTENT_BOOTSTRAP_CI.csv', intervals),
                        ('ALL_NEGATIVE_CELLS.csv', negatives)]:
        with (out/name).open() as f: existing = list(csv.DictReader(f))
        csvout(out/name, existing+extra)
    with (out/'PAIRED_SUMMARY.csv').open() as f: pairs = list(csv.DictReader(f))
    tests = [p for p in pairs if p.get('candidate') == primary and p.get('role') == 'SEARCH']
    success = (state['status'] == 'COMPLETE' and len(tests) == 3
               and all(comparison_passes(p, attribution_control, c.get('attribution_noninferiority_margin')) for p in tests))
    save(out/'FINAL_DECISION.json', dict(primary=primary, strong_development_signal=success,
         comparisons=public(tests), review_descriptive_only=True,
         independent_generalization=False, automatic_retuning=False))
    with (out/'FULL_RESULTS.csv').open() as f: main = list(csv.DictReader(f))
    intro = c.get('report_intro', 'R36 found a consistent but small W_LSO improvement over W; all modules still lost to C. '
             'R20 previously showed a one-seed C-host W transfer signal. This follow-up tests that '
             'transfer across three seeds/two orders, with CW isolating the added LSO contribution. '
             'The host transfer also removes the native G entropy/perturbation and dynamic learning '
             'rate and uses fixed C learning rate; it is a bundled host change, not causal proof.')
    lines = [c.get('report_heading', '# R37: fixed C+W orientation transfer'), '',
             f"Status: **{state['status']}**. Frozen primary: {primary}.", '', intro, '',
             '| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |',
             '|---|---:|---:|']
    for candidate in c['candidates']:
        means = []
        for role in ('SEARCH', 'SEALED_REVIEW'):
            values = [float(r['macro_Dice_percent']) for r in main
                      if r['condition'] == candidate['id'] and r['role'] == role]
            means.append(f'{statistics.mean(values):.4f}' if values else 'NA')
        lines.append(f"| {candidate['id']} | {means[0]} | {means[1]} |")
    ledger = read(out/'RESOURCE_LEDGER.json')
    gate = c.get('gate_description', 'Stage success requires CW_LSO SEARCH gains >=0.3pp against both C and W '
              'and >=0.1pp against CW; each comparison must pass both orders, >=5/6 positive '
              'trajectories, nonnegative imageweighted gain and worst seed-mean cell >=-2pp.')
    lines += ['', gate+' '
              f'Frozen primary passed: {success}. All negative cells remain reported.', '',
              f"GPU-worker {ledger['gpu_worker_seconds']/3600:.3f}/48h; CPU-worker "
              f"{ledger['cpu_worker_seconds']/3600:.3f}h. Original per-round T0 and attempts retained.", '',
              'Exactly four frozen conditions, three seeds, two orders, 1,951 arrivals each. '
              'No online labels, source images, source retraining, RL, extra views, output fusion '
              'or scientific retries. Separate CPU masks only after all online workers retire.', '',
              'SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; '
              'REVIEW is descriptive only. Gains do not establish independent generalization. '
              'Dice, empty-mask convention and content-cluster bootstrap are unchanged; '
              'patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.', '',
              'Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, '
              'trace and scalar provenance hashes are omitted from public receipts; originals remain private.']
    if state.get('reason'): lines += ['', 'Stopped reason: '+public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    retained.ID = ID
    retained.report = report
    retained.public = public
    retained.main()


if __name__ == '__main__': main()
