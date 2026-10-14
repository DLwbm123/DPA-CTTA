"""Five fixed arms on the existing guarded runtime; paired factorial reporting."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

from ..joint_sensitivity.factorial import Host
from ..r36_incremental_modules import run as retained
from ..r37_cw_orientation.run import public, passes
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout, diagnostics
from ..r10_12h_core.run import read, save

ID = 'R49_JOINT_LSO_FACTORIAL'
CONTRASTS = [('C_JS_SHARED', 'C'), ('C_LSO', 'C'),
             ('C_LSO_JS_SHARED', 'C_LSO'), ('C_LSO_JS_SHARED', 'C_JS_SHARED'),
             ('C_JS_SHARED', 'CW_LSO'), ('C_LSO_JS_SHARED', 'CW_LSO'), ('CW_LSO', 'C')]
ARMS = ('C', 'C_LSO', 'C_JS_SHARED', 'C_LSO_JS_SHARED')


def interaction_rows(rows):
    """Scoring-only signed contrasts; never predictions or an executable candidate."""
    groups = defaultdict(dict)
    for row in rows:
        if row['condition'] in ARMS:
            key = row['seed'], row['order'], row['content']
            if row['condition'] in groups[key]:
                raise ValueError('duplicate factorial observation')
            groups[key][row['condition']] = row
    if not groups or any(set(g) != set(ARMS) for g in groups.values()):
        return []
    out = []
    for group in groups.values():
        base = group['C']
        for row in group.values():
            if (row['role'], row['domain'], [m['channel'] for m in row['metrics']]) != (
                    base['role'], base['domain'], [m['channel'] for m in base['metrics']]):
                raise ValueError('factorial role/domain/channel mismatch')
        for label, a, b in [('J_EFFECT_WITH_LSO', 'C_LSO_JS_SHARED', 'C_LSO'),
                            ('J_EFFECT_WITHOUT_LSO', 'C_JS_SHARED', 'C')]:
            out.append(dict(condition=label, seed=base['seed'], order=base['order'],
                            content=base['content'], domain=base['domain'], role=base['role'],
                            metrics=[dict(channel=x['channel'], dice=x['dice'] - y['dice'])
                                     for x, y in zip(group[a]['metrics'], group[b]['metrics'])]))
    return out


def signal(pair):
    normalized = dict(pair)
    if isinstance(normalized.get('order_delta_pp'), list):
        normalized['order_delta_pp'] = json.dumps(normalized['order_delta_pp'])
    return passes(normalized, .1) and float(pair['CI_low_pp']) > 0


def interpretation(primary, conditional, interaction):
    if any(x.get('status') != 'COMPLETE' for x in (primary, conditional, interaction)):
        return 'NOT_EVALUABLE'
    if not signal(primary):
        return 'NO_REGISTERED_STANDALONE_SIGNAL'
    if float(conditional['CI_high_pp']) < 0:
        return 'INTERFERENCE_COMPATIBLE'
    if float(interaction['CI_high_pp']) < 0:
        if float(conditional['CI_low_pp']) >= -.05 and float(conditional['CI_high_pp']) <= .05:
            return 'REDUNDANCY_COMPATIBLE'
        return 'DIMINISHING_RETURNS_COMPATIBLE'
    if signal(conditional):
        return 'J_BENEFIT_IN_BOTH_BACKGROUNDS'
    return 'INTERACTION_INCONCLUSIVE'


def report(c, state):
    root = Path(c['output_root']); out = root / 'public'
    resources = retained.base.ledger(c, state)
    jobs = read(root / 'FINAL_JOBS.json') if (root / 'FINAL_JOBS.json').exists() else []
    done = [j for j in jobs if (root / 'scores/final' / (j['id'] + '.complete.json')).exists()]
    rows = retained.scoring.load_rows(root, 'FINAL', done, final=True)
    main, domains = aggregate(rows)
    csvout(out / 'FULL_RESULTS.csv', main); csvout(out / 'DOMAIN_CHANNEL.csv', domains)
    csvout(out / 'MECHANISM_DIAGNOSTICS.csv', diagnostics(rows))
    csvout(out / 'COST.csv', [dict(phase=a['phase'], attempt=a['attempt'], status=a['status'],
                                 wall_seconds=a['wall_seconds'], **a['cost']) for a in resources['attempts']])
    pairs, intervals, negatives, interactions, interaction_intervals, interaction_negatives = [], [], [], [], [], []
    contrast_rows = interaction_rows(rows)
    for role in ('SEARCH', 'SEALED_REVIEW'):
        for a, b in CONTRASTS:
            pair, ci, neg = bootstrap_pair(rows, a, b, c['seeds'], role=role)
            pairs.append(pair); intervals.extend(ci); negatives.extend(neg)
        pair, ci, neg = bootstrap_pair(contrast_rows, 'J_EFFECT_WITH_LSO', 'J_EFFECT_WITHOUT_LSO', c['seeds'], role=role)
        interactions.append(pair); interaction_intervals.extend(ci); interaction_negatives.extend(neg)
    for name, values in [('PAIRED_SUMMARY.csv', pairs), ('CONTENT_BOOTSTRAP_CI.csv', intervals),
                         ('ALL_NEGATIVE_CELLS.csv', negatives), ('INTERACTION_SUMMARY.csv', interactions),
                         ('INTERACTION_CI.csv', interaction_intervals), ('INTERACTION_NEGATIVE_CELLS.csv', interaction_negatives)]:
        csvout(out / name, values)
    primary = next(p for p in pairs if p['candidate'] == 'C_JS_SHARED' and p['baseline'] == 'C' and p['role'] == 'SEARCH')
    conditional = next(p for p in pairs if p['candidate'] == 'C_LSO_JS_SHARED' and p['baseline'] == 'C_LSO' and p['role'] == 'SEARCH')
    success = state['status'] == 'COMPLETE' and signal(primary)
    decision = dict(primary='C_JS_SHARED', primary_comparison='C_JS_SHARED minus C',
                    strong_development_signal=success, interpretation=interpretation(primary, conditional, interactions[0])
                    if state['status'] == 'COMPLETE' else 'NOT_EVALUABLE',
                    review_descriptive_only=True, independent_generalization=False, automatic_followup=False)
    save(out / 'FINAL_DECISION.json', decision)
    for name, value in [('RUN_STATE.json', state), ('RESOURCE_LEDGER.json', resources), ('CANDIDATES.json', c['candidates'])]:
        save(out / name, public(value))
    for name in ('FROZEN.json', 'MECHANICAL_TESTS.json', 'PROFILE_ADMISSION.json'):
        if (root / name).exists(): save(out / name, public(read(root / name)))
    save(out / 'COMPLETION_AUDIT.json', dict(formal_planned=30,
         formal_complete=sum(state['jobs'].get(j['id']) == 'COMPLETE' for j in c['jobs']),
         scored_jobs=len(done), scored_rows=len(rows), expected_scored_rows=58530,
         all_formal_workers_retired=not any(read(p).get('active') for p in (root / 'processes').glob('online_*.json')),
         online_target_label_reads=0, source_image_reads=0, all_attempts_charged=True))
    (out / 'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())
    lines = ['# R49: C-background LSO x shared sensitivity', '', f"Status: **{state['status']}**.",
             'Primary: C_JS_SHARED minus C. J is fixed R46 SHARED, without R48 descent constraint.', '',
             '| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |', '|---|---:|---:|']
    for candidate in c['candidates']:
        means = []
        for role in ('SEARCH', 'SEALED_REVIEW'):
            values = [x['macro_Dice_percent'] for x in main if x['condition'] == candidate['id'] and x['role'] == role]
            means.append(f'{statistics.mean(values):.6f}' if values else 'NA')
        lines.append(f"| {candidate['id']} | {means[0]} | {means[1]} |")
    lines += ['', f'Frozen standalone gate passed: {success}. Interpretation: {decision["interpretation"]}.',
              'The primary requires >=0.1pp, CI lower>0, both orders positive, >=5/6 positive trajectories, imageweighted>=0, worst seed-mean cell>=-2pp.',
              'Interaction I=[(C+LSO+J)-(C+LSO)]-[(C+J)-C]. All four arms are coupled within content during bootstrap. A negative interaction is not proof of identical internal function.',
              'CW_LSO stays unchanged; both J arms are compared to it. No secondary arm can replace the primary.',
              f"GPU-worker {resources['gpu_worker_seconds']/3600:.6f}/48h; CPU-worker {resources['cpu_worker_seconds']/3600:.6f}h. All attempts retained; T0 unchanged.",
              'Exactly five conditions, three seeds, two orders, 1951 arrivals: 30 fresh trajectories/58530 CPU scores. No source images, online labels, retuning, scientific/technical retry or next round.',
              'SEARCH/REVIEW were historically exposed; REVIEW descriptive only. Patient linkage UNKNOWN. ASSD/Brier/soft Dice uncomputed; no anatomical/clinical or independent generalization claim.',
              'Only own source/protocol and anonymous all-positive/all-negative aggregates are public. Images, labels, content identifiers/hashes, predictions, model states and raw/private logs remain private.']
    if state.get('reason'): lines += ['', 'Stopped reason: ' + public(state['reason'])]
    (out / 'REPORT.md').write_text('\n'.join(lines) + '\n')


def main():
    retained.ID = ID
    retained.Host = Host
    retained.report = report
    retained.public = public
    retained.main()


if __name__ == '__main__':
    main()
