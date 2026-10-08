"""CPU-only scoring after every online job retires; development evidence only."""
import copy
import json
import os
from pathlib import Path
import statistics as st
import time
from collections import defaultdict, Counter
import numpy as np
from .method import ARMS, ACTIONS
from .online import WIDTH
from ..r10_12h_core.run import read, save
from ..r20_model_only_search.report import csvout, aggregate, bootstrap_pair
from ..r24_c_context.run import hard_metrics
from ..r7_target_screen.runner import TargetReader


def mask_at(handle, offset):
    handle.seek(offset*WIDTH); raw = handle.read(WIDTH)
    if len(raw) != WIDTH:
        raise ValueError('incomplete prediction stream')
    return np.unpackbits(np.frombuffer(raw, np.uint8)).reshape(2, 512, 512).astype(bool)


def score_streams(c, guard):
    root = Path(c['output_root']); split = read(root/'scorer/SPLIT.private.json')
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask')
    all_rows = []; pre_rows = []; diagnostics = []
    for job in c['b_jobs']:
        rows = read(root/'scorer'/f'FULL_o{job["order"]}.json'); dest = root/'target'/job['id']
        receipt = read(dest/'COMPLETE.json')
        if receipt['arrivals'] != 1951 or (dest/'predictions.bits').stat().st_size != 1951*WIDTH:
            raise ValueError('incomplete full stream')
        traces = [json.loads(s) for s in (dest/'traces.jsonl').read_text().splitlines()]
        if len(traces) != len(rows):
            raise ValueError('trace coverage')
        with (dest/'predictions.bits').open('rb') as bits, (dest/'pre.bits').open('rb') as before:
            pre_index = 0
            for i, (row, trace) in enumerate(zip(rows, traces)):
                guard(); role = split.get(row['image_sha256'], 'CONTEXT')
                diagnostics.append(dict(condition=job['arm'], seed=job['seed'], order=job['order'],
                                        role=role, domain=row['domain'], **trace['diagnostics']))
                pre_offset = pre_index if trace['pre_recorded'] else None
                pre_index += trace['pre_recorded']
                if role not in ('SEARCH', 'SEALED_REVIEW'):
                    continue
                label = reader.read(row); mask = mask_at(bits, i); values = hard_metrics(mask, label)
                result = dict(condition=job['arm'], seed=job['seed'], order=job['order'], content=row['image_sha256'],
                              domain=row['domain'], subset=row['subset'], role=role, metrics=values,
                              containment_violations=int((mask[1] & ~mask[0]).sum()), empty_foreground=[v['pred_empty'] for v in values])
                all_rows.append(result)
                if pre_offset is not None:
                    pre_values = hard_metrics(mask_at(before, pre_offset), label)
                    pre_rows.append(dict(condition=job['arm'], seed=job['seed'], order=job['order'], domain=row['domain'],
                                         role=role, delta_pp=100*st.mean(a['dice']-b['dice'] for a, b in zip(values, pre_values))))
            if (dest/'pre.bits').stat().st_size != pre_index*WIDTH:
                raise ValueError('diagnostic before-mask coverage')
    expected = 48*(1017+678)
    if len(all_rows) != expected:
        raise ValueError('full-matrix score coverage')
    with (root/'scores/B.private.jsonl').open('x') as f:
        for row in all_rows:
            f.write(json.dumps(row)+'\n')
    main, domains = aggregate(all_rows)
    main = [{k: v for k, v in r.items() if k not in ('containment_violations', 'soft_Dice_percent', 'Brier', 'soft_channel_observations')}
            for r in main if r['role'] != 'PRIMARY_ALL']
    domains = [r for r in domains if r['role'] != 'PRIMARY_ALL']
    csvout(root/'public/FULL_TRAJECTORY_RESULTS.csv', main)
    csvout(root/'public/DOMAIN_CHANNEL.csv', domains)
    groups = defaultdict(list)
    for r in pre_rows:
        groups[r['condition'], r['seed'], r['order'], r['domain'], r['role']].append(r['delta_pp'])
    csvout(root/'public/BEFORE_AFTER.csv', [dict(condition=k[0], seed=k[1], order=k[2], domain=k[3], role=k[4],
                                               n=len(v), post_minus_pre_pp=st.mean(v)) for k, v in groups.items()])
    groups = defaultdict(list)
    for r in diagnostics:
        for key, value in r.items():
            if key not in ('condition', 'seed', 'order', 'role', 'domain') and isinstance(value, (int, float)):
                groups[r['condition'], r['seed'], r['order'], r['role'], key].append(value)
    csvout(root/'public/STATE_DIAGNOSTICS.csv', [dict(condition=k[0], seed=k[1], order=k[2], role=k[3], metric=k[4],
                                                    count=len(v), mean=st.mean(v), minimum=min(v), maximum=max(v))
                                                 for k, v in groups.items()])
    pairs = {('S_BATCH', 'S_SOURCE'), ('C_EPISODIC', 'S_BATCH'), ('C_CONT', 'C_EPISODIC')}
    pairs.update((a, b) for a in ARMS for b in ('C_CONT', 'ANCHOR') if a != b)
    contrasts = []; negatives = []; intervals = []
    for role in ('SEARCH', 'SEALED_REVIEW'):
        for a, b in sorted(pairs):
            result, ci, neg = bootstrap_pair(all_rows, a, b, c['seeds'], role=role)
            if result['status'] != 'COMPLETE':
                raise ValueError('incomplete paired contrast')
            contrasts.append(result); intervals.extend(ci); negatives.extend(neg)
    # Algebraic 2x2 interaction uses the same coupled-content bootstrap, not a new prediction arm.
    index = {(r['condition'], r['seed'], r['order'], r['content']): r for r in all_rows}
    interaction = []
    for r in all_rows:
        if r['condition'] != 'C_CONT':
            continue
        z = copy.deepcopy(r); z['condition'] = 'INTERACTION_ALGEBRA'
        others = [index[(a, r['seed'], r['order'], r['content'])] for a in ('C_BOTH32', 'C_PARAM32', 'C_OPT32')]
        for k, m in enumerate(z['metrics']):
            m['dice'] = others[0]['metrics'][k]['dice']-others[1]['metrics'][k]['dice']-others[2]['metrics'][k]['dice']+2*r['metrics'][k]['dice']
        interaction.extend((r, z))
    for role in ('SEARCH', 'SEALED_REVIEW'):
        result, ci, neg = bootstrap_pair(interaction, 'INTERACTION_ALGEBRA', 'C_CONT', c['seeds'], role=role)
        result['definition'] = 'BOTH32 - PARAM32 - OPT32 + CONT; algebraic contrast, not model Dice'
        contrasts.append(result); intervals.extend(ci); negatives.extend(neg)
    csvout(root/'public/PAIRED_CONTRASTS.csv', contrasts)
    csvout(root/'public/PAIRED_DOMAIN_INTERVALS.csv', intervals)
    csvout(root/'public/ALL_NEGATIVE_CELLS.csv', negatives)
    means = {a: st.mean(r['macro_Dice_percent'] for r in main if r['condition'] == a and r['role'] == 'SEARCH') for a in ARMS}
    reference = max(('C_CONT', 'ANCHOR'), key=lambda a: means[a])
    gates = {}
    for a in ('C_EPISODIC', 'C_OPT32', 'C_PARAM32', 'C_BOTH32'):
        r = next(r for r in contrasts if r['candidate'] == a and r['baseline'] == reference and r['role'] == 'SEARCH')
        gates[a] = dict(reference=reference, delta_pp=r['delta_pp'], order_delta_pp=r['order_delta_pp'],
                        positive_trajectories=r['positive_trajectories'], worst_cell_pp=r['worst_seed_averaged_cell_pp'],
                        passed=r['delta_pp'] >= .3 and min(r['order_delta_pp']) > 0 and r['positive_trajectories'] >= 5
                        and r['worst_seed_averaged_cell_pp'] >= -2.)
    return dict(SEARCH_macro=means, fixed_reference=reference, gates=gates,
                any_promising=any(r['passed'] for r in gates.values()))


def score_horizons(c, guard):
    root = Path(c['output_root']); split = read(root/'scorer/SPLIT.private.json')
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask'); results = []; channels = []
    eligibility = read(root/'scorer/WINDOWS.json')
    for job in c['a_jobs']:
        rows = read(root/'scorer'/f'FULL_o{job["order"]}.json'); dest = root/'target'/job['id']
        windows = [json.loads(s) for s in (dest/'windows.jsonl').read_text().splitlines()]
        if len(windows) != 16:
            raise ValueError('A window coverage')
        for window in windows:
            index, visit, available = window['index'], window['visit'], window['available_future']
            handles = [(dest/f'w{index}_{a}.bits').open('rb') for a in ACTIONS]
            try:
                scored = {}
                for offset in range(available+1):
                    guard(); row = rows[visit+offset-1]
                    if split.get(row['image_sha256']) != 'SEARCH':
                        continue
                    label = reader.read(row)
                    scored[offset] = np.array([[m['dice']*100 for m in hard_metrics(mask_at(f, offset), label)] for f in handles])
                for H in (0, 8, 32, 64):
                    e = eligibility[str(job['order'])][index][str(H)]
                    offsets = [k for k in scored if k == 0] if H == 0 else [k for k in scored if 1 <= k <= H]
                    if len(offsets) != e['score_count']:
                        raise ValueError('frozen window denominator mismatch')
                    result = dict(seed=job['seed'], order=job['order'], window=index, origin_domain=rows[visit-1]['domain'],
                                  H=H, available_future=available, n=len(offsets), eligible=e['eligible'], future_domains=e['domains'])
                    if e['eligible']:
                        means = np.mean([scored[k] for k in offsets], axis=0)
                        joint = means.mean(1); best = int(joint.argmax()); deltas = joint-joint[0]
                        result.update(zero_pp=float(deltas[1]), half_pp=float(deltas[2]), oracle_pp=float(deltas[best]),
                                      oracle_action=ACTIONS[best], zero_sum_pp=float(deltas[1]*len(offsets)),
                                      half_sum_pp=float(deltas[2]*len(offsets)))
                        for i, a in enumerate(ACTIONS):
                            for k, channel in enumerate(('OD', 'OC')):
                                channels.append(dict(seed=job['seed'], order=job['order'], window=index, origin_domain=result['origin_domain'],
                                                     H=H, action=a, channel=channel, delta_pp=float(means[i, k]-means[0, k])))
                    results.append(result)
            finally:
                for f in handles:
                    f.close()
    if len(results) != 96*4:
        raise ValueError('A result coverage')
    csvout(root/'public/HORIZON_RESULTS.csv', results)
    csvout(root/'public/HORIZON_CHANNELS.csv', channels)
    csvout(root/'public/HORIZON_NEGATIVE_CELLS.csv', [r for r in channels if r['delta_pp'] < 0])
    summaries = []
    for H in (0, 8, 32, 64):
        groups = defaultdict(list)
        for r in results:
            if r['H'] == H and r['eligible']:
                groups[r['seed'], r['order'], r['origin_domain']].append(r)
        if len(groups) != 24:
            raise ValueError('every seed/order/origin-domain requires eligible windows')
        cells = [dict(seed=k[0], order=k[1], domain=k[2], **{name: st.mean(x[name] for x in v) for name in ('zero_pp', 'half_pp', 'oracle_pp')})
                 for k, v in groups.items()]
        summary = dict(H=H, eligible_states=sum(len(v) for v in groups.values()),
                       **{name: st.mean(r[name] for r in cells) for name in ('zero_pp', 'half_pp', 'oracle_pp')})
        summary['oracle_minus_best_fixed_pp'] = summary['oracle_pp']-max(0., summary['zero_pp'], summary['half_pp'])
        summary['order_oracle_pp'] = [st.mean(r['oracle_pp'] for r in cells if r['order'] == o) for o in (0, 1)]
        summary['positive_oracle_groups'] = sum(st.mean(r['oracle_pp'] for r in cells if r['seed'] == s and r['order'] == o) > 0
                                               for s in c['seeds'] for o in (0, 1))
        summaries.append(summary)
    csvout(root/'public/HORIZON_SUMMARY.csv', summaries)
    primary = summaries[-1]
    primary['allocation_signal'] = primary['oracle_pp'] >= .3 and min(primary['order_oracle_pp']) > 0 and primary['positive_oracle_groups'] >= 5
    return dict(primary=primary, summaries=summaries, independent_windows=False, primary_H=64)


def score(c, guard):
    root = Path(c['output_root'])
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU scorer required')
    records = [read(root/'processes'/f'online_{j["id"]}.json') for j in c['b_jobs']+c['a_jobs']]
    if len(records) != 54 or any(r['active'] or r['exit_code'] != 0 for r in records):
        raise ValueError('all-online retirement barrier')
    save(root/'LABEL_RELEASE.json', dict(at=time.time(), all_workers_retired=True, SEARCH=True, REVIEW='development only'))
    b = score_streams(c, guard); a = score_horizons(c, guard)
    if b['any_promising']:
        decision = 'RECHECK_SIMPLE_STATE_STRATEGY_NO_AUTOMATIC_RL'
    elif a['primary']['allocation_signal']:
        decision = 'DELAY_SIGNAL_WITHOUT_FULL_STREAM_GAIN_NO_AUTOMATIC_RL'
    else:
        decision = 'STOP_REGISTERED_ACTION_HOST_PERIOD_SWEEPS'
    result = dict(A=a, B=b, next_decision=decision, independent_confirmation=False,
                  R31_launched=False, patient_linkage='UNKNOWN', ROI_provenance='UNKNOWN')
    save(root/'public/NEXT_DECISION.json', result)
    return result


def finish(c, state):
    root = Path(c['output_root']); public = root/'public'
    attempts = [read(p) for p in sorted((root/'attempts').glob('*.json'))]
    costs = [dict(phase=r['phase'], attempt=r['attempt'], status=r['status'], **r['cost']) for r in attempts]
    csvout(public/'COST.csv', costs)
    completed = state['status'] == 'COMPLETE'
    audit = dict(status=state['status'], completed_B=sum(state['jobs'][j['id']] == 'COMPLETE' for j in c['b_jobs']),
                 completed_A=sum(state['jobs'][j['id']] == 'COMPLETE' for j in c['a_jobs']),
                 wall_seconds=time.time()-c['origin']['T0_epoch'], gpu_worker_seconds=sum(r['cost'].get('gpu_seconds', 0) for r in attempts),
                 cpu_worker_seconds=sum(r['cost'].get('cpu_seconds', 0) for r in attempts),
                 failure_count=sum(r['status'] != 'COMPLETE' for r in attempts), label_reads_online=0,
                 barrier_verified=completed and read(root/'LABEL_RELEASE.json')['all_workers_retired'],
                 scientific_status='EVALUATED' if completed else 'INCOMPLETE_NOT_SCIENTIFIC_NEGATIVE')
    save(public/'COMPLETION_AUDIT.json', audit)
    text = '# R30: delayed effects and cross-arrival state history\n\nStatus: **'+state['status']+'**.\n\n'
    if completed:
        d = state['decision']; a = d['A']['primary']; b = d['B']
        text += f"H=64 local future oracle gain: {a['oracle_pp']:+.6f} pp; ZERO {a['zero_pp']:+.6f}, HALF {a['half_pp']:+.6f}. Delayed allocation signal: {a['allocation_signal']}.\n\n"
        text += f"Any simple state intervention passed the development gate: {b['any_promising']}. Stronger fixed SEARCH reference: {b['fixed_reference']}.\n\n"
        text += '| Arm | SEARCH macro Dice (%) |\n|---|---:|\n'+''.join(f'| {k} | {v:.6f} |\n' for k, v in b['SEARCH_macro'].items())
        text += '\nAll paired deltas against both C_CONT and ANCHOR, six trajectory effects, descriptive coupled-content intervals, and negative domain/channel cells are in PAIRED_CONTRASTS.csv and ALL_NEGATIVE_CELLS.csv.\n\n'
        for arm, g in b['gates'].items():
            text += f"- {arm}: {g['delta_pp']:+.6f} pp against {g['reference']}; worst seed-averaged domain/channel/order cell {g['worst_cell_pp']:+.6f} pp; gate {g['passed']}.\n"
        text += '\nNext decision: **'+d['next_decision']+'**. No R31 or RL training was launched.\n\n'
    else:
        text += 'The registered matrix was not completed. Do not interpret missing arms or unavailable metrics as negative evidence. See runtime status and completion audit.\n\n'
    text += f"Recorded GPU-worker time: {audit['gpu_worker_seconds']/3600:.4f} h; wall time: {audit['wall_seconds']/3600:.4f} h. Includes profile and failed attempts. R28 snapshot-generation cost is excluded from this incremental ledger.\n\n"
    text += 'SEARCH and old REVIEW are development-exposed; REVIEW is descriptive only. Patients and ROI provenance are UNKNOWN. Windows overlap and reuse content across seeds: they are not independent patients. Oracle includes FULL and is not executable. OPT32 resets moments and bias-correction time together; parameter-only restoration preserves potentially mismatched optimizer history. Reset period 32 is fixed, not selected from results. Interactions and bootstrap intervals are descriptive, without patient independence or selection-adjusted inference. No soft metric is inferred from hard masks.\n\n'
    text += 'Published artifacts contain code, protocol, aggregated metrics, and audit/cost summaries. Images, labels, masks, states, weights, private paths, and identities are excluded. GitHub delivery remains pending until remote and anonymous-access verification.\n'
    (public/'REPORT.md').write_text(text)
