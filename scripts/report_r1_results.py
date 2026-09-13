"""CPU-only report supplement from completed R1 scalars; never reruns a model.

Set RESULT_DIR, CONTROL_DIR and REPORT_DIR outside the execution checkout.
Import the frozen execution checkout via PYTHONPATH. Private identities stay there.
"""
import csv
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import torch
from dpa_ctta.r1 import analyze
from dpa_ctta.r1.plan import binding, matrix, science, stream
from dpa_ctta.host_diagnostic_analysis import channels, distribution
from dpa_ctta.m1_analysis import paired


def export():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    out, control, destination = (Path(os.environ[k]) for k in
                                 ('RESULT_DIR', 'CONTROL_DIR', 'REPORT_DIR'))
    destination.mkdir(mode=0o700, exist_ok=False)
    reg = json.loads((control / 'assets.private.json').read_text())['registration']
    aggregate = json.loads((out / 'public_aggregate.json').read_text())
    pointer = json.loads((out / 'current_result.json').read_text())
    assert pointer['valid'] and pointer['status'] == aggregate['status'] == 'R1_EXPERIMENT_COMPLETE'
    receipt, slots, smokes = analyze.execution_evidence(out, reg)
    assert pointer['binding'] == aggregate['binding'] == receipt['binding']
    process = json.loads((out / 'matrix.processes.json').read_text())
    exited = json.loads((control / 'launcher.exit.json').read_text())
    assert exited['exit_code'] == 0
    all_rows, jobs, mechanisms = {}, [], {}
    totals = dict(records=0, forwards=0, backwards=0, adam=0)
    for job in matrix()['jobs']:
        folder = out / job['job_id']
        rows = [json.loads(line) for line in (folder / 'records.jsonl').read_text().splitlines()]
        identity = binding(receipt, slots[job['job_id']], job)
        analyze.validate(rows, stream(reg, job['order']), job['arm'], job['order'], identity)
        done = json.loads((folder / 'completion.json').read_text())
        assert done['binding'] == identity and done['status'] == 'TRAJECTORY_COMPLETE'
        assert not list(folder.glob('*failure.json'))
        counts = {k: sum(r['counts'][k] for r in rows) for k in done['physical']}
        assert counts == done['physical'] and done['records'] == len(rows) == job['records']
        assert counts['forwards'] == job['forwards'] and counts['base_adam'] == job['adam']
        totals['records'] += len(rows)
        for target, source in [('forwards', 'forwards'), ('backwards', 'backwards'), ('adam', 'base_adam')]:
            totals[target] += counts[source]
        all_rows[job['order'], job['arm']] = rows
        costs = {k: distribution([r[k] for r in rows]) for k in ('host_seconds', 'pipeline_seconds')}
        io = {kind: dict(bytes=sum(r['asset_io'][kind]['bytes'] for r in rows),
                         seconds=sum(r['asset_io'][kind]['read_verify_decode_seconds'] for r in rows))
              for kind in ('image', 'mask')}
        mechanisms[job['job_id']] = dict(aggregate['mechanism'][job['job_id']])
        if job['arm'].startswith('C_PCA_'):
            ps = [r['pca'] for r in rows]
            banks = []
            for i, last in enumerate(ps[-1]['banks']):
                banks.append(dict(bank=i, final=last,
                    ready_basis_visits=sum(p['basis_versions_used'][i] is not None for p in ps),
                    contributing_visits=sum(p['input']['assigned_bank_counts'][i] > 0 for p in ps)))
            mechanisms[job['job_id']].update(banks=banks,
                subloss_distribution=distribution([p['subloss'] for p in ps]),
                base_loss_distribution=distribution([p['base_loss'] for p in ps]),
                no_selected_tokens_per_region=[sum(p['input']['selected_region_counts'][i] == 0 for p in ps) for i in range(4)],
                missing_foreground_visits=[sum(p['input']['missing_foreground'][i] for p in ps) for i in range(2)])
        if job['arm'] == 'C_SENS':
            cs = [r['controller'] for r in rows]
            eligible = [c['trend'] for c in cs if c['trend'] and c['trend']['age'] >= 50]
            mechanisms[job['job_id']]['controller_summary'] = dict(
                sensitivity_missing=sum(c['sensitivity'] is None for c in cs),
                eligible_visits=len(eligible), max_eligible_ema_over_best=max(t['ema']/t['best'] for t in eligible),
                sensitivity_distribution=distribution([c['sensitivity'] for c in cs if c['sensitivity'] is not None]))
        entry = dict(job_id=job['job_id'], arm=job['arm'], order=job['order'], worker=slots[job['job_id']],
                     physical_gpu=receipt['devices'][slots[job['job_id']]]['index'], status=done['status'],
                     records=len(rows), physical=counts, trajectory_seconds=done['seconds'],
                     peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),
                     latency=costs, asset_io=io, checkpoint_io=done['checkpoint_io'])
        jobs.append(entry)
    assert totals == aggregate['physical'] == {k: science()['formal_budget'][k] for k in totals}
    order_rows, domain_rows, pair_rows = [], [], []
    pairs = (*analyze.PAIRS, ('C_PCA_SHUFFLED', 'C'))
    for order in range(4):
        reference = all_rows[order, 'C']
        for arm in science()['arms']:
            for a, b in zip(all_rows[order, arm], reference):
                assert all(x[k] == y[k] for x, y in zip(a['metrics'], b['metrics'])
                           for k in ('gt_pixels', 'total_pixels', 'gt_empty', 'gt_full'))
        for subset in analyze.SUBSETS:
            selected = {arm: [r for r in all_rows[order, arm] if subset == 'all_dev' or r['subset'] == subset]
                        for arm in science()['arms']}
            means = {arm: {} for arm in selected}
            for domain in science()['orders'][0]:
                rs = {arm: [r for r in records if r['domain'] == domain] for arm, records in selected.items()}
                for arm, records in rs.items():
                    value = dict(order=order, subset=subset, domain=domain, arm=arm, n=len(records))
                    for channel in ('OD', 'OC', 'macro'):
                        score = np.mean([m['dice']*100 for m in channels(records, channel)])
                        expected = aggregate['target'][str(order)][subset]['domains'][domain]['arms'][arm][channel]['dice_percent']['mean']
                        assert abs(score-expected) < 1e-10
                        value[channel] = float(score)
                    means[arm][domain] = value
                    domain_rows.append(value)
                for a, b in pairs:
                    for channel in ('OD', 'OC', 'macro'):
                        stats = paired(channels(rs[a], channel), channels(rs[b], channel), channel == 'macro')
                        pair_rows.append(dict(order=order, subset=subset, domain=domain, left=a, right=b,
                            channel=channel, **{'dice_'+k: v for k, v in stats['dice_delta_pp'].items()},
                            **{k: v for k, v in stats.items() if k != 'dice_delta_pp'}))
            for arm in means:
                score = {c: float(np.mean([m[c] for m in means[arm].values()])) for c in ('OD', 'OC', 'macro')}
                assert abs(score['macro']-aggregate['target'][str(order)][subset]['task_domain_macro_dice_percent'][arm]) < 1e-10
                order_rows.append(dict(order=order, subset=subset, arm=arm, n=len(selected[arm]), **score))
    primary = []
    for arm in science()['arms']:
        values = [r for r in order_rows if r['arm'] == arm and r['subset'] == 'remaining_dev']
        scores = [r['macro'] for r in values]
        baseline = [r['macro'] for r in order_rows if r['arm'] == 'C' and r['subset'] == 'remaining_dev']
        primary.append(dict(arm=arm, **{c: float(np.mean([r[c] for r in values])) for c in ('OD', 'OC', 'macro')},
            delta_C_pp=float(np.mean(np.array(scores)-baseline)), worst_order=min(scores), best_order=max(scores),
            order_range=max(scores)-min(scores), positive_orders=sum(a>b for a,b in zip(scores,baseline))))
    audit = dict(status='R1_EXPERIMENT_COMPLETE', binding=receipt['binding'], launcher_exit_code=exited['exit_code'],
        started_unix=exited['started_unix'], ended_unix=exited['ended_unix'],
        total_wall_seconds=exited['ended_unix']-exited['started_unix'], supervisor_wall_seconds=process['wall_seconds'],
        active_worker_seconds=process['active_seconds'], physical=totals, smoke_physical=aggregate['smoke_physical'],
        process_count=len(process['processes']), process_exit_codes=process['exit_codes'], jobs=jobs,
        smoke=[dict(worker=i, physical_gpu=receipt['devices'][i]['index'], physical=s['physical'],
                    seconds=s['seconds'], checkpoint_io=s['checkpoint_io'], backend=s['backend']) for i,s in enumerate(smokes)],
        output_accounted_bytes=sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),
        validation='Frozen execution_evidence and validate, completion/call counts, GT metadata and aggregate means',
        cuda_initialized=torch.cuda.is_initialized(), real_images_masks_or_weights_read=False)
    assert not audit['cuda_initialized']
    for name, value in [('EXECUTION_AUDIT.json', audit), ('MECHANISM_AND_COST.json', mechanisms)]:
        (destination/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for name, data in [('PRIMARY.csv', primary), ('ORDER_SUBSET.csv', order_rows), ('DOMAIN_SCORES.csv', domain_rows),
                       ('PAIRED_DISTRIBUTIONS.csv', pair_rows)]:
        fields = list(dict.fromkeys(k for row in data for k in row))
        with (destination/name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(data)
    print(json.dumps(dict(status=audit['status'], records=totals['records'], validated_jobs=len(jobs),
                          primary=primary, output_files=sorted(p.name for p in destination.iterdir()))), flush=True)


if __name__ == '__main__':
    export()
