"""Export an already validated R3 run. Stdlib only; never opens image/model assets.

Set RESULT_INPUT, RESULT_CONTROL and RESULT_EXPORT in the environment, then run
this file. Inline assertions are the completion/coverage/counter self-check.
The original analyzer remains authoritative for matching and metric validation.
"""
import collections
import csv
import datetime
import gzip
import json
import os
from pathlib import Path


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()
                if k not in {'run_id', 'device_uuid', 'pid', 'pgid', 'result_directory'}}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def dump(name, value):
    text = json.dumps(clean(value), indent=2, allow_nan=False) + '\n'
    for forbidden in ('/data_nas/', '/home/', '/Users/', 'GPU-', 'sample_id',
                      'group_id', '.png', '.jpg', '.jpeg', 'wangbomin'):
        assert forbidden not in text, (name, forbidden)
    (destination / name).write_text(text)


def table(name, rows):
    with (destination / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    source = Path(os.environ['RESULT_INPUT'])
    control = Path(os.environ['RESULT_CONTROL'])
    destination = Path(os.environ['RESULT_EXPORT'])
    destination.mkdir(parents=True, exist_ok=True)
    read = lambda name: json.loads((source / name).read_text())
    current = read('current_result.json')
    aggregate = read('public_aggregate.json')
    matrix = read('matrix.processes.json')
    launcher = json.loads((control / 'launcher.exit.json').read_text())
    assert current['valid'] and current['status'] == 'R3_EXPERIMENT_COMPLETE'
    assert aggregate['binding'] == current['binding'] == matrix['binding']
    assert matrix['status'] == 'COMPUTE_COMPLETE' and not matrix['unstarted_jobs']
    assert len(matrix['processes']) == 87 and matrix['exit_codes'] == [0] * 87
    assert launcher['exit_code'] == 0
    assert not list(source.glob('**/*failure*.json'))
    jobs, costs, contexts = [], [], {}
    physical = collections.Counter()
    for process in matrix['processes']:
        if process['phase'] != 'formal':
            continue
        job = process['key']
        completion = read(job + '/completion.json')
        assert completion['status'] == 'TRAJECTORY_COMPLETE'
        assert completion['binding'] == process['binding']
        arm, order = completion['binding']['arm'], completion['binding']['order']
        cost = dict(job=job, arm=arm, order=order, records=0,
                    worker_seconds=completion['seconds'], host_seconds=0.,
                    pipeline_seconds=0., peak_allocated_bytes=0,
                    peak_bank_state_bytes=0, peak_context_tensor_bytes=0,
                    peak_slots=0, context_created=0, context_switches=0,
                    context_overflows=0, context_load_seconds=0.)
        counts = collections.Counter()
        domain_slots = collections.defaultdict(collections.Counter)
        blocks = []
        for line in (source / job / 'records.jsonl').open():
            row = json.loads(line)
            assert row['arm'] == arm and row['order'] == order
            cost['records'] += 1
            counts.update(row['r3']['counts'])
            cost['host_seconds'] += row['host_seconds']
            cost['pipeline_seconds'] += row['pipeline_seconds']
            cost['peak_allocated_bytes'] = max(cost['peak_allocated_bytes'], row['peak_allocated_bytes'])
            state_bytes = sum(b.get('state_bytes', 0) for b in row['r3'].get('banks', []))
            cost['peak_bank_state_bytes'] = max(cost['peak_bank_state_bytes'], state_bytes)
            context = row['r3'].get('context')
            if context:
                cost['peak_context_tensor_bytes'] = max(cost['peak_context_tensor_bytes'], context['context_tensor_bytes'])
                cost['peak_slots'] = max(cost['peak_slots'], context['slots'])
                cost['context_created'] += int(context['created'])
                cost['context_switches'] += int(context['switched'])
                cost['context_overflows'] += int(context['overflow'])
                cost['context_load_seconds'] += context['load_seconds']
                domain_slots[row['domain']][context['slot']] += 1
                if not blocks or blocks[-1]['domain'] != row['domain']:
                    blocks.append(dict(domain=row['domain'], visits=0, slots=collections.Counter(),
                                       created=0, switches=0, overflow=0))
                block = blocks[-1]
                block['visits'] += 1
                block['slots'][context['slot']] += 1
                for key, source_key in [('created', 'created'), ('switches', 'switched'), ('overflow', 'overflow')]:
                    block[key] += int(context[source_key])
        assert cost['records'] == completion['records'] == 1951
        assert dict(counts) == completion['physical']
        physical.update(counts)
        cost.update(counts)
        costs.append(cost)
        jobs.append(completion)
        if domain_slots:
            contexts[job] = dict(arm=arm, order=order, domain_slot_visits=domain_slots, blocks=blocks)
        print('PASS', job, arm, 'records=1951 counters=matched', flush=True)
    assert len(jobs) == 85
    assert len({(c['arm'], c['order']) for c in costs}) == 85
    assert dict(physical) == {k: v for k, v in aggregate['physical'].items() if k != 'records'}
    assert sum(c['records'] for c in costs) == aggregate['physical']['records'] == 165835
    smokes = [read('device%d/smoke.completion.json' % i) for i in range(2)]
    for smoke in smokes:
        assert smoke['status'] == 'MECHANICAL_SMOKE_COMPLETE'
    dump('aggregate.json', aggregate)
    aggregate_text = (destination / 'aggregate.json').read_bytes()
    (destination / 'aggregate.json.gz').write_bytes(gzip.compress(aggregate_text, mtime=0))
    (destination / 'aggregate.json').unlink()
    beijing = datetime.timezone(datetime.timedelta(hours=8))
    launcher.update(started_beijing=datetime.datetime.fromtimestamp(launcher['started_unix'], beijing).isoformat(),
                    ended_beijing=datetime.datetime.fromtimestamp(launcher['ended_unix'], beijing).isoformat(),
                    elapsed_seconds=launcher['ended_unix'] - launcher['started_unix'])
    dump('execution.json', dict(current=current, launcher=launcher, supervisor=matrix,
                                formal_completions=jobs, smoke_completions=smokes,
                                gpu_indices=[6, 7], maximum_concurrency=2))
    dump('contexts.json', contexts)
    table('costs_by_trajectory.csv', costs)
    primary = aggregate['primary_four_order_equal']['remaining_dev']['arms']
    secondary = aggregate['secondary_recurrence']['remaining_dev']['domain_equal_dice_percent']
    rows = []
    for arm, scores in primary.items():
        rows.append(dict(arm=arm, OD= scores['OD'], OC=scores['OC'], macro=scores['macro'],
                         delta_C_pp=scores['macro']-primary['C']['macro'],
                         delta_RP_pp=scores['macro']-primary['RP']['macro'],
                         secondary_macro=secondary[arm]['macro'],
                         secondary_delta_C_pp=secondary[arm]['macro']-secondary['C']['macro']))
    table('primary_and_secondary.csv', rows)
    print('PASS 85 trajectories; 165835 records; 87 zero exits; all physical counters match.')
    print('CPU-only stdlib export; no torch import, target assets, checkpoints or per-content export.')
