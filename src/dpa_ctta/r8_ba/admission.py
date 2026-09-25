"""Convert measured units and fixed output shapes into full-matrix admission."""
import math

from .execution_plan import tasks
from .scope import SCREEN
from .trainer import MAX_STEPS, SAVE_STEPS
from .resources import CAPS, MEASURES, SAFETY, category, project, units, json_size_bound


def physical(counts):
    return dict(model_forwards=counts.get('backbone_forwards', 0),
                backward_calls=sum(v for k,v in counts.items() if k.endswith('backward_calls')),
                optimizer_steps=sum(v for k,v in counts.items() if k.endswith(('Adam', 'AdamW'))),
                vjp_calls=sum(v for k,v in counts.items() if k.endswith('VJP')))


def build(graph, source, target, io, code_sha, free_gpu_bytes, prior_cost, prior_evidence):
    for row in (source, target, io):
        if (row['code_sha'] != code_sha or not row['status'].startswith('MEASURED_SURROGATE') or
                row['physical_gpu'] not in (5, 6, 7)):
            raise ValueError('R8 current-code measured profile required')
    if len(source.get('recovery', [])) != 8 or any(r['status'] != 'EXACT_CONTINUATION_MATCH' for r in source['recovery']):
        raise ValueError('R8 real source snapshot checks incomplete')
    if target.get('recovery') != dict(r7_c_full='EXACT_CONTINUATION_MATCH', r7_c_static='EXACT_CONTINUATION_MATCH'):
        raise ValueError('R8 frozen control recovery checks incomplete')
    measured = dict(source['units'], **target['units'])
    io_excess = max(0, io['unit']['seconds']/io['unit']['sample_units'] -
                    target['units']['new_b_max']['seconds']/target['units']['new_b_max']['sample_units'])
    source_work = sum(r['seconds'] for obs in source['observations'].values() for r in obs)
    target_work = sum(r['seconds'] for r in target['units'].values())
    setup = max(source['elapsed_seconds'] - source_work, target['elapsed_seconds'] - target_work,
                io['elapsed_seconds'] - io['unit']['seconds'])
    measured['worker_setup'] = dict(sample_units=1, seconds=setup, counts={},
                                   peak_gpu_bytes=max(r['peak_gpu_bytes'] for r in measured.values()))
    rows = {}
    for name, observation in measured.items():
        n = observation['sample_units']
        row = dict.fromkeys(MEASURES, 0)
        row.update({k: v/n for k,v in physical(observation['counts']).items()})
        row.update(gpu_seconds=observation['seconds']/n, peak_gpu_bytes=observation['peak_gpu_bytes'],
                   measured=True, sample_units=n)
        if name in {category(job) for job in graph['jobs'] if job['arrivals']} or name == 'gradient_lr_visit':
            row['gpu_seconds'] += io_excess
        # Reserve the GPU lane during independent scoring too; this is deliberately
        # an upper bound on accelerator allocation time, not kernel-only timing.
        rows[name] = row
    if SCREEN:
        rows = {k:v for k,v in rows.items() if k in units(graph)}
    if set(rows) != set(units(graph)):
        raise ValueError('R8 measured resource unit coverage')
    plan, _ = tasks(graph)
    storage = {}
    source_temps = []
    # Fixed-shape tensor caches: scaler float32 [512,111,134], full retained
    # snapshots plus sealed artifact. Basis includes a float64 1024-square audit.
    prepared = dict(ORACLE=4 * (768 * (1024*8 + 4096) + 65536),
                    SCALER=4 * (512*111*134*4 + 512*4096),
                    BASES=2 * (1024*1024*8 + 1024*144*8 + 65536),
                    CAPACITY=4*256*8192, GRADIENT_LR=4*1152*8192)
    prepared['ORACLE_SHARD'] = prepared['ORACLE']
    for task in plan:
        kind, job = task['kind'], task['job']
        # Config, receipt, metadata and error tails, bounded independently of visits.
        fixed = 128*1024
        if kind in prepared:
            fixed += prepared[kind]
        elif kind == 'SOURCE_JOB':
            route = 'mlp' if job['stage'] == 'SOURCE_MLP' else job['arm'].lower()
            source_temps.append(source['storage']['fit_' + route])
            fixed += 8 * source['storage']['fit_' + route]  # five selected + two slots + temp
            fit_name = 'source_fit_' + route + '_step'
            fixed += MAX_STEPS * source['record_bytes'][fit_name]
            val_name = 'source_val_' + route + '_visit'
            # Five (ten for A/B) sets, each two JSON snapshots + result + receipt.
            sets = len(SAVE_STEPS)*(1 if route == 'mlp' else 2)
            fixed += sets * (64*source['record_bytes'][val_name] + 64*32*192)
            if route != 'mlp':
                source_temps.extend([source['storage']['cal_' + route]]*len(SAVE_STEPS))
                fixed += len(SAVE_STEPS) * (3*source['storage']['cal_' + route] + source['storage']['method_' + route])
                fixed += len(SAVE_STEPS)*1024 * source['record_bytes']['source_cal_' + route + '_step']
        elif kind == 'TARGET_JOB':
            name = category(job)
            # The third snapshot slot is temporary; summing it per job is a safe
            # bound even though at most three jobs can write a temporary snapshot.
            fixed += 3*target['storage'][name] + target['context_bytes'][name]
            fixed += job['arrivals'] * (65536 + 2*target['record_bytes'][name])
        elif kind == 'SCORE_JOB':
            fixed += job['arrivals'] * (target['record_bytes']['score_visit'] + 160)
        else:
            raise ValueError('R8 unbudgeted work kind')
        storage[task['id']] = fixed
    # Only two snapshots persist. Use the largest three temporary snapshots in
    # the global bound; per-worker admission retains its own third-slot budget.
    retained = sum(storage.values())
    temps = [target['storage'][category(t['job'])] for t in plan if t['kind'] == 'TARGET_JOB']
    temps += source_temps
    retained -= sum(temps) - sum(sorted(temps, reverse=True)[:3])
    prediction_bytes = graph['target_arrivals'] * 65536
    profile = dict(schema='R8_RESOURCE_PROFILE_V1', code_sha=code_sha,
                   measured_on_physical_GPU=True, physical_GPU_ids=sorted({r['physical_gpu'] for r in (source,target,io)}),
                   units=rows, fixed_disk_bytes=retained - prediction_bytes)
    projection = project(graph, profile, free_gpu_bytes, code_sha)
    for key in CAPS:
        projection['with_margin'][key] += prior_cost[key]
        if projection['with_margin'][key] >= CAPS[key]:
            raise RuntimeError('R8 prior physical cost plus full matrix exceeds ' + key)
    budgets = {}
    for task in plan:
        budget = {key: math.ceil(SAFETY * sum(count*rows[name][key] for name,count in task['weights'].items())) for key in CAPS}
        budget['disk_bytes'] = math.ceil(SAFETY * storage[task['id']])
        # Zero is a prohibition; positive operation bounds include one extra call
        # because the physical guard stops on reaching its reservation.
        for key in CAPS:
            if budget[key] > 0:
                budget[key] += 1
        budgets[task['id']] = budget
    return dict(profile=profile, projection=projection, budgets=budgets,
                prior_cost=prior_cost, prior_evidence=prior_evidence,
                storage_by_task=storage, io_excess_seconds_per_visit=io_excess,
                setup_seconds=setup)
