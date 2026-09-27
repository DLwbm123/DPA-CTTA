"""Expanded measurement contract. Unit costs are never fabricated from plan arithmetic."""
from collections import Counter
import time
from .protocol import SPEC,CAPS,parse_recipe,digest
from .gradient import STEPS
from .target import PREDICTION_BYTES
from .admission import project


def node_units(node):
    result=Counter();kind=node['kind']
    if kind=='source':
        j=node['job'];recipe=j['recipe'];route=recipe.split('_')[0]
        fit='SELECTED_MAX' if 'SELECTED' in recipe else parse_recipe(recipe)[1]
        result[f'fit/{route}/{fit}']=16000
        result[f'validation_episode/{route}']=4*4*64
        result[f'source_setup/{route}']=1
        result[f'fit_checkpoint/{route}']=321
        result['journal_append']=16000+4*4*64
        result[f'validation_checkpoint/{route}']=4*4*3
        result[f'source_finalize/{route}']=1
        if route!='MLP':
            result[f'calibration/{route}']=4*256
            result[f'calibration_checkpoint/{route}']=4*7
            result['journal_append']+=4*256
    elif kind=='select_lr':
        for arm in STEPS:
            result[f'lr_episode/{arm}']=2*3*64
            result[f'lr_setup/{arm}']=2*3
        result['lr_source_setup']=1
    elif kind in ('online','score'):
        slot=node['job'];n=19510 if slot['order']=='LONG10' else 1951
        if kind=='online':
            result['online/'+slot['arm']]=n
            result['online_setup/'+slot['arm']]=1
            result['target_journal_append']=n
            result['target_checkpoint/'+slot['arm']]=1+n//50
        else:
            result['cpu_score_visit']=n;result['cpu_score_seal_visit']=n
    else:result['control/'+kind]=1
    return dict(result)


def units():
    from .protocol import graph
    result=Counter()
    for node in graph()['nodes']:result.update(node_units(node))
    return dict(result)


def projection(measurements,lr_seed_count,retained_disk,prior=None,node_budgets=None):
    from .protocol import graph
    if lr_seed_count!=2:raise ValueError('fixed first_two_mean source seeds')
    nodes=graph()['nodes']
    if node_budgets is None or set(node_budgets)!={n['id'] for n in nodes}:raise ValueError('full node budgets required')
    recovery=[]
    for node in nodes:
        budget=node_budgets[node['id']]
        lower=project(node_units(node),measurements,0,0)['upper_bound']
        if set(budget)!=set(CAPS) or any(budget[k]<lower[k] for k in CAPS):raise ValueError('node budget below measured full attempt')
        row=budget.copy()
        # Probability prefixes occupy the same file across attempts; only retained
        # non-probability output can accumulate in the finite recovery reserve.
        if node['kind']=='online':
            n=19510 if node['job']['order']=='LONG10' else 1951
            if row['disk_bytes']<n*PREDICTION_BYTES:raise ValueError('online probability budget')
            row['disk_bytes']-=n*PREDICTION_BYTES
        recovery.append(row)
    if retained_disk<sum(row['disk_bytes'] for row in recovery):raise ValueError('retained disk below full matrix output bound')
    proof=project(units(),measurements,retained_disk,19510*PREDICTION_BYTES,prior,recovery)
    base=project(units(),measurements,0,0)['upper_bound']
    for k in CAPS:
        if k!='disk_bytes':proof['upper_bound'][k]+=sum(b[k] for b in node_budgets.values())-base[k]
    proof['status']='PASS' if all(proof['upper_bound'][k]<=CAPS[k] for k in CAPS) else 'OVER_CAP'
    return proof


def measurement_binding(config):
    from .identity import verify_runtime
    from .protocol import RECOVERY_POLICY
    a=config.get('profile_authorization') or {}
    binding=dict(code_sha=config.get('code_sha'),bindings_sha256=digest(config.get('bindings')),
                 gpu_assignments=config.get('gpu_assignments'),output_root=config.get('output_root'))
    if (a.get('scope')!='REAL_DATA_PROFILE_ONLY' or not a.get('user_instruction') or
        any(a.get(k)!=v for k,v in binding.items()) or not config.get('bindings') or
        not binding['gpu_assignments'] or not binding['output_root']):raise PermissionError('explicit asset/GPU/root-bound profile authorization required')
    from pathlib import Path
    if not Path(binding['output_root']).is_absolute():raise ValueError('profile output root')
    if any(not x.get('uuid','').startswith('GPU-') or type(x.get('physical_id')) is not int for x in binding['gpu_assignments']):raise ValueError('profile GPU binding')
    if config.get('lr_source_policy')!='first_two_mean' or config.get('recovery_policy')!=RECOVERY_POLICY:raise ValueError('profile protocol choices')
    verify_runtime(config)
    return binding


def measure(call,meter,iterations,config):
    """Use the real production kernel with source-only inputs; authorization belongs upstream."""
    if type(iterations) is not int or iterations<1:raise ValueError('positive measured iterations')
    binding=measurement_binding(config)
    import os
    from .assets import gpu_policy
    assigned=[g for g in binding['gpu_assignments'] if str(g['physical_id'])==os.environ.get('CUDA_VISIBLE_DEVICES')]
    if len(assigned)!=1:raise ValueError('profile GPU not among authorized bindings')
    gpu_policy(assigned[0])
    import torch
    before=meter.cost.copy();torch.cuda.synchronize();start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iterations):call()
    torch.cuda.synchronize();meter.check();elapsed=time.monotonic()-start
    row={k:(meter.cost[k]-before[k])/iterations for k in CAPS}
    row.update(measured=True,iterations=iterations,elapsed_seconds=elapsed,
               max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
    row['gpu_seconds']=elapsed/iterations
    row['binding']=binding;row['evidence']=digest(row);return row


def measure_cpu(call,iterations,config):
    binding=measurement_binding(config)
    if type(iterations) is not int or iterations<1:raise ValueError('positive iterations')
    start=time.monotonic()
    for _ in range(iterations):call()
    row=dict.fromkeys(CAPS,0)
    row.update(measured=True,iterations=iterations,cpu_wall_seconds=(time.monotonic()-start)/iterations)
    row['binding']=binding;row['evidence']=digest(row);return row
