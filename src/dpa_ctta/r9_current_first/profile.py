"""Expanded measurement contract. Unit costs are never fabricated from plan arithmetic."""
from collections import Counter
import time
from .protocol import SPEC,CAPS,parse_recipe,digest
from .gradient import STEPS
from .target import PREDICTION_BYTES
from .admission import project


def units():
    result=Counter()
    for j in SPEC['source_tasks']:
        # Selected confirmation takes the maximum measured fit recipe for that route.
        recipe=j['recipe'];route=recipe.split('_')[0]
        kind='SELECTED_MAX' if 'SELECTED' in recipe else parse_recipe(recipe)[1]
        result[f'fit/{route}/{kind}']+=16000 # Includes all ten fresh-16k fallbacks.
        result[f'validation_episode/{route}']+=4*4*64 # 32 visits per complete episode
        result[f'source_setup/{route}']+=1
        result[f'fit_checkpoint/{route}']+=321
        result['journal_append']+=16000+4*4*64
        result[f'validation_checkpoint/{route}']+=4*4*3
        result[f'source_finalize/{route}']+=1
        if route!='MLP':
            result[f'calibration/{route}']+=4*256
            result[f'calibration_checkpoint/{route}']+=4*7
            result['journal_append']+=4*256
    # Seed aggregation is a pending scientific decision, not silently inferred.
    for arm in STEPS:
        result[f'lr_episode/{arm}/PER_SOURCE_SEED']=3*64
        result[f'lr_setup/{arm}/PER_SOURCE_SEED']=3
    result['lr_source_setup']=1
    for slot in SPEC['target_core_slots']+SPEC['target_final16k_slots_max']:
        n=19510 if slot['order']=='LONG10' else 1951
        result['online/'+slot['arm']]+=n;result['online_setup/'+slot['arm']]+=1
        result['target_journal_append']+=n
        result['target_checkpoint/'+slot['arm']]+=1+n//50
        result['cpu_score_visit']+=n;result['cpu_score_seal_visit']+=n
    return dict(result)


def projection(measurements,lr_seed_count,retained_disk,prior=None):
    if lr_seed_count not in (1,2,5):raise ValueError('registered LR source policy required')
    counts=units();expanded={}
    for name,n in counts.items():
        if name.endswith('/PER_SOURCE_SEED'):name=name.removesuffix('/PER_SOURCE_SEED');n*=lr_seed_count
        expanded[name]=n
    # One LONG10 probability stream at a time. CPU scoring must finish before next online.
    return project(expanded,measurements,retained_disk,19510*PREDICTION_BYTES,prior)


def measure(call,meter,iterations):
    """Use the real production kernel with source-only inputs; authorization belongs upstream."""
    if type(iterations) is not int or iterations<1:raise ValueError('positive measured iterations')
    import torch
    before=meter.cost.copy();torch.cuda.synchronize();start=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iterations):call()
    torch.cuda.synchronize();meter.check();elapsed=time.monotonic()-start
    row={k:(meter.cost[k]-before[k])/iterations for k in CAPS}
    row.update(measured=True,iterations=iterations,elapsed_seconds=elapsed,
               max_allocated_bytes=torch.cuda.max_memory_allocated(),max_reserved_bytes=torch.cuda.max_memory_reserved())
    row['gpu_seconds']=elapsed/iterations
    row['evidence']=digest(row);return row


def measure_cpu(call,iterations):
    if type(iterations) is not int or iterations<1:raise ValueError('positive iterations')
    start=time.monotonic()
    for _ in range(iterations):call()
    row=dict.fromkeys(CAPS,0)
    row.update(measured=True,iterations=iterations,cpu_wall_seconds=(time.monotonic()-start)/iterations)
    row['evidence']=digest(row);return row
