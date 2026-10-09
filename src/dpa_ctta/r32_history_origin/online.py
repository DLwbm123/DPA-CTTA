"""Offline counterfactual histories; the C updater receives images only."""
import json
import time
from pathlib import Path
import torch
from ..r30_state_history.method import StateHost, readonly, set_native_rng
from ..r30_state_history.online import pack, reference_mask, inventory
from ..r28_common_state.run import access_guard
from ..r19_model_only.method import clone, equal
from ..r19_model_only.runtime import weights, image
from ..r7_target_screen.runner import TargetReader
from ..r10_12h_core.run import read, save

HISTORIES = ('NATIVE', 'SOURCE', 'SAME64', 'CROSS64')
RULES = ('UPDATE', 'HOLD')


def validate_case(case, rows):
    same, cross, query = (case[k] for k in ('same', 'cross', 'query'))
    if len(same) != 64 or len(cross) != 64 or not query or same[-1] != case['pivot']:
        raise ValueError('registered history/query coverage')
    for seq in (same, cross, query):
        if seq != list(range(seq[0], seq[-1]+1)) or not 1 <= seq[0] <= seq[-1] <= len(rows):
            raise ValueError('invalid contiguous image indices')
    if query[0] != same[-1]+1 or cross[-1] >= same[0]:
        raise ValueError('history must precede query')
    ids = [{rows[i-1]['image_sha256'] for i in seq} for seq in (same, cross, query)]
    if any(len(a) != len(seq) for a, seq in zip(ids, (same,cross,query))) or ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2]:
        raise ValueError('history/query content overlap')


def coupled_step(h, x, rng_pair=None):
    if rng_pair is not None:
        set_native_rng(h, rng_pair[0])
    before = clone(h.native.rng)
    z, trace = h.step(x)
    after = clone(h.native.rng)
    if rng_pair is not None and not equal(after, rng_pair[1]):
        raise ValueError('augmentation RNG coupling failed')
    return z, trace, (before, after)


def branch_step(h, x, rule, rng_pair):
    if rule not in RULES:
        raise ValueError('unknown future rule')
    set_native_rng(h, rng_pair[0])
    pre = readonly(h, x)
    if rule == 'UPDATE':
        z, trace, _ = coupled_step(h, x, rng_pair)
    else:
        z = pre; h.visits += 1; trace = {}
    return pre, z, trace


def paths(c, job, arm):
    return Path(c['snapshot_input_root'])/f'B_{arm}_s{job["seed"]}_o{job["order"]}'/'predictions.bits'


def replay(c, job, h, rows, cases, reader, guard, permit, count=None):
    states = {}; schedule = {}; times = []; checks = 0
    wanted = {i for case in cases for i in case['query']}
    pivots = {case['pivot']: case['id'] for case in cases}
    for visit, row in enumerate(rows[:count] if count else rows, 1):
        guard(); tick = time.perf_counter(); x = image(reader, row, permit, guard)
        z, _, rng_pair = coupled_step(h, x)
        if pack(z) != reference_mask(paths(c, job, 'C_CONT'), visit):
            raise ValueError('native C replay differs from R30')
        checks += 1
        if visit in wanted:
            schedule[visit] = rng_pair
        if visit in pivots:
            states[pivots[visit]] = h.snapshot()
        torch.cuda.synchronize(); times.append(time.perf_counter()-tick)
        guard.extra['replay_arrivals'] = visit
    return states, schedule, dict(arrivals=len(times), seconds=times, reference_checks=checks)


def warm(h, source, indices, rows, reader, guard, permit, schedule=None):
    h.restore(source); draws = []; times = []
    for offset, visit in enumerate(indices):
        guard(); tick = time.perf_counter(); x = image(reader, rows[visit-1], permit, guard)
        _, _, rng_pair = coupled_step(h, x, None if schedule is None else schedule[offset])
        draws.append(rng_pair)
        torch.cuda.synchronize(); times.append(time.perf_counter()-tick)
        guard.extra['history_updates'] += 1
    if h.native.steps != 64:
        raise ValueError('matched history Adam clock must equal 64')
    return h.snapshot(), draws, times


def query(c, job, h, state, history, rule, case, schedule, rows, reader, guard, permit, dest, native_check=True):
    h.restore(state); start_steps = h.native.steps; start_visits = h.visits
    dest.mkdir(exist_ok=False); times = []; checks = 0; first_pre = None
    with (dest/'predictions.bits').open('xb') as post, (dest/'pre.bits').open('xb') as before, (dest/'traces.jsonl').open('x') as traces:
        for offset, visit in enumerate(case['query']):
            guard(); tick = time.perf_counter(); x = image(reader, rows[visit-1], permit, guard)
            pre, z, trace = branch_step(h, x, rule, schedule[visit])
            raw = pack(z); pre_raw = pack(pre); post.write(raw)
            if rule == 'UPDATE':
                before.write(pre_raw)
            if first_pre is None:
                first_pre = pre_raw
            reference = 'C_CONT' if history == 'NATIVE' and rule == 'UPDATE' and native_check else ('S_BATCH' if history == 'SOURCE' and rule == 'HOLD' else None)
            if reference:
                if raw != reference_mask(paths(c, job, reference), visit):
                    raise ValueError('counterfactual reference replay mismatch: '+reference)
                checks += 1
            diagnostics = h.diagnostics()
            if rule == 'UPDATE':
                diagnostics['consistency_loss'] = trace['native'][0]['diagnostics']['cal_consis_loss']
                diagnostics['update_norm'] = trace['diagnostics']['BN_update_norm']
            torch.cuda.synchronize(); elapsed = time.perf_counter()-tick; times.append(elapsed)
            traces.write(json.dumps(dict(visit=visit, query_offset=offset+1, seconds=elapsed, diagnostics=diagnostics))+'\n')
            post.flush(); before.flush(); traces.flush()
            guard.extra['query_arrivals'] += 1
    after = h.snapshot()
    if h.native.steps != start_steps+(len(case['query']) if rule == 'UPDATE' else 0) or h.visits != start_visits+len(case['query']):
        raise ValueError('future update/global clock mismatch')
    if rule == 'HOLD' and any(not equal(state[k],after[k]) for k in ('parameters','adam','buffers')):
        raise ValueError('HOLD changed learning state')
    h.check_frozen(True)
    result = dict(status='COMPLETE', arrivals=len(times), history=history, rule=rule,
                  start_adam_step=start_steps, end_adam_step=h.native.steps, reference_checks=checks,
                  seconds=times, label_reads=0, first_pre=first_pre)
    save(dest/'COMPLETE.json', {k:v for k,v in result.items() if k != 'first_pre'})
    return result


def stream(c, job, guard, permit, profiling=False):
    root = Path(c['output_root']); rows = read(root/f'private/ONLINE_o{job["order"]}.json')
    cases = read(root/'private/PLAN.json')[str(job['order'])]
    if len(rows) != 1951 or len(cases) != 2:
        raise ValueError('frozen full-stream plan changed')
    for case in cases:
        validate_case(case, rows)
    if profiling:
        cases = [dict(cases[0], query=cases[0]['query'][:8])]
    dest = root/'target'/job['id']; dest.mkdir(exist_ok=False)
    start = time.perf_counter(); h = StateHost(weights(c), 'C_CONT', job['seed'], job['id'])
    guard.meter.attach(h.native.model); init_seconds = time.perf_counter()-start
    source = h.snapshot(); save(dest/'inventory.json', inventory(h))
    reader = TargetReader(c['target_root'], 256*1024**2, 'image')
    branches = []; history_times = []
    try:
        states, schedule, replay_result = replay(c, job, h, rows, cases, reader, guard, permit, count=8 if profiling else None)
        if profiling:
            states[cases[0]['id']] = h.snapshot()
        for case in cases:
            same, rng, times = warm(h, source, case['same'], rows, reader, guard, permit); history_times += times
            cross, _, times = warm(h, source, case['cross'], rows, reader, guard, permit, schedule=rng); history_times += times
            options = dict(NATIVE=states[case['id']], SOURCE=source, SAME64=same, CROSS64=cross)
            if profiling:
                h.restore(same)
                for visit in case['query']:
                    _, _, schedule[visit] = coupled_step(h, image(reader, rows[visit-1], permit, guard))
            for history in HISTORIES:
                first = None
                for rule in RULES:
                    result = query(c, job, h, options[history], history, rule, case, schedule, rows, reader, guard, permit,
                                   dest/f'{case["id"]}_{history}_{rule}', native_check=not profiling)
                    first_pre = result.pop('first_pre')
                    if first is not None and first != first_pre:
                        raise ValueError('UPDATE/HOLD did not start from identical state')
                    first = first_pre; result['case'] = case['id']; branches.append(result)
        result = dict(status='COMPLETE', initialization_seconds=init_seconds, replay=replay_result,
                      history_updates=len(history_times), history_seconds=history_times, branches=branches,
                      query_arrivals=sum(b['arrivals'] for b in branches), label_reads=0,
                      profile_calibration_updates=8 if profiling else 0,
                      peak_reserved_bytes=torch.cuda.max_memory_reserved())
        updates = replay_result['arrivals'] + len(history_times) + result['profile_calibration_updates'] + sum(b['arrivals'] for b in branches if b['rule']=='UPDATE')
        expected = dict(model_forwards=8*updates+result['query_arrivals'],backward_calls=updates,optimizer_steps=updates,vjp_calls=0)
        if any(guard.meter.cost[k] != v for k,v in expected.items()):
            raise ValueError('physical counterfactual operation accounting')
        result['expected_operations'] = expected
        save(dest/'COMPLETE.json', result)
        return result
    finally:
        h.close()


def profile(c, job, guard, permit):
    return stream(c, dict(job, seed=c['seeds'][0], order=0), guard, permit, profiling=True)
