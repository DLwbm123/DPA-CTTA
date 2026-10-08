"""Label-free complete streams and delayed branches; all masks remain private."""
import copy
import json
from pathlib import Path
import time
import numpy as np
import torch
from .method import ARMS, ACTIONS, StateHost, FrozenHost, DelayHost, config, readonly, set_native_rng
from ..r24_c_context.method import Host as Reference
from ..r28_common_state.run import access_guard
from ..r20_model_only_search import runtime as base
from ..r10_12h_core.run import read, save
from ..r19_model_only.runtime import weights, image
from ..r19_model_only.method import clone, equal
from ..r7_target_screen.runner import TargetReader

WIDTH = 65536


def pack(z):
    return np.packbits(z.numpy()[0] >= 0).tobytes()


def reference_mask(path, visit):
    with path.open('rb') as f:
        f.seek((visit - 1) * WIDTH)
        raw = f.read(WIDTH)
    if len(raw) != WIDTH:
        raise ValueError('incomplete reference mask')
    return raw


def inventory(h):
    model = h.model if isinstance(h, FrozenHost) else h.native.model
    return dict(BN=[dict(name=k, affine=m.affine, track_running_stats=m.track_running_stats,
                        has_running_mean=m.running_mean is not None, training=m.training)
                    for k, m in model.named_modules() if isinstance(m, torch.nn.BatchNorm2d)],
                dropout=[dict(name=k, probability=m.p, training=m.training) for k, m in model.named_modules()
                         if isinstance(m, torch.nn.modules.dropout._DropoutNd)],
                trainable_names=[k for k, p in model.named_parameters() if p.requires_grad],
                adapter_trainable=0 if isinstance(h, FrozenHost) or h.adapter is None else sum(p.numel() for p in h.adapter.parameters()))


def stream(c, job, guard, permit, count=None, parity=False):
    root = Path(c['output_root']); arm = job['arm']
    rows = read(root/'private'/f'ONLINE_o{job["order"]}.json')
    if len(rows) != 1951:
        raise ValueError('full arrival stream changed')
    rows = rows[:count] if count else rows
    probes = {p['visit'] for p in read(root/'private/PLAN.json')[str(job['order'])]}
    if count:
        probes = {1, 32, 33}
    initial = weights(c); started = time.perf_counter()
    h = (FrozenHost(initial, arm, job['seed']) if arm in ARMS[:2]
         else StateHost(initial, arm, job['seed'], job['id']))
    guard.meter.attach(h.model if arm in ARMS[:2] else h.native.model)
    init_seconds = time.perf_counter() - started
    dest = root/'target'/job['id']; dest.mkdir(exist_ok=False)
    save(dest/'inventory.json', inventory(h))
    ref = None
    if parity and arm in ('C_CONT', 'ANCHOR'):
        ref = Reference(initial, config('C' if arm == 'C_CONT' else 'ANCHOR'), job['seed'], job['id'])
        guard.meter.attach(ref.native.model)
    reader = TargetReader(c['target_root'], 256*1024**2, 'image')
    times = []; checks = 0; diagnostics_forwards = 0
    try:
        with (dest/'predictions.bits').open('xb') as bits, (dest/'traces.jsonl').open('x') as traces, (dest/'pre.bits').open('xb') as pre:
            for i, row in enumerate(rows, 1):
                guard(); tick = time.perf_counter(); x = image(reader, row, permit, guard)
                reset = False; d = {}; before_mask = None
                if isinstance(h, StateHost):
                    previous_resets = h.resets; h.prepare_arrival(); reset = h.resets != previous_resets
                    if i in probes:
                        count_before = guard.meter.cost['model_forwards']
                        before_mask = pack(readonly(h, x)); pre.write(before_mask)
                        diagnostics_forwards += guard.meter.cost['model_forwards'] - count_before
                elif i in probes:
                    # Frozen arms have identical before/after predictions, without an extra forward.
                    before_mask = b''
                z, t = h.step(x); raw = pack(z); bits.write(raw)
                if before_mask == b'':
                    before_mask = raw; pre.write(raw)
                if ref is not None:
                    rz, _ = ref.step(x)
                    if not torch.equal(z, rz):
                        raise ValueError('unchanged reference replay mismatch')
                    for key in ('parameters', 'adam', 'adapter', 'native_rng'):
                        if not equal(h.snapshot()[key], ref.snapshot()[key]):
                            raise ValueError('reference state mismatch: '+key)
                    checks += 1
                if arm == 'ANCHOR' and not count and i in probes:
                    path = Path(c['snapshot_input_root'])/f's{job["seed"]}_o{job["order"]}'/f'masks_{i}.bits'
                    with path.open('rb') as f:
                        expected = f.read(WIDTH)
                    if raw != expected:
                        raise ValueError('ANCHOR differs from R28 saved reference')
                    checks += 1
                if isinstance(h, StateHost):
                    d = h.diagnostics()
                    d['update_norm'] = t['diagnostics']['BN_update_norm']
                    if not all(np.isfinite(v) for v in d.values()):
                        raise ValueError('nonfinite state diagnostics')
                if before_mask is not None:
                    d['mask_change_fraction'] = float(np.unpackbits(np.frombuffer(raw, np.uint8) ^ np.frombuffer(before_mask, np.uint8)).mean())
                torch.cuda.synchronize(); elapsed = time.perf_counter() - tick; times.append(elapsed)
                traces.write(json.dumps(dict(visit=i, reset=reset, pre_recorded=before_mask is not None,
                                             seconds=elapsed, diagnostics=d))+'\n')
                traces.flush(); bits.flush(); pre.flush()
                guard.extra['arrivals'] = i
            h.check_frozen(True)
            if h.visits != len(rows):
                raise ValueError('global arrival count mismatch')
        result = dict(status='COMPLETE', arrivals=len(rows), arm=arm, label_reads=0,
                      initialization_seconds=init_seconds, seconds=times, reference_checks=checks,
                      diagnostic_forwards=diagnostics_forwards, algorithm_forwards=(1 if arm in ARMS[:2] else 8)*len(rows),
                      resets=getattr(h, 'resets', 0), final_adam_local_step=None if arm in ARMS[:2] else h.native.steps)
        save(dest/'COMPLETE.json', result)
        return result
    finally:
        h.close()
        if ref is not None:
            ref.close()


def delay(c, job, guard, permit, smoke=False):
    root = Path(c['output_root']); order = job['order']; seed = job['seed']
    source_id = f's{seed}_o{order}'; source = Path(c['snapshot_input_root'])/source_id
    rows = read(root/'private'/f'ONLINE_o{order}.json')
    points = read(root/'private/PLAN.json')[str(order)]
    if smoke:
        points = points[:1]
    h = DelayHost(weights(c), config('GREEDY'), seed, source_id)
    guard.meter.attach(h.native.model)
    reader = TargetReader(c['target_root'], 256*1024**2, 'image')
    dest = root/'target'/job['id']; dest.mkdir(exist_ok=False)
    arrivals = 0; checks = 0; started = time.perf_counter()
    try:
        with (dest/'windows.jsonl').open('x') as windows:
            for index, point in enumerate(points):
                visit = point['visit']; length = min(8 if smoke else 64, len(rows)-visit)
                before = torch.load(source/f'before_{visit}.pt', map_location='cpu', weights_only=False)
                rng_plan = []
                for action in ACTIONS:
                    h.restore(before)
                    with (dest/f'w{index}_{action}.bits').open('xb') as bits:
                        for offset in range(length+1):
                            guard(); x = image(reader, rows[visit+offset-1], permit, guard)
                            if action == 'FULL':
                                rng_plan.append(clone(h.native.rng))
                            else:
                                set_native_rng(h, rng_plan[offset])
                            z = h.first(x, action) if offset == 0 else h.step(x)[0]
                            raw = pack(z); bits.write(raw); bits.flush()
                            if action == 'FULL':
                                if offset == 0:
                                    with (source/f'masks_{visit}.bits').open('rb') as f:
                                        expected = f.read(WIDTH)
                                    if raw != expected:
                                        raise ValueError('delayed FULL failed R28 current replay')
                                    checks += 1
                                if not smoke:
                                    path = root/'target'/f'B_ANCHOR_{source_id}'/'predictions.bits'
                                    if raw != reference_mask(path, visit+offset):
                                        raise ValueError('delayed FULL failed complete-stream reference')
                                    checks += 1
                            arrivals += 1; guard.extra['arrivals'] = arrivals
                        if h.visits != before['visits'] + length + 1:
                            raise ValueError('branch did not advance global stream position')
                        expected_step = before['steps'] + length + (action != 'ZERO')
                        if h.native.steps != expected_step:
                            raise ValueError('branch local Adam count mismatch')
                windows.write(json.dumps(dict(index=index, visit=visit, available_future=length, branch_count=3))+'\n'); windows.flush()
                guard.extra['windows'] = index+1
            h.check_frozen(True)
        result = dict(status='COMPLETE', windows=len(points), arrivals=arrivals, label_reads=0,
                      reference_checks=checks, seconds_per_arrival=(time.perf_counter()-started)/arrivals)
        save(dest/'COMPLETE.json', result)
        return result
    finally:
        h.close()


def profile(c, job, guard, permit):
    results = {}
    for arm in ARMS:
        j = dict(id=job['id']+'_'+arm, arm=arm, seed=c['seeds'][0], order=0)
        results[arm] = stream(c, j, guard, permit, count=34, parity=True)
    j = dict(id=job['id']+'_A', seed=c['seeds'][0], order=0)
    results['A'] = delay(c, j, guard, permit, smoke=True)
    return dict(status='PASS', profiles=results, peak_reserved_bytes=torch.cuda.max_memory_reserved())
