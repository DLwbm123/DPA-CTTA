"""Four matched-age states, with exact diagonal replay against R32."""
import sys
import os
import time
from pathlib import Path
import torch
from .method import exchange
from ..r32_history_origin import online as prior
from ..r19_model_only.method import equal
from ..r19_model_only.runtime import weights, image
from ..r7_target_screen.runner import TargetReader
from ..r10_12h_core.run import read, save

HISTORIES = ('SS', 'SC', 'CS', 'CC')
RULES = ('UPDATE',)


def access_guard(c):
    root = Path(c['output_root']).resolve()
    checkpoint = Path(c['checkpoint_path']).resolve()
    references = [Path(c[k]).resolve() for k in ('snapshot_input_root', 'r32_target_root')]
    libraries = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    current = {'image': None}
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root/'scorer') or p.is_relative_to(root/'scores'):
            raise PermissionError('online label/score barrier')
        if p.suffix.lower() in ('.png','.jpg','.jpeg','.pt','.pth','.npy','.npz','.bits'):
            reference = p.suffix == '.bits' and any(p.is_relative_to(a) for a in references)
            if p not in (checkpoint,current['image']) and not reference and not p.is_relative_to(root/'target') and not any(p.is_relative_to(a) for a in libraries):
                raise PermissionError('unregistered model/data access')
    sys.addaudithook(audit)
    try: (root/'scorer/denial_probe.json').read_bytes()
    except PermissionError: pass
    else: raise AssertionError('label guard did not reject')
    return current


def exact_diagonal(dest, reference, arrivals):
    from ..r30_state_history.report import WIDTH
    for name in ('predictions.bits', 'pre.bits'):
        with (dest/name).open('rb') as a, (reference/name).open('rb') as b:
            for _ in range(arrivals):
                x, y = a.read(WIDTH), b.read(WIDTH)
                if len(x) != WIDTH or x != y:
                    raise ValueError('R32 diagonal pre/post replay mismatch')
            if a.read(1) or b.read(1):
                raise ValueError('R32 diagonal excess coverage')
    return arrivals


def stream(c, job, guard, permit, profiling=False):
    root = Path(c['output_root'])
    rows = read(root/f'private/ONLINE_o{job["order"]}.json')
    cases = read(root/'private/PLAN.json')[str(job['order'])]
    if len(rows) != 1951 or len(cases) != 2: raise ValueError('matrix changed')
    for case in cases: prior.validate_case(case, rows)
    if profiling: cases = [dict(cases[0], query=cases[0]['query'][:8])]
    dest = root/'target'/job['id']; dest.mkdir(exist_ok=False)
    tick = time.perf_counter()
    h = prior.StateHost(weights(c), 'C_CONT', job['seed'], job['id'])
    guard.meter.attach(h.native.model)
    init = time.perf_counter()-tick
    source = h.snapshot(); save(dest/'inventory.json',prior.inventory(h))
    reader = TargetReader(c['target_root'],256*1024**2,'image')
    branches = []; warm_times = []; diagonal_checks = 0; first_checks = 0
    try:
        _, schedule, replay = prior.replay(c,job,h,rows,cases,reader,guard,permit,count=8 if profiling else None)
        for case in cases:
            same, rng, t = prior.warm(h,source,case['same'],rows,reader,guard,permit); warm_times += t
            cross, _, t = prior.warm(h,source,case['cross'],rows,reader,guard,permit,schedule=rng); warm_times += t
            if same['steps'] != 64 or cross['steps'] != 64: raise ValueError('history clock')
            states = dict(SS=exchange(same,same), SC=exchange(same,cross), CS=exchange(cross,same), CC=exchange(cross,cross))
            if not equal(states['SS'],same) or not equal(states['CC'],cross): raise ValueError('diagonal state changed')
            if profiling:
                h.restore(same)
                for visit in case['query']:
                    _, _, schedule[visit] = prior.coupled_step(h,image(reader,rows[visit-1],permit,guard))
            first = {}
            for arm in HISTORIES:
                result = prior.query(c,job,h,states[arm],arm,'UPDATE',case,schedule,rows,reader,guard,permit,dest/f'{case["id"]}_{arm}_UPDATE',native_check=False)
                first[arm] = result.pop('first_pre')
                result['case'] = case['id']
                if not profiling and arm in ('SS','CC'):
                    history = 'SAME64' if arm == 'SS' else 'CROSS64'
                    checks = exact_diagonal(dest/f'{case["id"]}_{arm}_UPDATE',Path(c['r32_target_root'])/job['id']/f'{case["id"]}_{history}_UPDATE',len(case['query']))
                    result['reference_checks'] = checks; diagonal_checks += checks
                branches.append(result)
            if first['SS'] != first['SC'] or first['CS'] != first['CC']:
                raise ValueError('optimizer swap changed pre-update prediction')
            first_checks += 2
        query = sum(b['arrivals'] for b in branches)
        updates = replay['arrivals']+len(warm_times)+(8 if profiling else 0)+query
        expected = dict(model_forwards=8*updates+query,backward_calls=updates,optimizer_steps=updates,vjp_calls=0)
        if any(guard.meter.cost[k] != v for k,v in expected.items()): raise ValueError('physical operation accounting')
        result = dict(status='COMPLETE',initialization_seconds=init,replay=replay,history_updates=len(warm_times),history_seconds=warm_times,branches=branches,query_arrivals=query,label_reads=0,profile_calibration_updates=8 if profiling else 0,diagonal_query_checks=diagonal_checks,first_pre_invariance_checks=first_checks,expected_operations=expected,peak_reserved_bytes=torch.cuda.max_memory_reserved())
        save(dest/'COMPLETE.json',result)
        return result
    finally: h.close()


def profile(c,job,guard,permit):
    return stream(c,dict(job,seed=c['seeds'][0],order=0),guard,permit,profiling=True)
