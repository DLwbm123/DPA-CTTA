"""Frozen-head amplitude comparison; this worker never opens segmentation labels."""
import copy
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .method import match, random_direction
from ..r78_cross_patient.objective import BNState, Field, ParameterPredictor
from ..r78_cross_patient.run import FINAL, SEED, adapt, direct, matched, normalize, perturb, seed
from ..r51_same_state.run import read, save, run_child

ARMS = ('B0', 'D', 'C_FULL', 'F', 'X_RAW', 'X_MATCH', 'P_MATCH', 'S_MATCH',
        'H_MATCH', 'C_A_MATCH', 'X_NEG', 'RANDOM_MATCH')
MODES = ('SELF', 'CROSS_MATCH', 'CROSS_MISMATCH')
PARITY = dict(B0='B0', D='D', C_FULL='C_FULL', F='F', X_RAW='X')


def guard_assets(c, panel):
    root = Path(c['output_root']).resolve();prior = Path(c['prior_root']).resolve()
    data = Path(c['source_root']).resolve()
    images = {Path(r['image_path']).resolve() for role in panel.values() for r in role}
    opened = set()
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        if p.is_relative_to(root/'scorer') or p.is_relative_to(prior/'scorer') or p.is_relative_to(prior/'public'):
            raise PermissionError('labels and prior outcome scores denied to worker')
        if p.is_relative_to(data):
            if p not in images:
                raise PermissionError('only the 16 registered evaluation RGB files are permitted')
            opened.add(str(p))
    sys.addaudithook(audit)
    return opened


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels
    from ..r37_cw_orientation.method import Host as Native, candidates
    root = Path(c['output_root']);prior = Path(c['prior_root']);started = time.time()
    status = dict(status='QUALIFYING', started=started, predictions=0, baseline_parity=0)
    save(root/'RUN_STATE.json', status)
    native = None
    try:
        if c['rdc'] or c['engram'] or c['new_training'] or c['grouping'] != 'CONTENT_DEVELOPMENT_USER_AUTHORIZED':
            raise ValueError('frozen scope mismatch')
        old_ledger = read(prior/'RESOURCE_LEDGER.json')
        if not old_ledger['all_workers_retired'] or any(a['exit_code'] for a in old_ledger['attempts']):
            raise ValueError('prior run must have completed successfully')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        panel = read(root/'private/panel.json')
        if set(panel) != {'support', 'query'} or any(len(rows) != 8 for rows in panel.values()):
            raise ValueError('fixed evaluation panel')
        if len({r['group_id'] for rows in panel.values() for r in rows}) != 16:
            raise ValueError('content overlap')
        opened = guard_assets(c, panel)
        for denied in (root/'scorer/panel.json', prior/'scorer/panel.json', prior/'public/RESULTS.json'):
            try:
                denied.read_bytes()
            except PermissionError:
                pass
            else:
                raise AssertionError('file access guard failed')
        state = torch.load(c['checkpoint_path'], map_location='cpu', weights_only=True)
        model, _ = build_reference_model('fundus');model.load_state_dict(state);model.cuda()
        host = BNState(model)
        if host.dimensions != 512:
            raise ValueError('fixed 512-dimensional state required')
        before = {n: p._version for n, p in model.named_parameters()}
        heads = {};bindings = {}
        for arm, epoch in dict(X=100, P=100, S=150, F=100, D=150).items():
            file = prior/'private'/f'head_{arm}.pt'
            payload = torch.load(file, map_location='cpu', weights_only=True)
            if payload['selected_episode'] != epoch:
                raise ValueError('prior selected checkpoint changed')
            head = ParameterPredictor(512) if arm == 'F' else Field()
            head.load_state_dict(payload['state_dict']);head.cuda().eval().requires_grad_(False)
            heads[arm] = head
            bindings[arm] = dict(selected_episode=epoch, parameters=sum(p.numel() for p in head.parameters()), bytes=file.stat().st_size)
        images = {role: [read_pixels(r['image_path'], 'fundus', r['image_size']) for r in rows] for role, rows in panel.items()}
        old_rows = read(prior/'private/predictions.json')
        old_lookup = {(r['pair'], r['condition'], r['arm'], r['mode']): r['index'] for r in old_rows}
        native = Native(state, candidates()[3], SEED, 'source_episode_reset', device='cuda:0')
        initial = native.snapshot();native_rng = copy.deepcopy(native.native.rng)
        null = random_direction(512, 'cuda:0')
        records = [];checks = [];total_match = 0;forward_count = 0
        def count(*args):
            nonlocal forward_count
            forward_count += 1
        handle = model.register_forward_hook(count)
        def record(pair, condition, arm, mode, z, z0, diagnostics, tick):
            pred = z.detach().cpu().numpy()[0] >= 0
            if arm in PARITY:
                old_index = old_lookup[pair, condition, PARITY[arm], mode]
                raw = (prior/'predictions'/f'{old_index:03d}.bits').read_bytes()
                previous = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(2, 512, 512).astype(bool)
                if not np.array_equal(pred, previous):
                    raise ValueError('frozen baseline parity failed: '+arm)
                status['baseline_parity'] += 1
            dz = z.to(z0)-z0
            dp = z.to(z0).sigmoid()-z0.sigmoid()
            diagnostics.update(logit_rms=float(dz.square().mean().sqrt()),
                               probability_abs_mean=float(dp.abs().mean()),
                               changed_pixels=int((pred != (z0.detach().cpu().numpy()[0] >= 0)).sum()))
            i = len(records)
            with (root/'predictions'/f'{i:04d}.bits').open('xb') as f:
                f.write(np.packbits(pred).tobytes())
            records.append(dict(index=i, pair=pair, condition=condition, arm=arm, mode=mode,
                                diagnostics=diagnostics, seconds=time.time()-tick))
            status['predictions'] = len(records)
        generation_started = time.time()
        for pair in range(8):
            for condition, spec in enumerate(FINAL):
                query_pixels = perturb(images['query'][pair], spec, SEED+pair)
                q = normalize(query_pixels).cuda()
                with torch.no_grad():
                    z0 = host(q, host.zero())
                    tick = time.time();record(pair, condition, 'B0', 'DIRECT', z0, z0, {}, tick)
                    tick = time.time();z = direct(host, heads['D'], q)
                    record(pair, condition, 'D', 'DIRECT', z, z0, {}, tick)
                for mode in MODES:
                    if time.time()-started > c['gpu_cap_seconds']-20:
                        raise TimeoutError('fixed worker cap')
                    tick = time.time()
                    k = (condition+1) % 4 if mode == 'CROSS_MISMATCH' else condition
                    pixels = query_pixels if mode == 'SELF' else perturb(images['support'][pair], FINAL[k], SEED+pair+100)
                    s = normalize(pixels).cuda()
                    a_f, df = adapt(host, heads['F'], 'F', s)
                    states = dict(F=(a_f, df));raws = {}
                    for arm in ('X', 'P', 'S', 'H'):
                        raws[arm] = adapt(host, heads.get(arm), arm, s)
                    raws['C_A'] = matched(host, pixels, native_rng)
                    states['X_RAW'] = raws['X']
                    for arm in ('X', 'P', 'S', 'H', 'C_A'):
                        value, detail = match(raws[arm][0], a_f)
                        states[arm+'_MATCH'] = (value, detail);checks.append(detail);total_match += 1
                    value, detail = match(-raws['X'][0], a_f)
                    states['X_NEG'] = (value, detail);checks.append(detail);total_match += 1
                    value, detail = match(null, a_f)
                    states['RANDOM_MATCH'] = (value, detail);checks.append(detail);total_match += 1
                    for arm, (a, diagnostics) in states.items():
                        with torch.no_grad():
                            z = host(q, a)
                        record(pair, condition, arm, mode, z, z0, dict(diagnostics), tick)
                    native.restore(initial);_, diagnostic = native.step(pixels)
                    z = native._readonly(native.native.model, query_pixels)
                    record(pair, condition, 'C_FULL', mode, z, z0, diagnostic['diagnostics'], tick)
                save(root/'RUN_STATE.json', status)
                if pair == condition == 0:
                    projection = time.time()-started+31*(time.time()-generation_started)*1.5+30
                    if projection > c['gpu_cap_seconds']-20:
                        raise TimeoutError('fixed matrix not admitted by first complete condition')
                    save(root/'QUALIFICATION.json', dict(status='PASS', projected_worker_seconds=projection,
                         parameter_dimension=512, matched_states_checked=total_match, baseline_parity=status['baseline_parity'],
                         heads=bindings, new_training=False, label_reads=0))
                    status['status'] = 'GENERATING'
        if len(records) != 1024 or status['baseline_parity'] != 352 or total_match != 672:
            raise ValueError('incomplete fixed matrix or parity/matching coverage')
        if any(p._version != before[n] or p.grad is not None for n, p in model.named_parameters()):
            raise ValueError('frozen backbone changed')
        handle.remove()
        save(root/'private/predictions.json', records)
        save(root/'MATCHING_AUDIT.json', dict(status='PASS', matched_states=total_match,
             maximum_absolute_norm_error=max(d['norm_absolute_error'] for d in checks),
             minimum_direction_cosine=min(d['direction_cosine'] for d in checks if d['direction_cosine'] is not None),
             nonzero_direction_failures=0, baseline_parity=352, raw_backbone_forwards=forward_count,
             readout_timing_scope='Shared support computations are repeated in timestamps; not per-arm latency benchmarks'))
        save(root/'ACCESS_AUDIT.json', dict(source_image_files=len(opened), source_label_reads=0,
             final_label_reads=0, prior_final_score_reads=0, new_training=False, backbone_unchanged=True,
             prior_selected_heads_reused=True, patient_independence=False, query_development_exposed=True))
        used = sum(p.stat().st_size for d in ('private', 'predictions', 'logs') for p in (root/d).rglob('*') if p.is_file())
        if used > c['output_cap_bytes']:
            raise ValueError('output cap')
        status.update(status='GPU_COMPLETE', output_bytes=used, peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as error:
        status.update(status='NOT_EVALUABLE', error_type=type(error).__name__, reason=str(error))
        raise
    finally:
        if native is not None:
            native.close()
        status.update(ended=time.time(), gpu_worker_seconds=time.time()-started)
        save(root/'RUN_STATE.json', status)


def controller(c):
    root = Path(c['output_root'])
    with (root/'START_ONCE.json').open('x') as f:
        json.dump(dict(started=time.time(), pid=os.getpid()), f)
    attempts = []
    try:
        attempts.append(run_child(c, 'worker', c['gpu_cap_seconds'], True))
        if attempts[-1]['exit_code'] == 0 and read(root/'RUN_STATE.json')['status'] == 'GPU_COMPLETE':
            attempts.append(run_child(c, 'score', c['cpu_cap_seconds'], False))
    finally:
        save(root/'RESOURCE_LEDGER.json', dict(attempts=attempts, all_workers_retired=True, no_retry=True,
             gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),
             cpu_worker_seconds=sum(a['cpu_worker_seconds'] for a in attempts),
             gpu_cap_seconds=c['gpu_cap_seconds'], cpu_cap_seconds=c['cpu_cap_seconds']))
    return 0 if len(attempts) == 2 and all(a['exit_code'] == 0 for a in attempts) else 1


if __name__ == '__main__':
    config = read(os.environ['DIAG_CONFIG']);mode = os.environ['DIAG_MODE']
    if mode == 'controller':
        sys.exit(controller(config))
    elif mode == 'worker':
        worker(config)
    elif mode == 'score':
        from .score import score
        score(config)
    else:
        raise ValueError('unknown mode')
