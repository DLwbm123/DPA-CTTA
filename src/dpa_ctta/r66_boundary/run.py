"""Frozen feature/output diagnostic; one worker retires before label scoring."""
import json
import os
from pathlib import Path
import time
import numpy as np
from ..r51_same_state.run import read, save, run_child
from .geometry import full_resolution, apply_fields

BRANCHES = ('CW', 'LOCAL', 'SHUFFLE', 'SIGN0', 'SIGN1', 'SIGN2', 'SIGN3')


def check_image_files(reader, permit):
    from ..r7_target_screen.runner import confined, file_identity
    from ..r7_source_prep.registry import verified
    for path, (expected, identity) in reader.used.items():
        permit['image'] = Path(path).resolve(strict=True)
        try:
            confined(reader.root, path)
            if file_identity(path) != identity: raise ValueError('target changed after read')
            verified(path, expected, reader.limit, reader.counts)
        finally: permit['image'] = None


def outputs(feature, masks, index):
    local, field, audit = full_resolution(feature, masks)
    rng = np.random.default_rng(20261017 + index)
    shuffled = feature.reshape(256, -1)[:, rng.permutation(128*128)].reshape(feature.shape)
    null, _, null_audit = full_resolution(shuffled, masks)
    result = [masks, local, null]
    vetoes = []
    for _ in range(4):
        random_field = np.abs(field) * rng.choice((-1., 1.), size=field.shape)
        assert np.array_equal(np.abs(random_field), np.abs(field))
        output, veto = apply_fields(masks, random_field)
        result.append(output); vetoes.append(veto)
    audit.update(shuffle=null_audit, sign_veto_pixels=vetoes,
                 positive_fields=(field > 0).sum((1, 2)).tolist(),
                 negative_fields=(field < 0).sum((1, 2)).tolist(),
                 edited_pixels=[(m != masks).sum((1, 2)).tolist() for m in result])
    return np.stack(result), audit


def check(c):
    from .geometry import self_check
    report = self_check()
    masks = np.zeros((2, 512, 512), bool); masks[0, :, :256] = True
    f = np.ones((256, 128, 128), np.float32)
    a, audit = outputs(f, masks, 0)
    b, _ = outputs(f, masks, 0)
    assert np.array_equal(a, b) and all(np.array_equal(m, masks) for m in a)
    assert a.shape == (7, 2, 512, 512)
    xx = np.indices((128, 128))[1]
    p = 1/(1+np.exp(np.clip(4*(xx-64.3), -60, 60)))
    f = np.zeros_like(f); f[0] = p; f[1] = 1-p
    a, audit = outputs(f, masks, 1); b, _ = outputs(f, masks, 1)
    assert np.array_equal(a, b) and not np.array_equal(a[0], a[1])
    assert any(not np.array_equal(a[1], m) for m in a[3:])
    report.update(controls_deterministic=True, zero_fields_all_controls_identity=True)
    save(Path(c['output_root'])/'CPU_CHECK.json', report)


def worker(c, produce=outputs):
    import torch
    from ..r19_model_only.runtime import filesystem_guard, weights
    from ..r7_target_screen.runner import TargetReader
    from ..r9_current_first.assets import gpu_policy
    from ..integrations.ctta_suite import build_reference_model
    from ..hosts.vptta import model_input_from_pixels
    root = Path(c['output_root']); started = time.time()
    state = dict(status='QUALIFYING', formal_pairs=0, feature_images=0, started=started)
    save(root/'RUN_STATE.json', state)
    torch.set_num_threads(2); torch.set_num_interop_threads(2); gpu_policy(c['gpu'])
    panel = read(root/'private/images.json')
    if len(panel) != 48: raise ValueError('fixed panel coverage')
    permit, opened = filesystem_guard(c)
    for path in (root/'scorer/panel.json', Path(c['denial_mask_path'])):
        try: path.read_bytes()
        except PermissionError: pass
        else: raise AssertionError('target label guard')
    hook = None
    try:
        model, _ = build_reference_model('fundus')
        model.load_state_dict(weights(c)); model.cuda().eval().requires_grad_(False)
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False
                m.running_mean = m.running_var = m.num_batches_tracked = None
        before = {n: v.detach().cpu().clone() for n, v in model.state_dict().items()}
        captured = []
        hook = dict(model.named_modules())['up3'].register_forward_hook(lambda m, args, out: captured.append(out.detach()))
        reader = TargetReader(c['target_root'], 256*1024**2, 'image')
        traces = []
        for i, row in enumerate(panel):
            tick = time.time()
            if tick-started > c['gpu_cap_seconds']-30: raise TimeoutError('worker cap')
            permit['image'] = Path(row['image_path']).resolve(strict=True)
            try: pixels = reader.read(row)
            finally: permit['image'] = None
            with torch.no_grad(): model(model_input_from_pixels(pixels, 'fundus').cuda())
            feature = captured.pop().float().cpu().numpy()[0]
            if captured or feature.shape != (256, 128, 128) or not np.isfinite(feature).all():
                raise ValueError('native up3 interface')
            state['feature_images'] += 1
            if i == 0:
                with torch.no_grad(): model(model_input_from_pixels(pixels, 'fundus').cuda())
                again = captured.pop().float().cpu().numpy()[0]
                if not np.array_equal(feature, again): raise ValueError('frozen feature repeat failed')
            for order in (0, 1):
                pid = order*48+i
                raw = (root/'private/candidates'/f'{pid:03d}.bits').read_bytes()
                if len(raw) != 65536: raise ValueError('single CW input required')
                masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(2, 512, 512).astype(bool)
                result, audit = produce(feature, masks, pid)
                with (root/'predictions'/f'{pid:03d}.bits').open('xb') as f:
                    f.write(np.packbits(result).tobytes())
                traces.append(dict(pair=pid, **audit)); state['formal_pairs'] += 1
            if i == 0:
                projection = time.time()-started + 47*(time.time()-tick)*1.5 + 60
                if projection > c['gpu_cap_seconds']-30: raise TimeoutError('matrix budget not admitted')
                save(root/'QUALIFICATION.json', dict(status='PASS', up3_shape=list(feature.shape),
                    exact_feature_repeat=True, target_label_reads=0, predicted_worker_seconds=projection,
                    normalization='native minmax; eval current-image BN; no running statistics'))
                state['status'] = 'GENERATING'
            if (i+1) % 4 == 0: save(root/'RUN_STATE.json', state)
        after = model.state_dict()
        if set(before) != set(after) or any(not torch.equal(v, after[n].cpu()) for n, v in before.items()):
            raise ValueError('frozen source state changed')
        check_image_files(reader, permit)
        save(root/'private/pairs.json', sorted(traces, key=lambda r:r['pair']))
        save(root/'private/ACCESS_AUDIT.json', dict(opened_paths=sorted(opened), target_label_reads=0,
             source_image_reads=0, source_mask_reads=0, frozen_state_unchanged=True,
             feature_images=48, feature_forwards=49, peak_reserved_bytes=torch.cuda.max_memory_reserved()))
        artifact_bytes = sum(p.stat().st_size for name in ('private', 'predictions', 'logs')
                             for p in (root/name).rglob('*') if p.is_file())
        if artifact_bytes > c['output_cap_bytes']: raise ValueError('artifact cap')
        state['new_artifact_bytes'] = artifact_bytes
        state.update(status='GPU_COMPLETE', frozen_state_unchanged=True)
    except BaseException as e:
        state.update(status='NOT_EVALUABLE', reason=str(e)); raise
    finally:
        if hook is not None: hook.remove()
        state.update(ended=time.time(), worker_seconds=time.time()-started)
        save(root/'RUN_STATE.json', state)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import interval, csvout
    root = Path(c['output_root']); state = read(root/'RUN_STATE.json'); retired = read(root/'processes/worker.json')
    if os.environ['CUDA_VISIBLE_DEVICES'] != '' or state['status'] != 'GPU_COMPLETE' or state['formal_pairs'] != 96 or retired['active'] or retired['exit_code'] != 0:
        raise ValueError('complete successful worker retirement required')
    panel = read(root/'scorer/panel.json'); traces = read(root/'private/pairs.json')
    if len(panel) != 48 or [r['pair'] for r in traces] != list(range(96)): raise ValueError('coverage')
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask'); values = np.empty((48, 2, 7, 2))
    for i, row in enumerate(panel):
        label = reader.read(row)
        for order in (0, 1):
            raw = (root/'predictions'/f'{order*48+i:03d}.bits').read_bytes()
            if len(raw) != 7*65536: raise ValueError('prediction coverage')
            masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(7, 2, 512, 512).astype(bool)
            for b, m in enumerate(masks): values[i, order, b] = [100*x['dice'] for x in hard_metrics(m, label)]
    reader.after_check(); np.save(root/'scorer/scores.private.npy', values)
    domains = np.array([r['domain'] for r in panel])
    if len(set(domains)) != 4 or any(sum(domains == d) != 12 for d in set(domains)): raise ValueError('domain coverage')
    contrasts = dict(local_minus_CW=values[:, :, 1]-values[:, :, 0],
        local_minus_shuffle=values[:, :, 1]-values[:, :, 2],
        local_minus_random_sign_mean=values[:, :, 1]-values[:, :, 3:].mean(2))
    summaries = []; cells = []; absolute = []
    for name, delta in contrasts.items():
        summaries.append(dict(contrast=name, mean_pp=float(delta.mean()), ci95_pp=interval(delta, domains), order_pp=delta.mean((0, 2)).tolist()))
        for order in (0, 1):
            for domain in sorted(set(domains)):
                for ch, channel in enumerate(('OD', 'OC')):
                    cells.append(dict(contrast=name, order=order, domain=domain, channel=channel,
                        delta_pp=float(delta[domains == domain, order, ch].mean())))
    for order in (0, 1):
        for domain in sorted(set(domains)):
            for b, branch in enumerate(BRANCHES):
                for ch, channel in enumerate(('OD', 'OC')):
                    absolute.append(dict(order=order, domain=domain, branch=branch, channel=channel,
                        dice_percent=float(values[domains == domain, order, b, ch].mean())))
    gate = (summaries[0]['mean_pp'] >= .1 and all(r['ci95_pp'][0] > 0 for r in summaries)
            and min(summaries[0]['order_pp']) > 0
            and min(r['delta_pp'] for r in cells if r['contrast'] == 'local_minus_CW') >= -1.)
    coverage = []
    for order in (0, 1):
        for domain in sorted(set(domains)):
            group = [traces[order*48+i] for i in np.flatnonzero(domains == domain)]
            for ch, channel in enumerate(('OD', 'OC')):
                coverage.append(dict(order=order, domain=domain, channel=channel,
                    boundaries=sum(t['channels'][ch]['boundary'] for t in group),
                    accepted=sum(t['channels'][ch]['accepted'] for t in group),
                    positive=sum(t['positive_fields'][ch] for t in group),
                    negative=sum(t['negative_fields'][ch] for t in group),
                    shuffle_accepted=sum(t['shuffle']['channels'][ch]['accepted'] for t in group),
                    mean_edit_pixels=np.mean([t['edited_pixels'] for t in group], axis=0)[:,ch].tolist()))
    result = dict(status='COMPLETE', directional_gate=bool(gate), contrasts=summaries, coverage=coverage,
                  independent_validation=False, output_only=True, source_training=False,
                  local_veto_pixels=sum(t['full_resolution_nesting_veto_pixels'] for t in traces),
                  shuffle_veto_pixels=sum(t['shuffle']['full_resolution_nesting_veto_pixels'] for t in traces),
                  sign_veto_pixels=np.sum([t['sign_veto_pixels'] for t in traces], axis=0).tolist())
    out = root/'public'; save(out/'RESULTS.json', result)
    csvout(out/'ALL_PAIRED_CELLS.csv', cells); csvout(out/'ALL_NEGATIVE_CELLS.csv', [r for r in cells if r['delta_pp'] < 0])
    csvout(out/'ALL_ABSOLUTE_CELLS.csv', absolute); save(out/'QUALIFICATION.json', read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json', dict(pairs=96, contents=48, orders=2, branches=7, predictions=672,
        target_label_reads_during_generation=0, source_state_unchanged=True, worker_retired_exit=0))
    state.update(status='COMPLETE', scored_predictions=672); save(root/'RUN_STATE.json', state)


def main():
    c = read(os.environ['DIAG_CONFIG']); root = Path(c['output_root']); mode = os.environ['DIAG_MODE']
    if mode == 'check': check(c); return 0
    if mode == 'worker': worker(c); return 0
    if mode == 'score': score(c); return 0
    if mode != 'controller' or read(root/'CPU_CHECK.json')['status'] != 'PASS': raise ValueError('entry mode/CPU gate')
    with (root/'started.json').open('x') as f: json.dump(dict(started=time.time(), pid=os.getpid()), f)
    attempts = []
    try:
        attempts.append(run_child(c, 'worker', c['gpu_cap_seconds'], True))
        if attempts[-1]['exit_code'] == 0: attempts.append(run_child(c, 'score', c['cpu_cap_seconds'], False))
    finally:
        ledger = dict(attempts=attempts, gpu_worker_seconds=c.get('prior_gpu_seconds', 0)+sum(a['gpu_worker_seconds'] for a in attempts),
            cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts),
            all_workers_retired=True, prior_cpu_seconds=c['prior_cpu_seconds'], prior_gpu_seconds=c.get('prior_gpu_seconds', 0))
        save(root/'RESOURCE_LEDGER.json', ledger); save(root/'public/RESOURCE_LEDGER.json', ledger)
    return 0 if len(attempts) == 2 and all(a['exit_code'] == 0 for a in attempts) else 1


if __name__ == '__main__': raise SystemExit(main())
