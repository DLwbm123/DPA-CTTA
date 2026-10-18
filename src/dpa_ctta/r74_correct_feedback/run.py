"""Fixed six-state supervised diagnostic; future labels remain scorer-only."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from ..r51_same_state.run import read, save, run_child

SEEDS = (20260907, 17011, 29009)
BRANCHES = ('NO_UPDATE', 'NATIVE', 'CORRECT')


def decision(delta, versus_no_update):
    if delta.shape != (3, 2, 50, 2) or versus_no_update.shape != delta.shape:
        raise ValueError('fixed seed/order/content/channel dimensions required')
    state = delta.mean((2, 3))
    cells = delta.mean((0, 2))
    checks = dict(mean_at_least_005=float(delta.mean()) >= .05,
                  both_orders_positive=bool((state.mean(0) > 0).all()),
                  five_states_positive=int((state > 0).sum()) >= 5,
                  worst_order_channel_ge_minus05=float(cells.min()) >= -.5,
                  correct_vs_no_update_nonnegative=float(versus_no_update.mean()) >= 0)
    return dict(allocation_signal=all(checks.values()), checks=checks,
                mean_pp=float(delta.mean()), order_pp=state.mean(0).tolist(),
                state_pp=state.tolist(), worst_order_channel_pp=float(cells.min()),
                positive_states=int((state > 0).sum()))


def conditional_interval(delta):
    rng = np.random.default_rng(20261018)
    draws = []
    for _ in range(2000):
        draws.append(np.mean([delta[:, o, rng.integers(0, 50, 50), :].mean() for o in (0, 1)]))
    return np.quantile(draws, [.025, .975]).tolist()


def self_check():
    import torch
    from torch.nn import functional as F
    from PIL import Image
    from ..source_io import read_mask
    from ..r72_crop_target.method import delta_loss
    torch.set_num_threads(2)
    z = torch.tensor([[[[-2., .2], [.7, -1.]], [[1., -.3], [-2., .4]]]], requires_grad=True)
    q = torch.full_like(z, .3)
    y = torch.tensor([[[[1., 1.], [0., 0.]], [[1., 0.], [0., 0.]]]], requires_grad=True)
    w = torch.tensor([[[[1., 2.], [3., 4.]], [[2., 1.], [4., 3.]]]])
    base = (w*F.binary_cross_entropy_with_logits(z, q, reduction='none')).mean()
    got = base+delta_loss(z, q, y, w)
    wanted = (w*F.binary_cross_entropy_with_logits(z, y.detach(), reduction='none')).mean()
    grads = torch.autograd.grad(got, (z, y), retain_graph=True, allow_unused=True)
    assert torch.allclose(got, wanted) and grads[1] is None
    assert torch.allclose(grads[0], torch.autograd.grad(wanted, z)[0], atol=1e-7)
    assert delta_loss(z, q, q, w) == 0
    raw = io.BytesIO()
    Image.fromarray(np.array([[0, 128], [255, 255]], dtype=np.uint8)).save(raw, format='PNG')
    raw.seek(0)
    label = read_mask(raw, 'fundus', [2, 2])
    assert label.shape == (1, 2, 512, 512)
    assert label[0, :, 0, 0].tolist() == [1., 1.]
    assert label[0, :, 0, -1].tolist() == [1., 0.]
    a = np.full((3, 2, 50, 2), .06)
    assert decision(a, a)['allocation_signal']
    a[:, 1] = -.01
    assert not decision(a, a)['allocation_signal']
    assert conditional_interval(np.zeros_like(a)) == [0., 0.]
    print('FEEDBACK_TARGET_AND_GATE_SELF_CHECK_PASS', flush=True)


def load_inputs(root):
    import torch
    jobs = read(root/'private/jobs.json')
    if [(j['seed'], j['order']) for j in jobs] != [(s, o) for s in SEEDS for o in (0, 1)]:
        raise ValueError('six fixed states required')
    anchors = []
    for job in jobs:
        p = root/'private'/job['anchor']
        raw = p.read_bytes()
        meta = read(p.with_suffix('.json'))
        # Existing journal integrity is mandatory; no new full-data scan.
        if hashlib.sha256(raw).hexdigest() != meta['sha256']:
            raise ValueError('journal anchor integrity mismatch')
        payload = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        h = payload['host']
        if payload['schema'] != 'R8_TARGET_JOURNAL_V1' or payload['identity'] != job['identity']:
            raise ValueError('anchor binding mismatch')
        if meta['visits'] != 1900 or h['visits'] != 1900 or h['steps'] != 1900 or h['context']['sha256'] != job['identity']['context_sha256']:
            raise ValueError('anchor position mismatch')
        anchors.append(h)
    images = [read(root/'private'/f'images_o{o}.json') for o in (0, 1)]
    feedback = read(root/'private/feedback.json')
    for o, rows in enumerate(images):
        if len(rows) != 51 or len({r['image_sha256'] for r in rows}) != 51:
            raise ValueError('fixed feedback/future content coverage')
        if any(set(r) != {'image_path', 'image_sha256', 'image_size'} for r in rows):
            raise ValueError('future manifest is not image-only')
        if feedback[o]['image_sha256'] != rows[0]['image_sha256']:
            raise ValueError('feedback identity mismatch')
    if {r['image_sha256'] for r in images[0]} & {r['image_sha256'] for r in images[1]}:
        raise ValueError('two tails unexpectedly overlap')
    return jobs, anchors, images, feedback


def output_bytes(root):
    return sum(p.stat().st_size for d in ('predictions', 'diagnostics', 'logs', 'public', 'attempts', 'processes')
               for p in (root/d).glob('*') if p.is_file()) + sum(p.stat().st_size for p in root.glob('*.json'))


def worker(c):
    import torch
    from .method import Host, readonly, branches
    from ..r37_cw_orientation.method import Host as RetainedHost, candidates
    from ..r19_model_only.runtime import weights, image
    from ..r19_model_only.method import equal
    from ..r36_incremental_modules.run import filesystem_guard
    from ..r9_current_first.assets import gpu_policy
    from ..r7_target_screen.runner import TargetReader, file_identity
    root = Path(c['output_root'])
    start = float(os.environ['DIAG_STARTED'])
    host = None
    status = dict(status='QUALIFYING', completed_states=0, future_predictions=0, started=start)
    save(root/'RUN_STATE.json', status)
    def guard():
        if time.time()-start > c['gpu_cap_seconds']-30:
            raise TimeoutError('fixed GPU process budget')
        if output_bytes(root) > c['output_cap_bytes']-4*1024**2:
            raise OSError('fixed output budget')
    guard.extra = dict(image_accesses=0)
    try:
        torch.set_num_threads(2)
        torch.set_num_interop_threads(2)
        gpu_policy(c['gpu'])
        permit, opened, rejected = filesystem_guard(c)
        jobs, anchors, images, feedback = load_inputs(root)
        initial = weights(c)
        reader = TargetReader(c['target_root'], 256*1024**2, 'image')
        masks = TargetReader(c['target_root'], 256*1024**2, 'mask')
        truths = []
        for row in feedback:
            # Reuse the existing exact-one-asset capability for the authorized feedback label.
            label_path = Path(row['mask_path']).resolve()
            try:
                label_path.read_bytes()
            except PermissionError:
                pass
            else:
                raise ValueError('feedback label readable without explicit capability')
            permit['image'] = label_path
            try:
                truths.append(masks.read(row))
            finally:
                permit['image'] = None
        if len(masks.used) != 2:
            raise ValueError('only two feedback labels are authorized')
        checks = []
        step_times = []
        probe_times = []
        for i, (job, anchor) in enumerate(zip(jobs, anchors)):
            guard()
            o = job['order']
            x = image(reader, images[o][0], permit, guard)
            host = RetainedHost(initial, candidates()[3], job['seed'], anchor['context']['sha256'], device='cuda:0')
            host.restore(anchor)
            if not equal(host.snapshot(), anchor):
                raise ValueError('restored anchor differs')
            tick = time.time()
            ref, _ = host.step(x)
            torch.cuda.synchronize()
            step_times.append(time.time()-tick)
            ref_state = host.snapshot()
            historical = np.unpackbits(np.frombuffer((root/'private'/job['historical_mask']).read_bytes(), dtype=np.uint8)).reshape(2, 512, 512)
            if not np.array_equal((ref[0] >= 0).numpy(), historical):
                raise ValueError('historical native arrival1901 mismatch')
            host.close()
            host = Host(initial, candidates()[3], job['seed'], anchor['context']['sha256'], device='cuda:0')
            states, logits, targets, detail = branches(host, anchor, x, truths[o], qualify=True)
            if not torch.equal(logits[1], ref[0]) or not equal(states[1], ref_state):
                raise ValueError('new native host differs from retained host')
            for state in states:
                tick = time.time()
                a = readonly(host, state, x)
                b = readonly(host, state, x)
                torch.cuda.synchronize()
                probe_times.append((time.time()-tick)/2)
                if not torch.equal(a, b):
                    raise ValueError('read-only repeat mismatch')
            checks.append(dict(seed=job['seed'], order=o, status='PASS', historical_native_exact=True,
                               native_full_state_exact=True, repeated_branches_exact=True,
                               original_weak_targets_rng_exact=True, correct_target_truth_exact=True,
                               readonly_state_and_output_exact=True))
            host.close()
            host = None
        estimate = 1.5*(12*max(step_times)+906*max(probe_times))+60
        if time.time()-start+estimate > c['gpu_cap_seconds']-30:
            raise TimeoutError('formal matrix not admitted within remaining budget')
        save(root/'QUALIFICATION.json', dict(status='PASS', checks=checks, formal_estimate_seconds=estimate,
             max_step_seconds=max(step_times), max_probe_seconds=max(probe_times)))
        status['status'] = 'RUNNING_FIXED_FUTURE'
        save(root/'RUN_STATE.json', status)
        details = []
        for i, (job, anchor) in enumerate(zip(jobs, anchors)):
            guard()
            o = job['order']
            host = Host(initial, candidates()[3], job['seed'], anchor['context']['sha256'], device='cuda:0')
            x = image(reader, images[o][0], permit, guard)
            states, logits, targets, info = branches(host, anchor, x, truths[o])
            details.extend(dict(seed=job['seed'], order=o, **d) for d in info)
            (root/'predictions'/f'feedback_{i}.bits').write_bytes(np.packbits((logits >= 0).numpy()).tobytes())
            np.savez(root/'diagnostics'/f'feedback_{i}.npz', probabilities=logits.sigmoid().numpy(), targets=targets.numpy())
            with (root/'predictions'/f'future_{i}.bits').open('xb') as f:
                for j, row in enumerate(images[o][1:]):
                    guard()
                    pixels = image(reader, row, permit, guard)
                    for branch_state in states:
                        z = readonly(host, branch_state, pixels)
                        f.write(np.packbits((z[0] >= 0).numpy()).tobytes())
                        status['future_predictions'] += 1
                f.flush()
            host.restore(anchor)
            if not equal(host.snapshot(), anchor):
                raise ValueError('final anchor differs')
            host.check_frozen(True)
            host.close()
            host = None
            status['completed_states'] += 1
            save(root/'RUN_STATE.json', status)
            print('completed states', status['completed_states'], 'future predictions', status['future_predictions'], flush=True)
        for rr in (reader, masks):
            for path, (_, identity) in rr.used.items():
                if file_identity(path) != identity:
                    raise ValueError('used asset metadata changed')
        save(root/'diagnostics/updates.json', details)
        save(root/'private/ACCESS_AUDIT.json', dict(feedback_label_contents=len(masks.used), future_label_reads=0,
             image_accesses=guard.extra['image_accesses'], opened_paths=sorted(opened), rejected=rejected))
        guard()
        status.update(status='GPU_COMPLETE', peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      output_bytes=output_bytes(root))
        return 0
    except BaseException as e:
        status.update(status='NOT_EVALUABLE', reason=str(e), error_type=type(e).__name__)
        traceback.print_exc()
        return 1
    finally:
        if host is not None:
            host.close()
        status.update(ended=time.time(), worker_seconds=time.time()-start)
        save(root/'RUN_STATE.json', status)


def score(c):
    from ..r7_target_screen.runner import TargetReader
    from ..r24_c_context.run import hard_metrics
    from ..r51_same_state.score import csvout
    from scipy.ndimage import binary_dilation, binary_erosion
    root = Path(c['output_root'])
    status = read(root/'RUN_STATE.json')
    receipt = read(root/'processes/worker.json')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '' or receipt['active'] or receipt['exit_code'] != 0:
        raise ValueError('retired successful GPU worker required')
    if status['status'] != 'GPU_COMPLETE' or status['completed_states'] != 6 or status['future_predictions'] != 900:
        raise ValueError('complete fixed matrix required')
    jobs = read(root/'private/jobs.json')
    rows = read(root/'scorer/windows.json')
    reader = TargetReader(c['target_root'], 256*1024**2, 'mask')
    values = np.empty((3, 2, 50, 3, 2))
    current = []
    for i, job in enumerate(jobs):
        s = SEEDS.index(job['seed'])
        o = job['order']
        raw = (root/'predictions'/f'future_{i}.bits').read_bytes()
        if len(raw) != 50*3*65536:
            raise ValueError('future bit coverage')
        masks = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(50, 3, 2, 512, 512).astype(bool)
        for j, row in enumerate(rows[o][1:]):
            truth = reader.read(row)
            for b in range(3):
                values[s, o, j, b] = [100*v['dice'] for v in hard_metrics(masks[j, b], truth)]
        truth = reader.read(rows[o][0])
        y = truth.numpy()[0].astype(bool)
        raw = (root/'predictions'/f'feedback_{i}.bits').read_bytes()
        if len(raw) != 3*65536:
            raise ValueError('feedback bit coverage')
        hard = np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).reshape(3, 2, 512, 512).astype(bool)
        with np.load(root/'diagnostics'/f'feedback_{i}.npz') as f:
            fields = np.concatenate((f['probabilities'], f['targets']), axis=0)
        if fields.shape != (5, 2, 512, 512) or not np.isfinite(fields).all() or not np.array_equal(fields[4], y):
            raise ValueError('captured feedback fields invalid')
        for k, name in enumerate(BRANCHES+('NATIVE_TARGET', 'CORRECT_TARGET')):
            metrics = hard_metrics(hard[k] if k < 3 else fields[k] >= .5, truth)
            for ch, metric in enumerate(metrics):
                boundary = (
                    binary_dilation(y[ch], structure=np.ones((7, 7)), border_value=0) ^
                    binary_erosion(y[ch], structure=np.ones((7, 7)), border_value=0))
                error = (fields[k, ch]-y[ch])**2
                regions = dict(all=np.ones_like(y[ch]), foreground=y[ch], background=~y[ch], boundary=boundary)
                record = dict(seed=job['seed'], order=o, branch=name, channel=('OD', 'OC')[ch],
                              Dice=100*metric['dice'], FP=metric['FP'], FN=metric['FN'])
                record.update({f'Brier_{region}':float(error[mask].mean()) if mask.any() else None for region, mask in regions.items()})
                current.append(record)
    reader.after_check()
    np.save(root/'scorer/future_scores.private.npy', values)
    cells = []
    contrasts = []
    for name, a, b in (('CORRECT-NATIVE', 2, 1), ('CORRECT-NO_UPDATE', 2, 0), ('NATIVE-NO_UPDATE', 1, 0)):
        delta = values[:, :, :, a]-values[:, :, :, b]
        contrasts.append(dict(contrast=name, mean_pp=float(delta.mean()), ci95_conditional_pp=conditional_interval(delta),
                              order_pp=delta.mean((0, 2, 3)).tolist(), state_pp=delta.mean((2, 3)).tolist()))
        for s, seed in enumerate(SEEDS):
            for o in (0, 1):
                for ch, channel in enumerate(('OD', 'OC')):
                    cells.append(dict(contrast=name, seed=seed, order=o, domain=('Drishti_GS', 'REFUGE')[o],
                                      channel=channel, contents=50, delta_pp=float(delta[s, o, :, ch].mean())))
    result = decision(values[:, :, :, 2]-values[:, :, :, 1], values[:, :, :, 2]-values[:, :, :, 0])
    result.update(status='COMPLETE', contrasts=contrasts, future_contents=100, feedback_contents=2,
                  states=6, future_labels_online=0, patient_linkage='UNKNOWN', independent_validation=False,
                  supervised_mechanism_diagnostic=True, automatic_followup=False)
    out = root/'public'
    save(out/'RESULTS.json', result)
    csvout(out/'FUTURE_CELLS.csv', cells)
    csvout(out/'FEEDBACK_CELLS.csv', current)
    csvout(out/'UPDATE_DIAGNOSTICS.csv', read(root/'diagnostics/updates.json'))
    save(out/'QUALIFICATION.json', read(root/'QUALIFICATION.json'))
    save(out/'COMPLETION_AUDIT.json', dict(states=6, current_masks=18, future_masks=900, total_masks=918,
         feedback_label_contents_online=2, future_label_reads_online=0, worker_retired_exit=0,
         no_future_updates=True, output_bytes=output_bytes(root)))
    status.update(status='COMPLETE', scored_predictions=918)
    save(root/'RUN_STATE.json', status)
    return 0


def main():
    mode = os.environ['DIAG_MODE']
    if mode == 'self_check':
        self_check()
        return 0
    c = read(os.environ['DIAG_CONFIG'])
    root = Path(c['output_root'])
    if mode == 'worker':
        return worker(c)
    if mode == 'score':
        return score(c)
    if mode != 'controller':
        raise ValueError('unknown mode')
    with (root/'started.json').open('x') as f:
        json.dump(dict(at=time.time(), pid=os.getpid()), f)
    if read(root/'CPU_CHECK.json')['status'] != 'PASS' or c['source_sha'] == 'PREPARATION':
        raise ValueError('frozen source and CPU check required')
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
    gpus = {int(i):(u.strip(), int(m)) for i, u, m in (line.split(',') for line in raw.splitlines())}
    uuid, free = gpus[c['gpu']['physical_id']]
    if uuid != c['gpu']['uuid'] or free < 6144:
        raise RuntimeError('GPU admission failed; no fallback')
    save(root/'GPU_ADMISSION.json', dict(free_MiB=free, **c['gpu']))
    attempts = []
    try:
        attempts.append(run_child(c, 'worker', c['gpu_cap_seconds'], True))
        if attempts[-1]['status'] == 'COMPLETE':
            attempts.append(run_child(c, 'score', c['cpu_cap_seconds'], False))
    finally:
        ledger = dict(attempts=attempts, gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),
                      cpu_worker_seconds=c['prior_cpu_seconds']+sum(a['cpu_worker_seconds'] for a in attempts),
                      preparation_cpu_seconds=c['prior_cpu_seconds'], all_workers_retired=True,
                      gpu_cap_seconds=c['gpu_cap_seconds'], scorer_cap_seconds=c['cpu_cap_seconds'])
        save(root/'RESOURCE_LEDGER.json', ledger)
        save(root/'public/RESOURCE_LEDGER.json', ledger)
    return 0 if len(attempts) == 2 and all(a['status'] == 'COMPLETE' for a in attempts) else 1


if __name__ == '__main__':
    raise SystemExit(main())
