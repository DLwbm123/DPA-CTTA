"""One WRITE/USE-ONLY experiment with immutable counters and matched future RNG."""
import copy
import gzip
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from dpa_ctta.r85_migration.process import child, finish_child, alive, watchdog

SEEDS = (20260907, 17011, 29009)
PROBES = (8, 32, 56, 80, 104, 128, 152, 176)


def read(p):
    return json.loads(p.read_text())


def save(p, d):
    p.write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')


def usage(c):
    total = sum(max(p.stat().st_size, p.stat().st_blocks*512) for p in Path(c['campaign_root']).rglob('*') if not p.is_symlink())
    if total >= c['output_cap_bytes']-c['recovery_reserve_bytes']:
        raise RuntimeError('campaign allocated-byte recovery reserve')
    return total


def use_state(before, after):
    state = copy.deepcopy(before)
    for key in ('visits', 'rng', 'native_rng'):
        state['base'][key] = copy.deepcopy(after['base'][key])
    return state


def zero_delta(adam):
    import torch
    group = adam['param_groups'][0]
    assert len(adam['param_groups']) == 1 and group['weight_decay'] == 0 and not group['amsgrad']
    assert group['lr'] == 1e-4 and tuple(group['betas']) == (.9, .999) and group['eps'] == 1e-8
    result = []
    for key in group['params']:
        item = adam['state'][key]
        t = int(item['step'])+1
        m, v = item['exp_avg'].double(), item['exp_avg_sq'].double()
        update = -group['lr']*(.9*m/(1-.9**t))/((.999*v/(1-.999**t)).sqrt()+group['eps'])
        result.append(update.flatten())
    return torch.cat(result)


def signal_for(before, host):
    import torch
    g = torch.cat([p.grad.detach().flatten().cpu() for p in host.native.params]).double()
    d = zero_delta(before['base']['adam'])
    assert g.shape == d.shape == (19136,) and torch.isfinite(g).all() and torch.isfinite(d).all()
    norm_product = float(g.norm()*d.norm()); dot = float((g*d).sum())
    bound = 64*np.finfo(np.float64).eps*(g.numel()+1)*(float((g*d).abs().sum())+norm_product)
    return dict(signal=dot/norm_product if norm_product else None, dot=dot, norm_product=norm_product,
                sign_bound=bound, trigger=bool(norm_product and dot > bound)), g


def access_guard(c):
    root = Path(c['output_root']).resolve(); mount = Path(c['campaign_root']).resolve().parent.parent
    checkpoint = Path(c['checkpoint_path']).resolve()
    refs = {Path(p).resolve() for p in c['reference_predictions'].values()}
    libraries = [Path(sys.prefix).resolve(), Path(sys.base_prefix).resolve()]
    code = [Path(os.environ[k]).resolve() for k in ('PYTHONPATH', 'DPA_CTTA_BASE_ROOT', 'DPA_GRATA_ROOT')]
    permit = {'image': None, 'reverify_images': set(), 'opened_images': set()}; opened = set(); rejected = []
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        p = Path(os.fsdecode(args[0])).resolve()
        protected = any(p.is_relative_to(root/d) for d in ('scorer', 'scores', 'reports'))
        mode = args[1] if len(args) > 1 and isinstance(args[1], str) else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        readonly = not any(k in mode for k in ('w', 'a', '+', 'x')) and not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        reference_read = p in refs and readonly
        image_read = p == permit['image'] and readonly
        reverify = p in permit['reverify_images'] and readonly
        allowed = (p == checkpoint and readonly) or image_read or reverify or reference_read or p.is_relative_to(root) or any(p.is_relative_to(x) for x in libraries+code)
        asset = p.suffix.lower() in ('.png', '.jpg', '.jpeg', '.pt', '.pth', '.npz', '.npy', '.safetensors')
        if protected or (p.is_relative_to(mount) or asset) and not allowed:
            rejected.append(str(p)); raise PermissionError('current-image and label-free metadata boundary')
        if p.is_relative_to(mount):
            opened.add(str(p))
        if image_read:
            permit['opened_images'].add(p)
    sys.addaudithook(audit)
    for p in (root/'scorer/denial.json', Path(c['campaign_root'])/'public/M0_GATE.json'):
        try:
            p.read_bytes()
        except PermissionError:
            continue
        raise AssertionError('offline score access not rejected')
    return permit, opened, rejected


def worker(c):
    import torch
    from dpa_ctta.r85_loss_aggregation.method import Host
    from dpa_ctta.r79_mechanism_control.method import style
    from dpa_ctta.r19_model_only.method import equal
    from dpa_ctta.r19_model_only.runtime import weights, image
    from dpa_ctta.r7_target_screen.runner import TargetReader
    from dpa_ctta.r9_current_first.assets import gpu_policy
    root = Path(c['output_root']); stage = os.environ['STAGE']; slot = int(os.environ['SLOT'])
    dest = root/'private'/f'{stage}_{slot}'; dest.mkdir()
    state = dict(status='RUNNING', started=time.time(), physical_updates=0, completed_physical_updates=0,
                 physical_forwards=0, training_backwards=0, online_label_reads=0, source_image_reads=0)
    cap = 7 if stage == 'qualify' else 960
    def persist():
        save(dest/'RUN_STATE.json', state)
    def guard():
        if time.time() > c['deadline']-180:
            raise TimeoutError('shared original deadline')
    guard.extra = {'image_accesses': 0}
    def charge(kind):
        guard()
        key = {'attempt': 'physical_updates', 'complete': 'completed_physical_updates'}[kind]
        if kind == 'attempt' and state[key] >= cap:
            raise RuntimeError('stage physical update cap')
        state[key] += 1
    def meter(h):
        def forward(*_):
            guard()
            if state['physical_forwards'] >= cap*10:
                raise RuntimeError('stage forward cap')
            state['physical_forwards'] += 1
        def backward(g):
            state['training_backwards'] += 1
            return g
        h.handles.append(h.native.model.register_forward_pre_hook(forward))
        h.handles.append(h.native.params[0].register_hook(backward))
    timings = []
    def step(h, x, raw):
        guard(); tick = time.time(); pre = h.before(x); post, detail = h.keyed_step(x, raw)
        torch.cuda.synchronize(); timings.append(time.time()-tick); persist()
        return pre, post
    persist(); hosts = []
    try:
        torch.set_num_threads(2); torch.set_num_interop_threads(2); gpu_policy(c['gpus'][slot])
        torch.cuda.set_per_process_memory_fraction(8*1024**3/torch.cuda.get_device_properties(0).total_memory, 0)
        permit, opened, rejected = access_guard(c)
        checkpoint = weights(c); reader = TargetReader(c['target_root'], 256*1024**2, 'image')
        canonical = read(root/'private/screen_o0.json'); content = {r['image_sha256']: i for i, r in enumerate(canonical)}
        assert len(content) == 192
        if stage == 'qualify':
            a = Host(checkpoint, 'VV', SEEDS[0], 'qualification', charge=charge)
            b = Host(checkpoint, 'VV', SEEDS[0], 'qualification', charge=charge)
            hosts = [a, b]
            for h in hosts:
                assert sum(p.numel() for p in h.native.params) == 19136
                meter(h)
            xs = [image(reader, row, permit, guard) for row in canonical[:3]]
            for i in (0, 1):
                if i == 1:
                    before_a, before_b = a.snapshot(), b.snapshot()
                raw = style(xs[i], SEEDS[0], i)
                _, za = step(a, xs[i], raw); _, zb = step(b, xs[i], raw)
                assert torch.equal(za, zb) and equal(a.snapshot()['base'], b.snapshot()['base'])
            after = a.snapshot(); sig, _ = signal_for(before_a, a)
            # An independent double-precision CPU optimizer tests zero-new-gradient algebra.
            params = [torch.nn.Parameter(p.double().clone()) for p in before_a['base']['parameters'].values()]
            opt = torch.optim.Adam(params, lr=1e-4, betas=(.9, .999), eps=1e-8)
            opt.load_state_dict(copy.deepcopy(before_a['base']['adam']))
            initial = [p.detach().clone() for p in params]
            for p in params:
                p.grad = torch.zeros_like(p)
            opt.step()
            actual = torch.cat([(p.detach()-q).flatten() for p, q in zip(params, initial)])
            assert torch.allclose(actual, zero_delta(before_a['base']['adam']), atol=1e-12, rtol=1e-9)
            use = use_state(before_a, after); a.restore(use)
            restored = a.snapshot()['base']
            for k in ('parameters', 'adam', 'steps', 'grata'):
                assert equal(restored[k], before_a['base'][k])
            for k in ('visits', 'rng', 'native_rng'):
                assert equal(restored[k], after['base'][k])
            raw = style(xs[2], SEEDS[0], 2)
            step(a, xs[2], raw); _, write = step(b, xs[2], raw)
            assert b.native.steps == a.native.steps+1
            a.restore(after); _, replay = step(a, xs[2], raw)
            assert torch.equal(write, replay) and equal(a.snapshot()['base'], b.snapshot()['base'])
            assert state['physical_updates'] == state['completed_physical_updates'] == 7
            assert state['physical_forwards'] == 70 and state['training_backwards'] == 7
            save(root/'public/QUALIFICATION.json', dict(status='PASS', updates=7, forwards=70, training_backwards=7,
                 original_VV_and_WRITE_replay_exact=True, restore_learning_pre_RNG_and_visits_post=True,
                 CPU_zero_gradient_Adam_formula=True, CPU_clone_optimizer_steps=1, physical_GPU_clone_updates=0,
                 transaction_seconds=timings, max_steady_transaction_seconds=max(timings[1:]),
                 online_label_reads=0, source_image_reads=0, sign_bound_multiplier=64))
        else:
            assert read(root/'FORMAL_ADMISSION.json')['admitted']
            rows = read(root/'private'/f'screen_o{slot}.json')
            for seed in SEEDS:
                folder = root/'jobs'/f'main_s{seed}_o{slot}'; folder.mkdir()
                h = Host(checkpoint, 'VV', seed, f's{seed}_o{slot}', charge=charge); hosts = [h]; meter(h)
                signals = []
                with gzip.GzipFile(filename=str(folder/'predictions.bits.gz'), mode='xb', compresslevel=1, mtime=0) as output, Path(c['reference_predictions'][f'{seed}_{slot}']).open('rb') as reference:
                    for visit, row in enumerate(rows):
                        ordinal = visit+1; x = image(reader, row, permit, guard); raw = style(x, seed, content[row['image_sha256']])
                        before = h.snapshot() if ordinal in PROBES else None
                        pre, post = step(h, x, raw)
                        packed = np.packbits(np.stack([(z[0] >= 0).numpy() for z in (pre, post)])).tobytes()
                        assert reference.read(len(packed)) == packed, 'main PRE/K original reference exact'
                        output.write(packed)
                        if before is not None:
                            after = h.snapshot(); sig, g = signal_for(before, h)
                            record = dict(arrival=ordinal, **sig); signals.append(record)
                            torch.save(dict(parameters=before['base']['parameters'], adam=before['base']['adam'],
                                            steps=before['base']['steps'], gradient=g.float()), folder/f'inbound_{ordinal}.private.pt')
                            h.restore(use_state(before, after))
                            with gzip.GzipFile(filename=str(folder/f'probe_{ordinal}.use.bits.gz'), mode='xb', compresslevel=1, mtime=0) as branch:
                                for future in range(visit+1, visit+17):
                                    rr = rows[future]; xx = image(reader, rr, permit, guard)
                                    p, k = step(h, xx, style(xx, seed, content[rr['image_sha256']]))
                                    branch.write(np.packbits(np.stack([(z[0] >= 0).numpy() for z in (p, k)])).tobytes())
                            h.restore(after)
                            assert equal(h.snapshot()['base'], after['base']), 'USE contaminated WRITE state'
                        if ordinal % 8 == 0:
                            state.update(seed=seed, main_arrival=ordinal, root_bytes=usage(c)); persist()
                    assert not reference.read(1)
                save(folder/'signals.json', signals)
                assert len(signals) == 8 and h.native.steps == h.visits == 192
                h.check_frozen(True); h.close(); hosts = []
                save(folder/'COMPLETE.json', dict(arrivals=192, probes=8, main_updates=192, branch_updates=128,
                                                PRE_K_reference_exact=True, online_label_reads=0))
        for h in hosts:
            h.check_frozen(True); h.close()
        hosts = []
        assert state['physical_updates'] == state['completed_physical_updates'] == cap
        assert state['physical_forwards'] == cap*10 and state['training_backwards'] == cap
        permit['reverify_images'] = {Path(p).resolve() for p in reader.used}
        assert permit['reverify_images'] <= permit['opened_images']
        try:
            reader.after_check()
        finally:
            permit['reverify_images'].clear()
        save(dest/'ACCESS_AUDIT.json', dict(online_label_reads=0, source_image_reads=0,
             image_accesses=guard.extra['image_accesses'], opened_paths=sorted(opened), rejections=rejected))
        state.update(status='GPU_COMPLETE', ended=time.time()); persist()
    except BaseException as exc:
        state.update(status='FAILED', error=type(exc).__name__, reason=str(exc), ended=time.time()); persist()
        raise
    finally:
        for h in hosts:
            h.close()


def controller(c):
    root = Path(c['output_root']); attempts = []
    def status(stage, **extra):
        save(root/'RUN_STATE.json', dict(stage=stage, shared_started=c['started'], deadline=c['deadline'], **extra))
        save(root/'RESOURCE_LEDGER.json', dict(attempts=attempts,
             GPU_process_seconds=sum(a['gpu_process_seconds'] for a in attempts),
             CPU_process_seconds=sum(a['cpu_process_seconds'] for a in attempts)))
    def group(mode, stage, slots):
        pending, running, done = list(slots), [], []
        while pending or running:
            if time.time() > c['deadline']-180:
                raise TimeoutError('shared deadline group admission')
            if pending:
                try:
                    running.append(child(c, mode, stage, pending[0])); pending.pop(0)
                except RuntimeError as exc:
                    if 'GPU UUID/free-memory' not in str(exc):
                        raise
                    status('WAIT_RESOURCE', pending=pending)
            for item in running[:]:
                result = finish_child(c, item)
                if result is not None:
                    attempts.append(result); done.append(result); running.remove(item); status(stage)
            if pending or running:
                time.sleep(5 if running else 30)
        return all(a['exit_code'] == 0 for a in done)
    try:
        status('QUALIFY')
        if not group('worker', 'qualify', [0]):
            raise RuntimeError('qualification failed; preserve attempt')
        q = read(root/'public/QUALIFICATION.json')
        projection = q['max_steady_transaction_seconds']*1920*1.5
        admitted = q['status'] == 'PASS' and projection/2+2400 < c['deadline']-time.time()
        admitted = admitted and c.get('prior_updates', 0)+1927 <= c['max_updates'] and c.get('prior_forwards', 0)+19270 <= c['max_forwards']
        save(root/'FORMAL_ADMISSION.json', dict(admitted=admitted, projected_serial_GPU_seconds=projection,
                                               parallel_plus_score_headroom_seconds=projection/2+2400,
                                               shared_deadline=c['deadline']))
        if not admitted:
            raise RuntimeError('profile not admitted; do not shrink controls or alter protocol')
        status('FORMAL')
        if not group('worker', 'formal', [0, 1]):
            raise RuntimeError('formal engineering failure')
        status('SCORING')
        if not group('score', 'final', [0]):
            raise RuntimeError('CPU score failure')
        status('COMPLETE')
        save(root/'CONTROLLER_EXIT.json', dict(status='COMPLETE', ended=time.time()))
    except BaseException as exc:
        status('FAILED', error=type(exc).__name__, reason=str(exc))
        save(root/'CONTROLLER_EXIT.json', dict(status='FAILED', ended=time.time()))
        raise


if __name__ == '__main__':
    config = read(Path(os.environ['CONFIG']))
    mode = os.environ['MODE']
    if mode == 'watchdog':
        watchdog(config)
    elif mode == 'controller':
        controller(config)
    elif mode == 'worker':
        worker(config)
