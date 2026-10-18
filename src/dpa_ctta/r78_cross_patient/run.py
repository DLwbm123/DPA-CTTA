"""One finite source-development mechanism run, followed by a separate scorer."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import threading
import time
import traceback

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .objective import BNState, Field, ParameterPredictor, direct_logits, energy, project, reference, ribbon, update
from ..r51_same_state.run import read, save, run_child

SEED = 20261018
TRAIN = ('X', 'P', 'S', 'F', 'D')
ARMS = ('B0', 'C_FULL', 'C_A', 'H', 'P', 'S', 'X', 'F', 'D')
MODES = ('SELF', 'CROSS_MATCH', 'CROSS_MISMATCH')


def prepare_metadata(records, source_root):
    """Deterministic content split; final mask paths are returned separately."""
    import random
    rows = sorted(records, key=lambda r: r['sample_id'])
    if len({r['group_id'] for r in rows}) != len(rows) or len(rows) < 40:
        raise ValueError('at least 40 distinct registered content groups required')
    random.Random(SEED).shuffle(rows);rows = rows[:40]
    source_root = Path(source_root).resolve()
    def record(row, with_label):
        def confined(relative):
            path = (source_root/relative).resolve()
            if not path.is_relative_to(source_root) or path == source_root:
                raise ValueError('registered source path escapes data root')
            return str(path)
        value = dict(group_id=row['group_id'], image_path=confined(row['image_relative']), image_size=row['image_size'])
        if with_label:
            value['mask_path'] = confined(row['mask_relative'])
        return value
    panel = dict(train=[record(r, True) for r in rows[:16]], validation=[record(r, True) for r in rows[16:24]],
                 support=[record(r, False) for r in rows[24:32]], query=[record(r, False) for r in rows[32:40]])
    return panel, [record(r, True) for r in rows[32:40]]


def seed(value=SEED):
    import random
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def perturb(image, spec, noise_seed):
    kind, value = spec
    if kind == 'identity':
        result = image.clone()
    elif kind == 'gamma':
        result = image.pow(value)
    elif kind == 'rgb':
        result = image*image.new_tensor(value).view(1, 3, 1, 1)
    elif kind == 'noise':
        g = torch.Generator().manual_seed(noise_seed)
        result = image+value*torch.randn(image.shape, generator=g)
    elif kind == 'blur':
        axis = torch.arange(-3, 4, dtype=image.dtype)
        k = torch.exp(-axis.square()/(2*value*value));k /= k.sum()
        kernel = (k[:, None]*k[None, :]).expand(3, 1, 7, 7)
        result = F.conv2d(F.pad(image, (3,)*4, mode='reflect'), kernel, groups=3)
    else:
        raise ValueError('unknown appearance')
    return result.clamp(0, 1)


FINAL = (('identity', None), ('gamma', .7), ('rgb', (1.15, 1., .85)), ('blur', 1.))
VALIDATION = (('identity', None), ('gamma', .9), ('rgb', (1.08, 1., .92)), ('noise', .015))


def episodes():
    rng = np.random.default_rng(SEED)
    result = []
    for i in range(200):
        pair = rng.choice(16, 2, replace=False).tolist()
        kind = int(rng.integers(4))
        spec = [('identity', None), ('gamma', float(rng.uniform(.85, 1.15))),
                ('rgb', rng.uniform(.9, 1.1, 3).tolist()), ('noise', float(rng.uniform(0, .02)))][kind]
        result.append(dict(pair=pair, appearance=spec, noise_seeds=rng.integers(0, 2**31-1, 2).tolist()))
    return result


def supervision(logits, label):
    q = logits.sigmoid()
    dice = (2*(q*label).sum((-2, -1))+1)/(q.sum((-2, -1))+label.sum((-2, -1))+1)
    return F.binary_cross_entropy_with_logits(logits, label)+(1-dice).mean()


def dice(logits, label):
    pred = logits >= 0
    gt = label.bool()
    n = pred.sum((-2, -1))+gt.sum((-2, -1))
    return torch.where(n > 0, 2*(pred & gt).sum((-2, -1))/n.clamp_min(1), 1.).mean()


def adapt(host, head, arm, image, train=False):
    a = host.zero()
    if arm in ('B0', 'D'):
        return a, dict(empty=False, norm=0., projected=False)
    with torch.no_grad():
        ref = reference(image, host(image, a))
    if arm == 'F':
        state = head(ref.inputs, ref.empty)
        delta = None
    else:
        delta = ribbon(ref) if arm == 'H' else head(ref.inputs, geometry_only=arm == 'P')
        z = host(image, a)
        gradient, = torch.autograd.grad(energy(z, ref, delta), a, create_graph=train)
        raw = a-.01*gradient
        state = project(raw)
    detail = dict(empty=ref.empty, anchors=[len(x) for x in ref.anchors],
                  norm=float(state.detach().norm()), projected=bool(state.detach().norm() >= .049999))
    if delta is not None:
        detail.update(delta_mean=float(delta.detach().mean()), delta_positive=float((delta.detach() > 0).float().mean()),
                      delta_saturation=float((delta.detach().abs() > .99).float().mean()))
    return (state if train else state.detach()), detail


def direct(host, head, image):
    with torch.no_grad():
        z = host(image, host.zero())
        ref = reference(image, z)
    return direct_logits(z, ref, head(ref.inputs))


def train_step(host, head, arm, opt, spec, images, labels, device):
    opt.zero_grad(set_to_none=True)
    i, j = spec['pair']
    pixels = [perturb(images[k], spec['appearance'], s) for k, s in zip((i, j), spec['noise_seeds'])]
    xs = [normalize(x).to(device) for x in pixels]
    losses = []
    for support, query in ((0, 1), (1, 0)):
        if arm in ('S', 'D'):
            query = support
        target = labels[(i, j)[query]]
        if arm == 'D':
            logits = direct(host, head, xs[query])
        else:
            a, _ = adapt(host, head, arm, xs[support], train=True)
            logits = host(xs[query], a)
        loss = supervision(logits, target)/2
        if not torch.isfinite(loss):
            raise ValueError('nonfinite source objective')
        loss.backward()
        losses.append(float(loss.detach()))
    if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()):
        raise ValueError('missing/nonfinite outer gradient')
    norm = sum(float(p.grad.square().sum()) for p in head.parameters())**.5
    opt.step()
    return dict(loss=sum(losses), gradient_norm=norm)


def normalize(pixels):
    from ..hosts.vptta import model_input_from_pixels
    return model_input_from_pixels(pixels, 'fundus')


def validation(host, head, arm, images, labels, device):
    values = []
    for pair in range(4):
        for condition, spec in enumerate(VALIDATION):
            x = normalize(perturb(images[pair], spec, SEED+1000+pair)).to(device)
            q = normalize(perturb(images[pair+4], spec, SEED+2000+pair)).to(device)
            if arm == 'D':
                with torch.no_grad():
                    z = direct(host, head, q)
            else:
                a, _ = adapt(host, head, arm, x)
                with torch.no_grad():
                    z = host(q, a)
            values.append(float(dice(z, labels[pair+4])))
    return float(np.mean(values))


def access_guard(c, labels):
    root = Path(c['output_root']).resolve()
    data = Path(c['source_root']).resolve()
    allowed = {Path(p).resolve() for p in labels}
    opened = set()
    def audit(event, args):
        if event != 'open' or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(root/'scorer'):
            raise PermissionError('final labels are scorer-only')
        if path.is_relative_to(data):
            if 'mask' in path.parts and path not in allowed:
                raise PermissionError('source final or unregistered label read rejected')
            opened.add(str(path))
    sys.addaudithook(audit)
    return opened


def matched(host, pixels, native_rng):
    from ..b1_host import official
    from ..r20_model_only_search.method import weights_for
    from ..r36_incremental_modules.method import local_structure_loss
    from ..host_diagnostic import restore as restore_native
    from ..r8_ba.rng import capture, restore
    old = capture();restore_native(native_rng)
    a = host.zero();weak = []
    class Model(nn.Module):
        def forward(self, x):
            z = host(x, a)
            if not torch.is_grad_enabled():
                weak.append(z.detach().cpu())
            return z, None
    try:
        opt = torch.optim.SGD([a], lr=.01)
        grata = official().GraTa([a], opt, Model(), device=str(a.device))
        def criterion(z, q):
            if len(weak) != 6:
                raise ValueError('six aligned weak predictions required')
            w = weights_for(torch.stack(weak).sigmoid(), .1, 2.).to(z)
            orientation, _ = local_structure_loss(z, q, w)
            return (F.binary_cross_entropy_with_logits(z, q, reduction='none')*w).mean()+.1*orientation
        grata.cal_consis_loss({'data': normalize(pixels).numpy().copy()}, criterion=criterion)
        if a.grad is None or not torch.isfinite(a.grad).all():
            raise ValueError('matched BN gradient missing')
        result = project(a-.01*a.grad).detach()
        return result, dict(norm=float(result.norm()), projected=bool(result.norm() >= .049999))
    finally:
        restore(old)


def worker(c):
    from ..integrations.ctta_suite import build_reference_model
    from ..source_io import read_pixels, read_mask
    from ..r37_cw_orientation.method import Host as Native, candidates
    root = Path(c['output_root']);started = time.time()
    status = dict(status='QUALIFYING', started=started, training_episodes=0, predictions=0)
    save(root/'RUN_STATE.json', status)
    device = torch.device('cuda:0')
    def guard():
        if time.time()-started > c['gpu_cap_seconds']-20:
            raise TimeoutError('fixed total worker budget exceeded')
    native = None
    try:
        if c['grouping'] != 'CONTENT_DEVELOPMENT_USER_AUTHORIZED' or c['rdc'] or c['engram']:
            raise ValueError('only authorized content-development scope without modules')
        torch.set_num_threads(2);torch.set_num_interop_threads(2);seed()
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        panel = read(root/'private/panel.json')
        if [len(panel[k]) for k in ('train', 'validation', 'support', 'query')] != [16, 8, 8, 8]:
            raise ValueError('fixed content coverage')
        all_rows = sum(panel.values(), [])
        if len({r['group_id'] for r in all_rows}) != 40:
            raise ValueError('content overlap')
        train_rows = panel['train']+panel['validation']
        opened = access_guard(c, [r['mask_path'] for r in train_rows])
        try:
            (root/'scorer/panel.json').read_bytes()
        except PermissionError:
            pass
        else:
            raise AssertionError('final label guard failed')
        state = torch.load(c['checkpoint_path'], map_location='cpu', weights_only=True)
        model, _ = build_reference_model('fundus');model.load_state_dict(state);model.to(device)
        host = BNState(model)
        if host.dimensions != 512:
            raise ValueError('512-dimensional state required')
        params_before = {n: p._version for n, p in model.named_parameters()}
        model_counts = dict(forwards=0)
        handle = model.register_forward_hook(lambda *args: model_counts.__setitem__('forwards', model_counts['forwards']+1))
        images = {role: [read_pixels(r['image_path'], 'fundus', r['image_size']) for r in rows]
                  for role, rows in panel.items()}
        labels = {role: [read_mask(r['mask_path'], 'fundus', r['image_size']).to(device) for r in panel[role]]
                  for role in ('train', 'validation')}
        heads, opts = {}, {}
        for arm in TRAIN:
            seed();heads[arm] = (ParameterPredictor(512) if arm == 'F' else Field()).to(device)
            opts[arm] = torch.optim.Adam(heads[arm].parameters(), lr=.001, betas=(.9, .999), eps=1e-8, weight_decay=0)
        schedule = episodes();save(root/'private/episodes.json', schedule)
        times = {};history = {arm: [] for arm in TRAIN};best = {arm: (-float('inf'), None, None) for arm in TRAIN}
        # These first two episodes per arm count toward the fixed 200, including their labels.
        for arm in TRAIN:
            tick = time.time()
            for i in range(2):
                guard();history[arm].append(train_step(host, heads[arm], arm, opts[arm], schedule[i], images['train'], labels['train'], device))
                status['training_episodes'] += 1
            torch.cuda.synchronize();times[arm] = (time.time()-tick)/2
        projected = time.time()-started+sum(198*t for t in times.values())*2+600
        if projected > c['gpu_cap_seconds']-30:
            raise TimeoutError('full fixed matrix not admitted by measured training time')
        save(root/'QUALIFICATION.json', dict(status='PASS', adaptation_dimensions=host.dimensions,
             first_two_episode_seconds=times, conservative_projected_worker_seconds=projected,
             label_budget_included=True, final_labels_read=0,
             peak_allocated_bytes=torch.cuda.max_memory_allocated(), current_image_bn=True,
             head_parameters={k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}))
        status['status'] = 'TRAINING';save(root/'RUN_STATE.json', status)
        for arm in TRAIN:
            for i in range(2, 200):
                guard();history[arm].append(train_step(host, heads[arm], arm, opts[arm], schedule[i], images['train'], labels['train'], device))
                status['training_episodes'] += 1
                if (i+1) % 50 == 0:
                    value = validation(host, heads[arm], arm, images['validation'], labels['validation'], device)
                    if value > best[arm][0]:
                        best[arm] = (value, i+1, copy.deepcopy(heads[arm].state_dict()))
                    save(root/'private'/f'training_{arm}.json', dict(steps=history[arm], validation_episode=i+1,
                         validation_dice=value, selected_episode=best[arm][1], seconds=time.time()-started))
                    status.update(current_arm=arm, current_episode=i+1);save(root/'RUN_STATE.json', status)
            heads[arm].load_state_dict(best[arm][2])
            torch.save(dict(state_dict=best[arm][2], selected_episode=best[arm][1]), root/'private'/f'head_{arm}.pt')
        save(root/'TRAINING_SUMMARY.json', dict(arms={k: dict(episodes=200, supervised_query_label_visits=400,
             selected_episode=best[k][1], validation_dice=best[k][0], first_loss=history[k][0]['loss'],
             last_loss=history[k][-1]['loss'], first_gradient_norm=history[k][0]['gradient_norm'],
             last_gradient_norm=history[k][-1]['gradient_norm']) for k in TRAIN},
             training_episodes=status['training_episodes']))
        # Full CW_LSO is the existing C host (six weak views, strong view, Adam, all BN).
        # It does not use the G host's entropy perturbation or cosine learning rate.
        native = Native(state, candidates()[3], SEED, 'source_episode_reset', device='cuda:0')
        initial = native.snapshot();native_rng = copy.deepcopy(native.native.rng)
        counters = dict(native_forwards=0, native_backwards=0)
        records = [];normalization_changes = []
        status['status'] = 'GENERATING';save(root/'RUN_STATE.json', status)
        for pair in range(8):
            support = images['support'][pair];query = images['query'][pair]
            for condition, spec in enumerate(FINAL):
                query_pixels = perturb(query, spec, SEED+pair)
                q = normalize(query_pixels).to(device)
                normalization_changes.append(float((normalize(query_pixels)-normalize(query)).abs().mean()))
                for arm in ARMS:
                    for mode in (('DIRECT',) if arm in ('B0', 'D') else MODES):
                        guard();tick = time.time();detail = {}
                        if arm == 'B0':
                            with torch.no_grad():
                                z = host(q, host.zero())
                        elif arm == 'D':
                            with torch.no_grad():
                                z = direct(host, heads[arm], q)
                        else:
                            k = (condition+1) % 4 if mode == 'CROSS_MISMATCH' else condition
                            s_pixels = query_pixels if mode == 'SELF' else perturb(support, FINAL[k], SEED+pair+100)
                            s = normalize(s_pixels).to(device)
                            if arm == 'C_FULL':
                                native.restore(initial)
                                _, d = native.step(s_pixels)
                                counters['native_forwards'] += d['native'][0]['counts']['forwards']+1
                                counters['native_backwards'] += d['native'][0]['counts']['backwards']
                                z = native._readonly(native.native.model, query_pixels)
                                detail = d['diagnostics']
                            else:
                                if arm == 'C_A':
                                    a, detail = matched(host, s_pixels, native_rng)
                                else:
                                    a, detail = adapt(host, heads.get(arm), arm, s)
                                with torch.no_grad():
                                    z = host(q, a)
                        pred = z.detach().cpu().numpy()[0] >= 0
                        index = len(records)
                        with (root/'predictions'/f'{index:03d}.bits').open('xb') as f:
                            f.write(np.packbits(pred).tobytes())
                        records.append(dict(index=index, pair=pair, condition=condition, arm=arm, mode=mode,
                                            seconds=time.time()-tick, diagnostics=detail))
                        status['predictions'] = len(records)
            save(root/'RUN_STATE.json', status)
        if len(records) != 736 or status['training_episodes'] != 1000:
            raise ValueError('fixed matrix incomplete')
        if any(p._version != params_before[n] or p.grad is not None for n, p in model.named_parameters()):
            raise ValueError('frozen base parameters changed')
        if not all(normalization_changes[i] > 0 for i in range(32) if i % 4):
            raise ValueError('nonidentity appearance erased by preprocessing')
        save(root/'private/predictions.json', records)
        save(root/'ACCESS_AUDIT.json', dict(source_image_files=len([p for p in opened if '/image/' in p]),
             source_train_validation_label_files=len([p for p in opened if '/mask/' in p]), final_labels_read=0,
             target_image_reads=0, target_label_reads=0, source_backbone_unchanged=True, patient_grouping='UNKNOWN',
             source_checkpoint_training_membership='UNKNOWN', normalized_input_change_mean=normalization_changes))
        save(root/'COUNTS.json', dict(**model_counts, **counters, deployment_readouts=736,
             training_query_label_visits=2000, training_outer_backward_calls=2000, training_inner_autograd_grad_calls=1200, selected_head_parameters={k: sum(p.numel() for p in h.parameters()) for k, h in heads.items()}))
        handle.remove()
        used = sum(p.stat().st_size for folder in ('private', 'predictions', 'logs') for p in (root/folder).rglob('*') if p.is_file())
        if used > c['output_cap_bytes']:
            raise ValueError('output budget exceeded')
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
        if attempts[-1]['status'] == 'COMPLETE' and read(root/'RUN_STATE.json')['status'] == 'GPU_COMPLETE':
            attempts.append(run_child(c, 'score', c['cpu_cap_seconds'], False))
    finally:
        save(root/'RESOURCE_LEDGER.json', dict(attempts=attempts, all_workers_retired=True,
             gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in attempts),
             cpu_worker_seconds=sum(a['cpu_worker_seconds'] for a in attempts),
             no_retry=True, gpu_cap_seconds=c['gpu_cap_seconds'], cpu_cap_seconds=c['cpu_cap_seconds']))
    return 0 if len(attempts) == 2 and all(a['status'] == 'COMPLETE' for a in attempts) else 1


def self_check(c):
    """Exercise both control hosts with generated pixels and random source weights."""
    import unittest
    from ..integrations.ctta_suite import build_reference_model
    from ..r37_cw_orientation.method import Host as Native, candidates
    started = time.time();root = Path(c['output_root'])
    torch.set_num_threads(2);seed()
    sys.path.insert(0, str(root/'code/tests'))
    tests = unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromName('test_r78_cross_patient'))
    if not tests.wasSuccessful():
        raise ValueError('synthetic tests failed')
    model, _ = build_reference_model('fundus')
    state = copy.deepcopy(model.state_dict())
    host = BNState(model)
    native = Native(state, candidates()[3], SEED, 'synthetic_qualification', device='cpu')
    pixels = torch.rand(1, 3, 512, 512)
    try:
        a, detail = matched(host, pixels, native.native.rng)
        native_before = native.snapshot()
        z, d = native.step(pixels)
        native.restore(native_before)
        again, _ = native.step(pixels)
        if not torch.equal(z, again) or not torch.isfinite(a).all() or a.norm() > .050001:
            raise ValueError('control parity/repeat qualification failed')
        if d['native'][0]['counts']['forwards'] != 8:
            raise ValueError('full CW_LSO physical count')
        x = normalize(pixels)
        with torch.no_grad():
            z0 = host(x, host.zero())
            native.restore(native_before)
            b0 = native._readonly(native.native.model, pixels)
        if not torch.equal(z0, b0):
            raise ValueError('functional core base differs from original host')
    finally:
        native.close()
    if torch.cuda.is_initialized():
        raise ValueError('CPU qualification initialized CUDA')
    save(root/'CPU_CHECK.json', dict(status='PASS', tests=tests.testsRun,
         complete_host_forwards=d['native'][0]['counts']['forwards'], matched_state_norm=detail['norm'],
         functional_base_exact=True, native_repeat_exact=True, real_images=0, real_labels=0,
         checkpoint_reads=0, gpu_seconds=0, cpu_seconds=time.time()-started))


def main():
    c = read(os.environ['DIAG_CONFIG']);mode = os.environ['DIAG_MODE']
    if mode == 'worker':
        worker(c);return 0
    if mode == 'controller':
        return controller(c)
    if mode == 'score':
        from .score import score
        score(c);return 0
    if mode == 'self_check':
        self_check(c);return 0
    raise ValueError('unknown mode')


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        raise
