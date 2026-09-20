"""Finite screen only. No source preparation, retries, resume or follow-on scopes.

Authority files are externally reviewed inputs, not signatures or self-issued
permissions. CUDA additionally requires an exact eight-arm qualification binding.
"""
import copy
import io
import json
import os
import platform
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from ..r7_shared.context import SCIENCE, json_digest, validate_context
from ..r7_shared.io import checked_path
from ..r7_shared.plan import matrix, dry_run
from ..r7_shared.host import OnlineHost
from ..r7_shared.report import review as report
from ..r7_source_prep.registry import verified, ordinary, digest, sha
from ..r7_source_prep.runner import (code_identity, load_model, load_artifact,
    BudgetOutput as _BudgetOutput, resource_check as _resource_check, cleanup_owned, TERMINAL_RESERVE, configure_backend)
from ..r7_shared.numerics import COUNTS
from ..r3.plan import stream

ROOT = Path(__file__).resolve().parents[3]
SCOPE = 'TARGET_SCREEN'
ARMS = [j['arm'] for j in matrix(SCOPE)[:8]]
METHODS = set(ARMS) - {'C_BASE', 'C0'}
SEED = 20260907


class BudgetOutput(_BudgetOutput):
    """Retain reviewed byte/evidence limits, additionally pin the owned directory."""
    def check_root(self):
        path = checked_path(self.path)
        st = path.stat(); identity = (st.st_dev, st.st_ino)
        if getattr(self, '_directory_identity', identity) != identity:
            raise ValueError('owned output directory replaced')
        self._directory_identity = identity

    def bytes(self, name, data):
        self.check_root()
        return super().bytes(name, data)

    def evidence(self, name, value):
        self.check_root()
        return super().evidence(name, value)


def resource_check(out, caps, started, reserve=TERMINAL_RESERVE):
    out.check_root()
    return _resource_check(out, caps, started, reserve)


def schedule(workers):
    if type(workers) is not int or workers not in (1, 3):
        raise ValueError('review either one serial worker or three order workers')
    jobs = matrix(SCOPE)
    return [jobs] if workers == 1 else [[j for j in jobs if j['order'] == o] for o in (0, 1, 4)]


def runtime():
    return dict(python=platform.python_version(), torch=torch.__version__,
                system=platform.system(), machine=platform.machine(), hostname=platform.node(), cpu_count=os.cpu_count(), threads=2)


def device_policy(config):
    device = config.get('device')
    ids = config.get('physical_GPU_ids')
    if device == 'cpu':
        if ids is not None: raise PermissionError('CPU route cannot bind GPU IDs')
        dtype = 'R7_CPU_FP32_MODEL_FP64_LATENT_V1'
    elif device == 'cuda:0':
        if config.get('baseline_device') != 'cpu':
            raise PermissionError('historical C requires qualified CPU baseline route')
        if (not isinstance(ids, list) or len(ids) != config.get('workers')
                or any(type(i) is not int or i < 0 for i in ids) or len(set(ids)) != len(ids)):
            raise PermissionError('one exact physical GPU per isolated worker')
        if not config.get('qualification'): raise PermissionError('target GPU qualification required')
        dtype = 'R7_CUDA_FP32_BACKBONE_CPU_METHOD_V1'
    else: raise PermissionError('unsupported device')
    if config.get('runtime') != runtime() or config.get('dtype_policy') != dtype:
        raise ValueError('exact runtime/dtype binding')
    schedule(config.get('workers'))
    if config.get('threads') != 2 or config.get('retry') is not False or config.get('resume') is not False:
        raise ValueError('two threads, no retry/resume')
    for k in ('wall_seconds', 'output_bytes', 'job_wall_seconds', 'job_output_bytes', 'max_asset_bytes', 'max_image_pixels'):
        if type(config.get(k)) is not int or config[k] <= 0:
            raise ValueError('finite positive integer resource caps')
    if config['job_output_bytes'] <= 2 * TERMINAL_RESERVE:
        raise ValueError('terminal evidence reserve')
    if config['output_bytes'] < 24 * config['job_output_bytes'] + 2 * TERMINAL_RESERVE:
        raise ValueError('aggregate output cap must cover all job caps and report reserve')


def confined(root, path):
    root = checked_path(root)
    p = checked_path(path)
    if p == root or not p.is_relative_to(root):
        raise ValueError('target path outside approved root')
    ordinary(p)
    return p


def disjoint(a, b):
    a, b = Path(a), Path(b)
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('input/output/code roots overlap')


def preflight(receipt, *, owned_output=None):
    # Fail before code probes, asset paths, output creation or device discovery.
    if receipt.get('schema') != 'R7_TARGET_SCREEN_AUTH_V1' or receipt.get('scope') != SCOPE or receipt.get('enabled') is not True:
        raise PermissionError('TARGET_SCREEN disabled by default')
    auth = receipt.get('user_authorization', {})
    if auth.get('granted') is not True or auth.get('scope') != SCOPE or not auth.get('receipt_id'):
        raise PermissionError('new exact-scope authorization required')
    binding = receipt['binding']
    external = receipt.get('execution_layer_review', {})
    waived = (external.get('status') == 'USER_WAIVED'
              and auth.get('external_review_waiver') == dict(scope=SCOPE, binding_sha256=json_digest(binding), explicit=True))
    if ((external.get('status') != 'PASS' and not waived) or external.get('scope') != SCOPE
            or not external.get('reference') or external.get('binding_sha256') != json_digest(binding)):
        raise PermissionError('external execution review binding')
    if binding['code_sha'] != code_identity() or binding['science_sha256'] != SCIENCE:
        raise ValueError('exact clean code/science binding')
    for name, expected in SCIENCE.items():
        verified(ROOT/'docs/review/r7/input/specs'/name, expected, 1024**2)
    if binding['jobs'] != matrix(SCOPE) or binding['seed'] != SEED:
        raise ValueError('exact frozen matrix/seed')
    cfg = binding['resources']; device_policy(cfg)
    if cfg.get('resource_authorization') != {'scope': SCOPE, 'receipt_id': auth['receipt_id']}:
        raise PermissionError('resource authorization binding')
    docs = {}
    for key in ('registration', 'inventory'):
        row = binding[key]
        docs[key] = json.loads(verified(row['path'], row['sha256'], 32*1024**2))
    meta = dry_run(docs['registration'])
    if binding['registration_digest'] != meta['recurrence']['registration_digest'] or binding['recurrence_digest'] != meta['recurrence']['stream_digest']:
        raise ValueError('registration/recurrence mismatch')
    source = binding['source_release']
    source_accepted = source.get('status') == 'REVIEWED' or (waived and source.get('status') == 'USER_ACCEPTED_VERIFIED_ARTIFACTS')
    if (not source_accepted or source.get('artifact_identity') != binding['inventory']['sha256']
            or not source.get('external_review_reference') or source.get('trusted_loader_verified') is not True):
        raise PermissionError('source artifacts PENDING or not independently reviewed')
    sha(source['manifest_sha256']); sha(source['split_sha256'])
    if set(docs['inventory']) != METHODS:
        raise ValueError('exact six released artifacts')
    cp = binding['checkpoint']; registered = docs['registration']['checkpoint']
    if cp['sha256'] != registered['sha256'] or cp['bytes'] != registered['bytes']:
        raise ValueError('registered original checkpoint binding')
    root = checked_path(binding['target_root']); artifact_root = checked_path(binding['artifact_root'])
    storage = checked_path(cfg['storage_root']); out = checked_path(binding['output_dir'])
    if owned_output is not None:
        record = json.loads((out/'owner.json').read_text())
        if record != dict(owner=owned_output, binding=dict(binding_sha256=json_digest(binding))):
            raise ValueError('existing output ownership binding')
    if not all(p.is_dir() for p in (root, artifact_root, storage)) or (out.exists() and owned_output is None) or not out.is_relative_to(storage):
        raise ValueError('existing input/storage roots and fresh approved output required')
    protected = [root, artifact_root, checked_path(cp['path']), checked_path(ROOT)]
    protected += [checked_path(binding[k]['path']) for k in ('registration', 'inventory')]
    for p in protected:
        disjoint(out, p)
    disjoint(root, artifact_root); disjoint(root, checked_path(cp['path']))
    # Metadata/stat only; do not open target bytes during preflight.
    for row in docs['registration']['target']:
        if row['domain'] not in ('REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS'):
            raise ValueError('source RGB/mask domain forbidden')
        if np.prod(row['image_size']) > cfg['max_image_pixels']:
            raise ValueError('decoded geometry cap')
        for kind in ('image', 'mask'):
            confined(root, row[kind+'_path'])
    contexts = {}
    for name, row in docs['inventory'].items():
        if set(row) != {'file','training_asset_file_sha256','bytes','context_sha256','context_file_sha256'}:
            raise ValueError('exact source inventory row schema')
        if row['file'] != name+'.pt':
            raise ValueError('artifact filename binding')
        raw = verified(artifact_root/row['file'], row['training_asset_file_sha256'], cfg['max_asset_bytes'])
        if len(raw) != row['bytes']: raise ValueError('artifact length binding')
        context = json.loads(verified(artifact_root/(name+'.context.json'), row['context_file_sha256'], 1024**2))
        validate_context(context)
        src = context['payload']['source']; method = context['payload']['method']
        if (context['sha256'] != row['context_sha256'] or src['source_binding_status'] != 'BOUND'
                or src['checkpoint_file_sha256'] != cp['sha256'] or src['source_manifest_sha256'] != source['manifest_sha256']
                or src['source_split_sha256'] != source['split_sha256'] or method['group'] != name[0]
                or method['mode'] != name.split('_')[1] or context['payload']['ablation'] is not None):
            raise ValueError('trusted artifact context/provenance binding')
        contexts[name] = context
    backends = [c['payload']['environment'].get('execution_backend') for c in contexts.values()]
    expected_backend = cfg.get('execution_backend')
    if cfg['device'] == 'cpu':
        if any(v is not None for v in backends): raise ValueError('GPU context cannot use CPU resources')
    else:
        if not expected_backend or any(v != expected_backend for v in backends):
            raise ValueError('exact source GPU backend binding')
        q = cfg['qualification']
        qualification = json.loads(verified(q['path'], q['sha256'], 1024**2))
        if (qualification.get('status') != 'PASSED' or qualification.get('code_sha') != binding['code_sha']
                or qualification.get('arms') != ARMS or qualification.get('execution_backend') != expected_backend
                or qualification.get('physical_GPU_ids') != cfg['physical_GPU_ids']
                or qualification.get('arm_devices') != {arm: ('cpu' if arm == 'C_BASE' else 'cuda:0') for arm in ARMS}):
            raise ValueError('eight-arm GPU qualification/code/device binding')
    # Fresh tensor loading for the WHOLE matrix precedes every target pixel read.

    return dict(receipt=copy.deepcopy(receipt), binding=copy.deepcopy(binding), config=cfg,
                registration=docs['registration'], inventory=docs['inventory'], contexts=contexts)


def file_identity(path):
    _, st = ordinary(path)
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, st.st_nlink)


class TargetReader:
    """Image and label capabilities are distinct; online receives image records only."""
    def __init__(self, root, limit, kind):
        if kind not in ('image', 'mask'): raise ValueError('read capability')
        self.root, self.limit, self.kind = root, limit, kind
        self.used = {}; self.counts = Counter()

    def read(self, row):
        from ..source_io import read_pixels, read_mask
        kind = self.kind
        p = confined(self.root, row[kind+'_path'])
        # Register attempted input BEFORE decode, so decode failures still get after-read audit.
        identity = file_identity(p)
        prior = self.used.setdefault(str(p), (row[kind+'_sha256'], identity))
        if prior != (row[kind+'_sha256'], identity): raise ValueError('target identity changed')
        raw = verified(p, row[kind+'_sha256'], self.limit, self.counts)
        fn = read_pixels if kind == 'image' else read_mask
        return fn(io.BytesIO(raw), 'fundus', row['image_size'])

    def after_check(self):
        for path, (expected, identity) in self.used.items():
            confined(self.root, path)
            if file_identity(path) != identity: raise ValueError('target changed after read')
            verified(path, expected, self.limit, self.counts)


def image_records(rows):
    return [{k: r[k] for k in ('image_path', 'image_sha256', 'image_size')} for r in rows]


def physical(trace, arm):
    c = trace['counts']
    value = dict(forwards=c.get('backbone_forwards', c.get('forwards', 0)),
                 backwards=c.get('backwards', 0), Adam=c.get('base_adam', 0))
    expected = dict(forwards=8 if arm == 'C_BASE' else 1 if arm == 'C0' else 2,
                    backwards=int(arm == 'C_BASE'), Adam=int(arm == 'C_BASE'))
    if value != expected: raise ValueError('per-visit frozen physical count mismatch')
    return value


def cost_state():
    return dict(observed={}, physical=None, committed_visits=0, prediction_files=0,
                phase='initialization', completeness='NOT_STARTED', unobserved_tail='unknown')


def capture_cost(host, arm, before, cost):
    observed = dict(host.counts) if arm == 'C_BASE' else dict(COUNTS - before)
    cost['observed'] = observed
    cost['physical'] = dict(forwards=observed.get('forwards', observed.get('backbone_forwards', 0)),
                            backwards=observed.get('backwards', 0), Adam=observed.get('base_adam', 0))
    cost['committed_visits'] = host.steps if arm == 'C_BASE' else host.visits


def online(host, rows, reader, arm, out, check, cost=None):
    """No masks, evaluator, domain, subset or scores in this call graph.

    Persist the exact >=0.5 decisions consumed by the existing evaluator. All
    1951 visits are retained privately; report filters remaining_dev unchanged.
    """
    cost = cost_state() if cost is None else cost
    before = COUNTS.copy(); counts = Counter()
    capture_cost(host, arm, before, cost)
    out.write('cost_initial.json', cost)
    for index, row in enumerate(rows):
        first = None
        try:
            cost.update(phase='image_read', completeness='EXACT_OBSERVED', unobserved_tail=None)
            check(); pixels = reader.read(row)
            cost.update(phase='host_step', completeness='LOWER_BOUND', unobserved_tail='failed call may have unobserved work')
            logits, trace = host.step(pixels)
            cost.update(phase='prediction', completeness='EXACT_OBSERVED', unobserved_tail=None)
            if logits.requires_grad or tuple(logits.shape) != (1, 2, 512, 512) or not torch.isfinite(logits).all():
                raise ValueError('uncommitted or nonfinite prediction')
            counts.update(physical(trace, arm))
            bits = np.packbits((logits.detach().cpu().sigmoid() >= .5).numpy().reshape(-1)).tobytes()
            out.bytes(f'prediction_{index:04d}.bits', bits)
            cost['prediction_files'] += 1
            check()
        except BaseException as exc: first = exc
        finally:
            capture_cost(host, arm, before, cost)
            try: out.write(f'cost_{index:04d}.json', cost)
            except BaseException as exc:
                cost.setdefault('evidence_errors', []).append(error(exc))
                first = first or exc
        if first: raise first
    cost.update(phase='online_complete', completeness='EXACT_OBSERVED', unobserved_tail=None)
    if cost['physical'] != dict(counts): raise ValueError('observed versus trace count mismatch')
    return dict(counts)


def posthoc(rows, reader, arm, order, out, check):
    from ..p1_analysis import evaluate
    result = []
    for index, row in enumerate(rows):
        check()
        raw = (out.path/f'prediction_{index:04d}.bits').read_bytes()
        if len(raw) != 65536: raise ValueError('prediction dimensions')
        probability = torch.from_numpy(np.unpackbits(np.frombuffer(raw, dtype=np.uint8)).copy()).float().reshape(1, 2, 512, 512)
        metrics = evaluate(probability, reader.read(row), 'fundus')
        result.append(dict(arm=arm, order=order, content=row['group_id'], domain=row['domain'],
                           subset=row['subset'], dice=[m['dice'] for m in metrics], assd=[m['assd'] for m in metrics]))
    out.write('scalars.private.json', result)
    return result


def make_host(approved, arm):
    from ..source_pilot import seed_all
    from ..b1_host import Host
    binding = approved['binding']; cp = binding['checkpoint']
    raw = verified(cp['path'], cp['sha256'], approved['config']['max_asset_bytes'])
    if len(raw) != cp['bytes']: raise ValueError('checkpoint length')
    seed_all(SEED)
    if arm == 'C_BASE':
        state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
        return Host('C', state, 'cpu'), state
    segmenter = load_model(raw) if approved['config']['device'] == 'cpu' else load_model(raw, device='cuda:0')
    if arm == 'C0':
        # Independent trusted environment from source release, no candidate-minted identity.
        context = copy.deepcopy(approved['contexts']['A_FULL'])
        context['payload']['method'] = dict(schema='R7_METHOD_DIGEST_V1', group='C0', mode='ZERO', weights_sha256=None)
        context['sha256'] = json_digest(context['payload'])
        return OnlineHost(segmenter, expected_context=context), None
    return load_artifact(segmenter, arm, binding['artifact_root'], approved['inventory'], approved['config']['max_asset_bytes']), None


def error(exc): return dict(type=type(exc).__name__, message=str(exc)[:3000])


def execute_job(approved, job, out):
    cfg = approved['config']; b = approved['binding']; started = time.monotonic()
    caps = dict(wall_seconds=cfg['job_wall_seconds'], output_bytes=cfg['job_output_bytes'])
    check = lambda: resource_check(out, caps, started)
    images = TargetReader(b['target_root'], cfg['max_asset_bytes'], 'image')
    masks = TargetReader(b['target_root'], cfg['max_asset_bytes'], 'mask')
    first = None; secondary = []; host = None; cost = cost_state()
    try:
        rows = stream(approved['registration'], job['order'])
        if len(rows) != job['arrivals'] or sum(r['subset']=='remaining_dev' for r in rows) != job['scored_contents']:
            raise ValueError('job stream coverage')
        host, state = make_host(approved, job['arm'])
        counts = online(host, image_records(rows), images, job['arm'], out, check, cost)
        expected = dict(forwards=job['network_forwards'], backwards=job['backwards'], Adam=job['Adam'])
        if counts != expected: raise ValueError('terminal physical counts')
        cost['phase'] = 'counts_persistence'
        out.write('counts.json', counts)
        cost['phase'] = 'model_close'
        if job['arm'] == 'C_BASE': host.finish(state)
        else: host._check_frozen(boundary=True); host.segmenter.close()
        # Drop all model/optimizer/state references before the first target mask read.
        del host, state; host = None
        cost['phase'] = 'posthoc'
        posthoc(rows, masks, job['arm'], job['order'], out, check)
        cost['phase'] = 'completed'
    except BaseException as exc:
        first = exc
    finally:
        try:
            if host is not None:
                if hasattr(host, 'segmenter'): host.segmenter.close()
                else:
                    for h in host.handles: h.remove()
        except BaseException as exc: secondary.append(error(exc)); first = first or exc
        for reader in (images, masks):
            try: reader.after_check()
            except BaseException as exc: secondary.append(error(exc)); first = first or exc
        # After-read metadata and checkpoint/artifacts remain immutable too.
        try:
            for key in ('registration', 'inventory', 'checkpoint'):
                entry = b[key]; verified(entry['path'], entry['sha256'], cfg['max_asset_bytes'])
            for name, row in approved['inventory'].items():
                verified(Path(b['artifact_root'])/row['file'], row['training_asset_file_sha256'], cfg['max_asset_bytes'])
                verified(Path(b['artifact_root'])/(name+'.context.json'), row['context_file_sha256'], 1024**2)
            check()
        except BaseException as exc: secondary.append(error(exc)); first = first or exc
        records = [('worker.json', dict(status='FAILED' if first else 'JOB_PENDING_TERMINAL_AUDIT',
                   target_after_check='FAILED' if secondary else 'UNCHANGED', secondary=secondary,
                   model_cost=copy.deepcopy(cost), image_IO=dict(images.counts), mask_IO=dict(masks.counts), wall_seconds=time.monotonic()-started, retry=False))]
        if first: records.insert(0, ('first_error.json', error(first)))
        for name, value in records:
            try: out.evidence(name, value)
            except BaseException as exc:
                first = first or exc
                try: print(json.dumps(dict(evidence_error=error(exc), first_error=error(first))), file=sys.stderr, flush=True)
                except BaseException: pass
    if first: raise first


def terminal(record, cfg):
    """Always reap and audit, including already-exited/nonzero workers."""
    if record.get('terminal_done'):
        if record.get('terminal_first') is not None: raise record['terminal_first']
        return record['terminal_errors']
    first = None; failures = []
    process = record['process']
    if process.poll() not in (None, 0): first = RuntimeError('worker nonzero exit '+str(process.returncode))
    for fn in (lambda: cleanup_owned(record), lambda: resource_check(record['out'],
               dict(wall_seconds=cfg['job_wall_seconds'], output_bytes=cfg['job_output_bytes']), record['started'])):
        try: fn()
        except BaseException as exc: failures.append(error(exc)); first = first or exc
    record.update(terminal_done=True, terminal_first=first, terminal_errors=failures)
    # Even an uncatchable worker death has explicit last durable observations;
    # no inference of physical calls from prediction-file or visit counts.
    try:
        snapshots = sorted(record['out'].path.glob('cost_[0-9]*.json'))
        p = snapshots[-1] if snapshots else record['out'].path/'cost_initial.json'
        last = json.loads(p.read_text()) if p.exists() else None
        record['out'].evidence('terminal_cost.json', dict(last_durable=last,
            completeness='LOWER_BOUND' if first else 'WORKER_EVIDENCE_REQUIRED',
            unobserved_tail='unknown' if first else None))
    except BaseException as exc:
        failures.append(error(exc)); first = first or exc
        record['terminal_first'] = first
    if first: raise first
    return failures


def supervise(approved, out, start):
    """At most three live jobs, one per order lane. Fresh OS process per job."""
    cfg = approved['config']; queues = schedule(cfg['workers']); active = {}; completed = []
    started = time.monotonic(); first = None; secondary = []; handlers = {}
    pending_signal = None
    def interrupted(signum, frame):
        nonlocal pending_signal
        pending_signal = signum
    def check_signal():
        if pending_signal is not None: raise InterruptedError('screen supervisor signal '+str(pending_signal))
    try:
        for sig in (signal.SIGINT, signal.SIGTERM): handlers[sig] = signal.signal(sig, interrupted)
        while active or any(queues):
            check_signal()
            resource_check(out, cfg, started)
            # Audit every current worker BEFORE permitting any next dispatch.
            for slot, rec in list(active.items()):
                if rec['process'].poll() is None:
                    resource_check(rec['out'], dict(wall_seconds=cfg['job_wall_seconds'], output_bytes=cfg['job_output_bytes']), rec['started'])
                    continue
                terminal(rec, cfg)
                rec['out'].evidence('terminal.json', dict(exit_code=rec['process'].returncode, cleaned=rec.get('cleaned', False), errors=rec['terminal_errors']))
                summary = json.loads((rec['out'].path/'worker.json').read_text())
                if summary['status'] != 'JOB_PENDING_TERMINAL_AUDIT' or summary['target_after_check'] != 'UNCHANGED':
                    raise ValueError('worker after-read evidence absent/failed')
                rec['out'].evidence('completion.pending.json', dict(job=rec['job'], worker=slot, status='JOB_COMPLETE', terminal_resource_audit=True))
                resource_check(rec['out'], dict(wall_seconds=cfg['job_wall_seconds'], output_bytes=cfg['job_output_bytes']), rec['started'], reserve=0)
                (rec['out'].path/'completion.pending.json').rename(rec['out'].path/'completion.json')
                completed.append(rec['job']); del active[slot]
            for slot, queue in enumerate(queues):
                check_signal()
                if any(rec['process'].poll() not in (None, 0) for rec in active.values()):
                    raise RuntimeError('worker nonzero exit '+str(next(rec['process'].returncode for rec in active.values() if rec['process'].returncode not in (None, 0))))
                if slot in active or not queue: continue
                job = queue.pop(0)
                job_out = BudgetOutput(out.path/job['job_id'], dict(binding_sha256=json_digest(approved['binding']), job=job, worker=slot), cfg['job_output_bytes'])
                rec = dict(job=job, out=job_out, started=time.monotonic())
                rec['process'] = start(slot, job, job_out)
                active[slot] = rec  # Own it before any fallible persistence.
                check_signal()
                job_out.write('process.json', dict(pid=rec['process'].pid, pgid=rec['process'].pid, worker=slot))
            if active: time.sleep(.1)
        rows = []
        for job in matrix(SCOPE): rows.extend(json.loads((out.path/job['job_id']/'scalars.private.json').read_text()))
        out.write('report.json', report(rows))
    except BaseException as exc: first = exc
    finally:
        for sig in handlers: signal.signal(sig, signal.SIG_IGN)
        for rec in active.values():
            try: terminal(rec, cfg)
            except BaseException as exc: secondary.append(dict(job=rec['job']['job_id'], first=error(exc), terminal=rec.get('terminal_errors', []))); first = first or exc
        for sig, old in handlers.items(): signal.signal(sig, old)
        try: resource_check(out, cfg, started)
        except BaseException as exc: secondary.append(error(exc)); first = first or exc
        records = [('supervisor.json', dict(complete=first is None, completed_jobs=completed, retry=False, secondary=secondary))]
        if first: records.insert(0, ('supervisor.first_error.json', error(first)))
        for name, value in records:
            try: out.evidence(name, value)
            except BaseException as exc:
                first = first or exc
                try: print(json.dumps(dict(evidence_error=error(exc), first_error=error(first))), file=sys.stderr, flush=True)
                except BaseException: pass
    if first: raise first
    # Matrix completion also waits for all terminal evidence and report bytes.
    out.evidence('completion.pending.json', dict(status='TARGET_SCREEN_COMPLETE_PENDING_SCIENTIFIC_REVIEW', jobs=24, automatic_nomination=False, next_execution_authorized=False))
    resource_check(out, cfg, started, reserve=0)
    (out.path/'completion.pending.json').rename(out.path/'completion.json')


def matrix_readiness(approved):
    """Zero-forward fresh loading of all arms before any target reader is called."""
    checked = []
    for arm in ARMS:
        host, state = make_host(approved, arm)
        try:
            if arm == 'C_BASE': host.finish(state)
            else: host._check_frozen(boundary=True)
        finally:
            if arm != 'C_BASE': host.segmenter.close()
        del host, state
        checked.append(arm)
    if approved['config']['device'] != 'cpu': torch.cuda.empty_cache()
    return dict(arms=checked, model_forwards=0, target_pixel_reads=0)


def process_audit(out, gpu=False):
    command = subprocess.check_output(['ps', '-ww', '-p', str(os.getpid()), '-o', 'args='], text=True).strip()
    from ..b3_runtime import FORBIDDEN
    if any(word in command.lower() for word in FORBIDDEN): raise ValueError('non-neutral process command')
    out.write('process_audit.json', dict(command=command, neutral=True, GPU_execution=gpu))


def main():
    path = os.environ.get('SCREEN_RECEIPT')
    if not path: raise PermissionError('no receipt; TARGET_SCREEN disabled')
    receipt = json.loads(Path(path).read_bytes()); approved = preflight(receipt)
    b = approved['binding']; cfg = approved['config']
    out = BudgetOutput(b['output_dir'], dict(binding_sha256=json_digest(b)), cfg['output_bytes'])
    from ..b3_runtime import ENTRY
    process_audit(out)
    out.write('authorization.json', receipt)
    def start(slot, job, job_out):
        env = dict(os.environ, RUN_FILE=str(ROOT/'scripts/r7/target_screen.py'), SCREEN_CHILD='1',
                   SCREEN_RECEIPT=str(out.path/'authorization.json'), SCREEN_JOB=job['job_id'],
                   SCREEN_SLOT=str(slot), SCREEN_PARENT_OWNER=job_out.owner,
                   CUDA_VISIBLE_DEVICES='' if cfg['device'] == 'cpu' or job['arm'] == 'C_BASE' else str(cfg['physical_GPU_ids'][slot]), CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
        with (job_out.path/'worker.log').open('xb') as log:
            return subprocess.Popen([sys.executable, '-c', ENTRY], cwd=ROOT, env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # Readiness is its own bounded owned process, never a target job. All eight
    # factories must load successfully before supervise can dispatch C_BASE.
    ready = BudgetOutput(out.path/'readiness', dict(binding_sha256=json_digest(b)), cfg['job_output_bytes'])
    env = dict(os.environ, RUN_FILE=str(ROOT/'scripts/r7/target_screen.py'), SCREEN_READY='1',
               SCREEN_RECEIPT=str(out.path/'authorization.json'), CUDA_VISIBLE_DEVICES='' if cfg['device']=='cpu' else str(cfg['physical_GPU_ids'][0]),
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1')
    rec = dict(out=ready, started=time.monotonic())
    try:
        with (ready.path/'worker.log').open('xb') as log:
            rec['process'] = subprocess.Popen([sys.executable, '-c', ENTRY], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        while rec['process'].poll() is None:
            resource_check(ready, dict(wall_seconds=cfg['job_wall_seconds'], output_bytes=cfg['job_output_bytes']), rec['started'])
            time.sleep(.1)
        terminal(rec, cfg)
        evidence = json.loads((ready.path/'matrix_readiness.json').read_text())
        if evidence['arms'] != ARMS: raise ValueError('incomplete matrix readiness')
    except BaseException as first:
        if 'process' in rec:
            try: terminal(rec, cfg)
            except BaseException: pass
        try: out.evidence('readiness.first_error.json', error(first))
        except BaseException as secondary:
            print(json.dumps(dict(first_error=error(first), evidence_error=error(secondary))), file=sys.stderr, flush=True)
        raise
    supervise(approved, out, start)


def child_entry():
    # Parent-created root is expected; revalidate original authority except freshness
    # via explicit ownership, never by weakening public preflight's fresh-output gate.
    receipt = json.loads(Path(os.environ['SCREEN_RECEIPT']).read_bytes())
    b = receipt['binding']; root = checked_path(b['output_dir'])
    owner = json.loads((root/'owner.json').read_text())
    if owner['binding'] != dict(binding_sha256=json_digest(b)): raise ValueError('parent binding')
    approved = preflight(receipt, owned_output=owner['owner'])
    slot = int(os.environ['SCREEN_SLOT']); job_id = os.environ['SCREEN_JOB']
    jobs = schedule(approved['config']['workers'])
    if slot not in range(len(jobs)): raise ValueError('worker slot')
    job = next((j for j in jobs[slot] if j['job_id'] == job_id), None)
    if job is None: raise ValueError('job/worker assignment')
    p = checked_path(root/job_id); record = json.loads((p/'owner.json').read_text())
    expected = dict(binding_sha256=json_digest(b), job=job, worker=slot)
    if record != dict(owner=os.environ['SCREEN_PARENT_OWNER'], binding=expected): raise ValueError('job ownership')
    out = object.__new__(BudgetOutput); out.path=p; out.owner=record['owner']; out.binding=expected
    out.limit=approved['config']['job_output_bytes']; out.used=0; out.started=time.monotonic()
    torch.set_num_threads(2); torch.set_default_dtype(torch.float32)
    gpu = approved['config']['device'] != 'cpu' and job['arm'] != 'C_BASE'
    configure_backend(approved['config'] if gpu else dict(device='cpu'))
    process_audit(out, gpu)
    execute_job(approved, job, out)


def readiness_entry():
    receipt = json.loads(Path(os.environ['SCREEN_RECEIPT']).read_bytes())
    root = Path(receipt['binding']['output_dir'])
    owner = json.loads((root/'owner.json').read_text())
    approved = preflight(receipt, owned_output=owner['owner'])
    configure_backend(approved['config']); torch.set_num_threads(2)
    result = matrix_readiness(approved)
    with (root/'readiness/matrix_readiness.json').open('x') as f: json.dump(result, f)
