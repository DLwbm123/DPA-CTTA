"""Profile-only queue: wait for R44 retirement, then four bounded real-model jobs."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time
import torch
from .method import Host
from ..r20_model_only_search import runtime as base
from ..r36_incremental_modules.run import filesystem_guard
from ..r19_model_only.runtime import weights, image
from ..r7_target_screen.runner import TargetReader
from ..r10_12h_core.run import read, save, sha
from ..r9_current_first.storage import lease

ID = 'JOINT_SENSITIVITY_PROFILE_ONLY'


def alive(pid, start_ticks=None):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split()
        return fields[2] != 'Z' and (start_ticks is None or fields[21] == str(start_ticks))
    except FileNotFoundError:
        return False


def r44_retired(root):
    root = Path(root)
    state = read(root/'RUN_STATE.json')
    if state['status'] in ('INCOMPLETE', 'BLOCKED'):
        raise RuntimeError('R44 stopped unsuccessfully; profile queue will not intervene')
    if state['status'] != 'COMPLETE':
        return False
    if len(state['jobs']) != 24 or any(v != 'COMPLETE' for v in state['jobs'].values()):
        raise ValueError('R44 completion coverage mismatch')
    processes = [read(p) for p in (root/'processes').glob('*.json')]
    if any(p.get('active') or alive(p['pid'], p.get('start_ticks')) for p in processes):
        return False
    launch = read(root/'LAUNCH.json'); watch = read(root/'watchdog.json')
    return not any(alive(pid) for pid in (launch['pid'], watch['supervisor_pid']))


def check_joint_update(candidate, diagnostic):
    if candidate.get('joint_descent_constraint', False):
        proposal, actual = diagnostic['descent_proposal_dot'], diagnostic['descent_actual_dot']
        if not all(math.isfinite(v) for v in (proposal, actual)):
            raise ValueError('nonfinite loss descent diagnostic')
        if proposal < 0 and actual-proposal > 1e-5*max(abs(proposal), 1e-10):
            raise ValueError('real-model first-order loss descent violated')
    elif diagnostic['rank'] and diagnostic['filtered_metric_norm'] > diagnostic['proposal_metric_norm']*1.00001+1e-10:
        raise ValueError('real-model metric contraction violated')


def measured_profile(c, guard, permit, job):
    root = Path(c['output_root'])
    rows = read(root/'private/SCREEN_o0.json')
    if len(rows) != 12:
        raise ValueError('fixed 12-image profile required')
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    h = Host(weights(c), job['candidate'], c['native_seed'], sha(job))
    base.attach(guard.meter, h)
    torch.cuda.synchronize()
    initialization = time.perf_counter()-started
    cold_allocated = torch.cuda.max_memory_allocated()
    cold_reserved = torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    reader = TargetReader(c['target_root'], 256*1024**2, 'image')
    timings = []; device_timings = []; ranks = []; evidence = []
    enabled = bool(job['candidate'].get('joint_strength', 0))
    expected = dict(forwards=10 if enabled else 8, backwards=5 if enabled else 1,
                    base_adam=1, perturb=0, restore=0)
    try:
        for row in rows:
            guard(); pixels = image(reader, row, permit, guard)
            torch.cuda.synchronize()
            begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
            begin.record(); tick = time.perf_counter()
            logits, detail = h.step(pixels)
            end.record(); torch.cuda.synchronize()
            timings.append(time.perf_counter()-tick)
            device_timings.append(begin.elapsed_time(end)/1000)
            if detail['native'][0]['counts'] != expected:
                raise ValueError('real-model physical call mismatch')
            if enabled:
                diagnostic = detail['diagnostics']['joint']
                ranks.append(diagnostic['rank'])
                evidence.append(detail['diagnostics']['structure_evidence_mean'])
                check_joint_update(job['candidate'], diagnostic)
            del logits
        h.check_frozen(True)
        result = dict(status='PASS', condition=job['candidate']['id'], physical_gpu=job['physical_gpu'],
                      warmup_arrivals=4, timed_arrivals=8, seconds=timings[4:], cuda_event_seconds=device_timings[4:],
                      initialization_seconds=initialization, peak_allocated_bytes=max(cold_allocated, torch.cuda.max_memory_allocated()),
                      peak_reserved_bytes=max(cold_reserved, torch.cuda.max_memory_reserved()),
                      trainable_BN_parameters=sum(p.numel() for p in h.native.params), counts_per_image=expected,
                      sensitivity_ranks=ranks, evidence_channel_means=evidence, source_reads=0, label_reads=0)
        save(root/'profiles'/(job['id']+'.json'), result)
        return result
    finally:
        h.close()


def summaries(receipts):
    rows = []
    for receipt in receipts:
        if receipt['status'] != 'COMPLETE':
            continue
        p = receipt['result']; times = p['seconds']; device = p['cuda_event_seconds']
        ranks = p['sensitivity_ranks']; evidence = p['evidence_channel_means']
        rows.append(dict(condition=p['condition'], physical_gpu=p['physical_gpu'], timed_arrivals=len(times),
                         mean_seconds=statistics.mean(times), median_seconds=statistics.median(times),
                         p95_seconds=float(torch.tensor(times, dtype=torch.float64).quantile(.95)),
                         min_seconds=min(times), max_seconds=max(times), mean_cuda_event_seconds=statistics.mean(device),
                         peak_allocated_MiB=p['peak_allocated_bytes']/1024**2, peak_reserved_MiB=p['peak_reserved_bytes']/1024**2,
                         initialization_seconds=p['initialization_seconds'], trainable_BN_parameters=p['trainable_BN_parameters'],
                         counts_per_image=p['counts_per_image'], rank_min=min(ranks) if ranks else None,
                         rank_max=max(ranks) if ranks else None, rank_nonzero_arrivals=sum(r > 0 for r in ranks),
                         mean_evidence_channels=[statistics.mean(a[k] for a in evidence) for k in (0, 1)] if evidence else None))
    ratios = []
    for gpu in (0, 1):
        pair = {r['condition']: r for r in rows if r['physical_gpu'] == gpu}
        if set(pair) == {'CW_LSO', 'CW_LSO_JS'}:
            a, b = pair['CW_LSO'], pair['CW_LSO_JS']
            ratios.append(dict(physical_gpu=gpu, wall_time_ratio=b['mean_seconds']/a['mean_seconds'],
                               cuda_event_ratio=b['mean_cuda_event_seconds']/a['mean_cuda_event_seconds'],
                               allocated_ratio=b['peak_allocated_MiB']/a['peak_allocated_MiB'],
                               reserved_ratio=b['peak_reserved_MiB']/a['peak_reserved_MiB']))
    return rows, ratios


def report(c, state):
    root = Path(c['output_root'])
    receipts = [read(p) for p in sorted((root/'attempts').glob('*.json'))]
    rows, ratios = summaries(receipts)
    costs = [dict(phase=r['phase'], status=r['status'], **r.get('cost', {})) for r in receipts]
    result = dict(status=state['status'], formal_experiment_started=False, source_code=c['code_sha'],
                  profiles=rows, paired_ratios=ratios, cost=costs,
                  GPU_worker_seconds=sum(r.get('gpu_seconds', 0) for r in costs),
                  scope='12 fixed previously authorized images, no scoring; 4 warmup + 8 timed per condition/device',
                  timing_limit='shared GPUs; CUDA event elapsed includes host dispatch gaps, not isolated kernel time')
    save(root/'public/RESULTS.json', result)
    lines = ['# Joint sensitivity: real-model GPU profile only', '',
             f"Status: **{state['status']}**. Formal experiment started: false.", '',
             'Original source ResUNet checkpoint, batch1 RGB512; BN-affine CW_LSO versus CW_LSO_JS. '
             'GPU0 runs base then JS; GPU1 runs JS then base. Each fresh source initialization uses '
             'the same 12 prior authorized images and seed; 4 warmup,8 timed. No labels or predictions saved.', '',
             '| GPU | Condition | Mean s/image | P95 s/image | Peak allocated MiB | Peak reserved MiB |',
             '|---|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['physical_gpu']} | {r['condition']} | {r['mean_seconds']:.4f} | {r['p95_seconds']:.4f} | {r['peak_allocated_MiB']:.1f} | {r['peak_reserved_MiB']:.1f} |")
    for r in ratios:
        lines.append(f"\nGPU{r['physical_gpu']}: JS/base wall time {r['wall_time_ratio']:.3f}x; allocated {r['allocated_ratio']:.3f}x; reserved {r['reserved_ratio']:.3f}x.")
    lines += ['', f"Charged GPU-worker seconds, including initialization and failures: {result['GPU_worker_seconds']:.3f}.", '',
              'R44 must fully retire before this queue initializes CUDA. Its configuration, artifacts, '
              'workers and budget are not modified. Other users may share these GPUs; contention and '
              'eight timed images limit transfer of the latency estimate. All profiles/failures remain '
              'reported. No efficacy gate, new seed matrix, promotion or automatic formal run.']
    (root/'public/REPORT.md').write_text('\n'.join(lines)+'\n')


def watch():
    c = base.config(); root = Path(c['output_root'])
    if (root/'execution_started.json').exists():
        raise ValueError('profile queue already launched')
    save(root/'execution_started.json', dict(at=time.time()))
    state = dict(status='WAITING_R44_RETIREMENT', formal_experiment_started=False)
    save(root/'RUN_STATE.json', state)
    try:
        while not r44_retired(c['wait_root']):
            if time.time() > c['origin']['absolute_deadline_epoch']-1200:
                raise TimeoutError('resource wait cap; no GPU profile started')
            time.sleep(30)
        save(root/'R44_RETIREMENT_ADMISSION.json', dict(at=time.time(), status='COMPLETE_ALL_PROCESSES_RETIRED'))
        raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.free', '--format=csv,noheader,nounits'], text=True)
        gpus = {int(i): (u.strip(), int(m)) for i, u, m in (line.split(',') for line in raw.splitlines())}
        for assignment in c['gpu_assignments']:
            if assignment['physical_id'] not in (0, 1) or gpus[assignment['physical_id']][0] != assignment['uuid'] or gpus[assignment['physical_id']][1] < 6144:
                raise RuntimeError('authorized GPU memory admission failed; no retry')
        save(root/'GPU_START.json', dict(at=time.time(), free_MiB={g: gpus[g][1] for g in (0, 1)}))
        state['status'] = 'PROFILING_ONLY'; save(root/'RUN_STATE.json', state)
        for offset in (0, 2):
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                tasks = [pool.submit(base.run_task, c, 'profile_'+j['id'], c['gpu_assignments'][j['physical_gpu']], 450, j)
                         for j in c['jobs'][offset:offset+2]]
                receipts = [task.result() for task in tasks]
            if any(r['status'] != 'COMPLETE' for r in receipts):
                raise RuntimeError('bounded profile failed; no automatic retry or formal run')
        state['status'] = 'COMPLETE_PROFILE_ONLY'
    except Exception as error:
        state.update(status='INCOMPLETE_PROFILE_ONLY', reason=str(error)); raise
    finally:
        state['ended'] = time.time(); save(root/'RUN_STATE.json', state)
        base.ledger(c, state); report(c, state)


def main():
    base.ID = ID; base.Host = Host; base.filesystem_guard = filesystem_guard; base.profile = measured_profile
    if os.environ['RUN_MODE'] == 'watch':
        c = base.config()
        with lease(Path(c['output_root'])/'queue', dict(source=c['code_sha'], config=sha(c))):
            watch()
    elif os.environ['RUN_MODE'] == 'worker':
        if not os.environ['RUN_PHASE'].startswith('profile_'):
            raise ValueError('formal worker phase forbidden')
        receipt = base.worker()
        raise SystemExit(receipt['status'] != 'COMPLETE')
    else:
        raise ValueError('profile-only mode; formal execution forbidden')


if __name__ == '__main__': main()
