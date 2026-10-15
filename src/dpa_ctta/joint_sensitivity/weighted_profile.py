"""Bounded resource/derivative qualification only; cannot launch formal trajectories."""
import concurrent.futures
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time
import traceback

from .acquisition_weighted import Host as WeightedHost, candidates as weighted_candidates
from .factorial import Host as FactorialHost, candidates as factorial_candidates
from .method import Host as JointHost, flatten
from .shared_span import candidates as shared_candidates


def conditions():
    return [factorial_candidates()[0], factorial_candidates()[2],
            shared_candidates()[-1], weighted_candidates()[-1]]


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def summary(values):
    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values))


def derivatives(gradient, proposed, actual):
    proposal = float(gradient.double()@proposed.double())
    submitted = float(gradient.double()@actual.double())
    if not all(math.isfinite(value) for value in (proposal, submitted)):
        raise ValueError('finite measured directional derivatives required')
    return dict(proposal_dot=proposal, actual_dot=submitted, dot_change=submitted-proposal)


def worker(config, job):
    import torch
    from ..r36_incremental_modules.run import filesystem_guard
    from ..r19_model_only.runtime import image
    from ..r7_target_screen.runner import TargetReader
    root = Path(config['output_root'])
    if os.environ['CUDA_VISIBLE_DEVICES'] != str(job['gpu']) or job['gpu'] not in (4, 5, 6, 7):
        raise ValueError('physical GPU binding')
    torch.set_num_threads(2)
    permit, opened, rejected = filesystem_guard(config)
    state = torch.load(config['checkpoint_path'], map_location='cpu', weights_only=True)
    candidate = conditions()[job['condition_index']]
    factory = (WeightedHost if candidate['id']=='C_JS_ACQ_WEIGHTED' else
               JointHost if candidate['id']=='CW_LSO_JS_SHARED' else FactorialHost)
    started = time.perf_counter()
    host = factory(state, candidate, config['seed'], 'bounded-resource-profile', device='cuda:0')
    torch.cuda.synchronize()
    initialization = time.perf_counter()-started
    cold_allocated, cold_reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    torch.cuda.reset_peak_memory_stats()
    rows = json.loads((root/'private'/f'INPUT_o{job["order"]}.json').read_text())
    if len(rows) != 32:
        raise ValueError('exact frozen 32-arrival profile prefix required')
    reader = TargetReader(config['target_root'], 256*1024**2, 'image')
    observation = {}
    def before(opt, *args):
        observation['before'] = flatten(host.native.params)
        observation['gradient'] = flatten([p.grad for p in host.native.params])
    def after(opt, *args):
        observation['proposal'] = flatten(host.native.params)-observation['before']
    handles = [host.native.base.register_step_pre_hook(before), host.native.base.register_step_post_hook(after)]
    wall, cuda, numerical, diagnostics = [], [], [], []
    joint = bool(candidate.get('joint_strength'))
    expected = dict(forwards=10 if joint else 8, backwards=5 if joint else 1,
                    base_adam=1, perturb=0, restore=0)
    def guard():
        if time.time() > float(os.environ['PROFILE_DEADLINE']):
            raise TimeoutError('bounded profile deadline')
    guard.extra = dict(image_accesses=0)
    try:
        for index, row in enumerate(rows):
            guard()
            pixels = image(reader, row, permit, guard)
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tick = time.perf_counter(); begin.record()
            logits, detail = host.step(pixels)
            end.record(); torch.cuda.synchronize()
            wall.append(time.perf_counter()-tick); cuda.append(begin.elapsed_time(end)/1000)
            if detail['native'][0]['counts'] != expected:
                raise ValueError('physical calls differ from frozen profile')
            actual = flatten(host.native.params)-observation['before']
            measured = derivatives(observation['gradient'], observation['proposal'], actual)
            if not joint and measured['actual_dot'] != measured['proposal_dot']:
                raise ValueError('C native update changed')
            diagnostic = detail['diagnostics'].get('joint', {})
            if candidate['id']=='C_JS_ACQ_WEIGHTED':
                if measured['proposal_dot'] != diagnostic['loss_proposal_dot'] or measured['actual_dot'] != diagnostic['loss_actual_dot']:
                    raise ValueError('independent hook derivative disagrees with method')
            numerical.append(measured)
            diagnostics.append(diagnostic)
            with (root/'private'/f'{job["id"]}.steps.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(visit=index+1, derivatives=measured, joint=diagnostic), allow_nan=False)+'\n')
            save(root/'progress'/f'{job["id"]}.json', dict(visit=index+1, total=32))
            del pixels, logits
        host.check_frozen(True)
        result = dict(status='PASS', condition=candidate['id'], physical_gpu=job['gpu'], order=job['order'],
            arrivals=32, warmup_arrivals=4, timed_arrivals=28,
            wall_seconds=summary(wall[4:]), cuda_event_seconds=summary(cuda[4:]),
            initialization_seconds=initialization,
            peak_allocated_bytes=max(cold_allocated, torch.cuda.max_memory_allocated()),
            peak_reserved_bytes=max(cold_reserved, torch.cuda.max_memory_reserved()), counts_per_image=expected,
            derivatives={key: summary([row[key] for row in numerical]) for key in numerical[0]},
            native_non_descent=sum(row['proposal_dot']>=0 for row in numerical),
            actual_non_descent=sum(row['actual_dot']>=0 for row in numerical),
            derivative_increased=sum(row['dot_change']>0 for row in numerical),
            source_image_reads=0, label_reads=0, target_image_reads=guard.extra['image_accesses'],
            access_rejections=len(rejected)-2, trainable_BN_parameters=sum(p.numel() for p in host.native.params))
        if joint:
            result['joint'] = dict(rank=summary([d['rank'] for d in diagnostics]),
                relative_update_change=summary([d.get('relative_metric_update_change', 0.) for d in diagnostics]),
                mean_filter_eigenvalues=[statistics.mean(d['filter_eigenvalues'][i] for d in diagnostics)
                                        for i in range(min(d['rank'] for d in diagnostics))])
        save(root/'results'/f'{job["id"]}.json', result)
    finally:
        for handle in handles: handle.remove()
        host.close()


def run(config):
    root = Path(config['output_root'])
    if config['formal_experiment_authorized'] or config['arrivals'] != 32 or config['warmup'] != 4:
        raise ValueError('profile-only frozen scope')
    with (root/'started.json').open('x') as stream:
        json.dump(dict(at=time.time(), pid=os.getpid()), stream)
    started = time.time()
    stop = threading.Event(); lock = threading.Lock(); receipts = []; live = {}
    def lane(gpu, order, sequence):
        for index in sequence:
            if stop.is_set(): return
            job = dict(id=f'g{gpu}_o{order}_c{index}', gpu=gpu, order=order, condition_index=index)
            if (root/'attempts'/f'{job["id"]}.json').exists():
                raise ValueError('no repeated profile launch')
            free = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'], text=True)
            memory = {int(row.split(',')[0]):int(row.split(',')[1]) for row in free.splitlines()}
            if memory[gpu] < 5120:
                stop.set(); raise RuntimeError('cold profile 5GiB free-memory guard')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PROFILE_JOB=json.dumps(job), PROFILE_MODE='worker',
                       PROFILE_DEADLINE=str(min(started+config['wall_cap_seconds'], time.time()+config['job_cap_seconds'])))
            tick = time.time()
            with (root/'private'/f'{job["id"]}.log').open('x') as log:
                process = subprocess.Popen([os.environ['PROFILE_PYTHON'], os.environ['PROFILE_ENTRY']], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                with lock: live[job['id']] = tick
                save(root/'processes'/f'{job["id"]}.json', dict(pid=process.pid, active=True, started=tick, job=job))
                last_disk_check = 0.
                while process.poll() is None:
                    with lock:
                        charged = sum(a['gpu_worker_seconds'] for a in receipts)+sum(time.time()-value for value in live.values())
                    if time.time()-last_disk_check > 10:
                        last_disk_check = time.time()
                        if sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) > config['output_cap_bytes']:
                            stop.set()
                    if stop.is_set() or time.time()>float(env['PROFILE_DEADLINE']) or charged>config['gpu_cap_seconds']:
                        process.terminate()
                        try: process.wait(timeout=10)
                        except subprocess.TimeoutExpired: process.kill(); process.wait()
                        break
                    time.sleep(.5)
                elapsed = time.time()-tick
            receipt = dict(job=job, exit_code=process.returncode, status='COMPLETE' if process.returncode==0 else 'FAILED',
                           gpu_worker_seconds=elapsed, started=tick, ended=time.time())
            with lock:
                del live[job['id']]; receipts.append(receipt)
            save(root/'attempts'/f'{job["id"]}.json', receipt)
            save(root/'processes'/f'{job["id"]}.json', dict(pid=process.pid, active=False, job=job))
            if process.returncode:
                stop.set(); return
    error = None
    try:
        with concurrent.futures.ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(lane, 5, 0, [0, 1, 2, 3]), pool.submit(lane, 7, 1, [3, 2, 1, 0])]
            for future in concurrent.futures.as_completed(jobs):
                try: future.result()
                except Exception:
                    stop.set(); error = traceback.format_exc()
    finally:
        status = 'COMPLETE' if len(receipts)==8 and all(a['status']=='COMPLETE' for a in receipts) else 'INCOMPLETE'
        save(root/'RESOURCE_LEDGER.json', dict(status=status, gpu_worker_seconds=sum(a['gpu_worker_seconds'] for a in receipts),
             wall_seconds=time.time()-started, attempts=receipts, failure=error,
             charging='Full subprocess elapsed time, including imports, initialization, diagnostics and failures', formal_experiment_started=False))
        save(root/'RUN_STATE.json', dict(status=status, completed=sum(a['status']=='COMPLETE' for a in receipts), planned=8,
             formal_experiment_started=False, supervisor_retiring=True))


def main():
    config = json.loads(Path(os.environ['PROFILE_CONFIG']).read_text())
    if os.environ.get('PROFILE_MODE') == 'worker': worker(config, json.loads(os.environ['PROFILE_JOB']))
    elif os.environ.get('PROFILE_MODE') == 'controller': run(config)
    else: raise ValueError('profile controller or worker only')


if __name__ == '__main__': main()
