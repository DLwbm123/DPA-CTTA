from dpa_ctta.r8_ba.scope import SCREEN, GRAPH_PATH, SPEC_PATH, SOURCE_JOBS, TARGET_JOBS
"""Detached fixed-matrix scheduler. Worker commands contain only neutral paths."""
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from dpa_ctta.r8_ba.execution_plan import tasks
from dpa_ctta.r8_ba.inputs import bind_metadata
from dpa_ctta.r8_ba.journal import _replace
from dpa_ctta.r8_ba.ledger import Ledger
from dpa_ctta.r8_ba.launch_binding import verify
from dpa_ctta.r8_ba.paths import SOURCE_ROOT, TARGET_ROOT
from dpa_ctta.r8_ba.protocol import PROTOCOL_SHA256
from dpa_ctta.r8_ba.source_artifacts import digest, select_completed_grid, source_index, lock_targets

ROOT = Path(__file__).resolve().parents[2]


def write(path, value):
    _replace(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())


def main():
    launch = json.loads(Path(os.environ['R8_QUEUE_CONFIG']).read_text())
    if launch.get('scope','FULL') != ('SCREEN24' if SCREEN else 'FULL'):
        raise ValueError('R8 queue scope binding')
    verify(launch, ROOT)
    run = Path(launch['run_root']).resolve()
    package = SOURCE_ROOT.parent
    if not run.is_relative_to(package / 'runs') or run == package / 'runs':
        raise ValueError('R8 queue output isolation')
    run.mkdir(parents=True, exist_ok=True)
    lock = (run / 'queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    graph = json.loads(GRAPH_PATH.read_text())
    configs = json.loads(SPEC_PATH.read_text())['configs']
    work, cpu_deps = tasks(graph)
    identity = dict(code_sha=launch['code_sha'], protocol_sha256=PROTOCOL_SHA256,
                    graph_sha256=digest(GRAPH_PATH))
    report = json.loads(Path(launch['admission']['path']).read_text())
    if (digest(launch['admission']['path']) != launch['admission']['sha256'] or
            report['identity'] != identity or report['projection']['status'] != 'WITHIN_PROPOSED_CAPS' or
            set(report['budgets']) != {row['id'] for row in work}):
        raise ValueError('R8 launch admission binding')
    bound = bind_metadata(launch['refs'])
    common = dict(code_sha=launch['code_sha'], protocol_sha256=PROTOCOL_SHA256,
                  refs=launch['refs'], source_root=launch['source_root'], target_root=launch['target_root'],
                  checkpoint_path=launch['checkpoint_path'], code_inventory=launch['code_inventory'],
                  gpu_uuids=launch['gpu_uuids'], required_gpu_bytes=launch['required_gpu_bytes'],
                  environment=launch['environment'])
    ledger = Ledger(run / 'ledger', identity)
    state_path = run / 'queue.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['identity'] != identity or state['launch_sha256'] != digest(os.environ['R8_QUEUE_CONFIG']):
            raise ValueError('R8 queue restart identity')
        # An unknown live PID must not be relaunched. Monitoring reconciles orphaned attempts.
        if any(row['status'] == 'RUNNING' for row in state['tasks'].values()):
            raise RuntimeError('R8 prior queue has unreconciled running workers')
    else:
        ledger.create(report['prior_cost'], report['prior_evidence'])
        state = dict(schema='R8_QUEUE_V1', scope='SCREEN24' if SCREEN else 'FULL', started_unix=time.time(), identity=identity,
                     launch_sha256=digest(os.environ['R8_QUEUE_CONFIG']), status='RUNNING',
                     tasks={row['id']: dict(status='PENDING', attempts=[]) for row in work},
                     cpu={key: 'PENDING' for key in cpu_deps})
        write(state_path, state)
    source_dir = SOURCE_ROOT / run.name
    target_dir = TARGET_ROOT / run.name
    source_dir.mkdir(parents=True, exist_ok=True)
    target_dir.mkdir(parents=True, exist_ok=True)
    roots = {row['id']: str(source_dir / row['id']) for row in work if row['kind'] == 'SOURCE_JOB'}
    by_config = {row['id']: row for row in configs}
    if SCREEN:
        write(source_dir / 'predeclared.json', dict(schema='R8_SOURCE_ONLY_GRID_SELECTION_V1',
              protocol_sha256=PROTOCOL_SHA256, selected_config={c['route']:c['id'] for c in configs},
              selection_kind='user-authorized predeclared configurations'))
    active = {}
    stopped = False
    def stop_signal(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop_signal)
    signal.signal(signal.SIGINT, stop_signal)
    def done(key):
        return state['cpu'].get(key) == 'COMPLETE' or state['tasks'].get(key, {}).get('status') == 'COMPLETE'
    def run_cpu(key):
        if SCREEN and key == 'oracle_0.3':
            from dpa_ctta.r8_ba.oracle_shards import merge
            identity = dict(code_sha=launch['code_sha'], protocol_sha256=PROTOCOL_SHA256,
                refs=launch['refs'],checkpoint_sha256=bound['docs']['manifest']['checkpoint']['sha256'],amplitude=0.3)
            merge(source_dir / 'oracle_0.3', [source_dir / f'oracle_shard_{i}' for i in range(3)],
                  identity, bound['docs']['split']['folds'])
        elif SCREEN and key == 'grid_selection':
            from dpa_ctta.r8_ba.screen import select_grid
            write(source_dir / 'selection.json', select_grid(graph,configs,roots,launch['code_sha']))
        elif key == 'grid_selection':
            grid_roots = {k: roots[k] for k in cpu_deps[key]}
            costs = {}
            for candidate in configs:
                ids = [j['id'] for j in graph['jobs'] if j['stage'] == 'SOURCE_GRID' and j['config'] == candidate['id']]
                source_cost = sum(sum(json.loads((run / 'ledger' / ('attempt-' + a + '.json')).read_text())['observed']['gpu_seconds']
                                      for a in state['tasks'][ident]['attempts']) for ident in ids)
                online = report['profile']['units']['new_' + candidate['route'].lower() + '_max']['gpu_seconds']
                costs[candidate['id']] = source_cost + online * 1951 * 5 * 4
            selection = select_completed_grid(graph, configs, grid_roots, launch['code_sha'], costs,
                                               dict(ledger=str(run / 'ledger'), admission=launch['admission']))
            write(source_dir / 'selection.json', selection)
        elif key == 'source_index':
            selection = json.loads((source_dir / 'selection.json').read_text())
            basis_refs = {}
            for amplitude in (('0.3',) if SCREEN else ('0.1', '0.3')):
                directory = source_dir / ('bases_' + amplitude)
                basis_refs[amplitude] = dict(root=str(directory), identity=json.loads((directory / 'worker_complete.json').read_text())['identity'])
            result = source_index(graph, configs, selection, roots, launch['code_sha'], launch['refs'],
                                  bound['docs']['manifest']['checkpoint']['sha256'], basis_refs, launch['r7_inventory'])
            write(source_dir / 'source_index.json', result)
        else:
            path = source_dir / 'source_index.json'
            if SCREEN:
                from dpa_ctta.r8_ba.screen import lock_targets as lock_screen
                result = lock_screen(json.loads(path.read_text()), digest(path), source_dir / 'capacity_0.3')
            else:
                result = lock_targets(json.loads(path.read_text()), digest(path), source_dir / 'gradient_lr',
                                      {a: str(source_dir / ('capacity_' + a)) for a in ('0.1', '0.3')})
            write(source_dir / 'artifact_lock.json', result)
        state['cpu'][key] = 'COMPLETE'
        write(state_path, state)
    def config_for(row, gpu, attempt):
        value = dict(common, schema='R8_' + row['kind'] + '_WORK_V1', physical_gpu=gpu)
        job = row['job']
        value['job_root'] = str((target_dir / job['id']) if row['kind'] in ('TARGET_JOB', 'SCORE_JOB') else source_dir / row['id'])
        if row['amplitude'] is not None:
            value['amplitude'] = row['amplitude']
            value['oracle_root'] = str(source_dir / ('oracle_' + str(row['amplitude'])))
            value['bases_root'] = str(source_dir / ('bases_' + str(row['amplitude'])))
        if row['kind'] == 'ORACLE_SHARD':
            value['shard'] = int(row['id'].rsplit('_',1)[1])
        if row['kind'] == 'SOURCE_JOB':
            cid = job['config']
            if job['stage'] != 'SOURCE_GRID':
                selection = source_dir / ('predeclared.json' if SCREEN else 'selection.json')
                route = 'B' if job['stage'] == 'SOURCE_MLP' else job['arm']
                cid = json.loads(selection.read_text())['selected_config'][route]
                value['selection_path'] = str(selection)
            candidate = by_config[cid]
            value.update(job_id=job['id'], config=candidate, source_seed=job['source_seed'], mode=job['mode'],
                         oracle_root=str(source_dir / ('oracle_' + str(candidate['film_amplitude']))),
                         bases_root=str(source_dir / ('bases_' + str(candidate['film_amplitude']))),
                         scaler_root=str(source_dir / 'scaler'))
        elif row['kind'] == 'GRADIENT_LR':
            path = source_dir / 'source_index.json'
            index = json.loads(path.read_text())
            candidate = by_config[index['selected_config']['B']]
            value.update(source_index=dict(path=str(path), sha256=digest(path)),
                         oracle_root=str(source_dir / ('oracle_' + str(candidate['film_amplitude']))))
        elif row['kind'] in ('TARGET_JOB', 'SCORE_JOB'):
            path = source_dir / 'artifact_lock.json'
            value.update(job_id=job['id'], artifact_lock=dict(path=str(path), sha256=digest(path)))
        failure = state['tasks'][row['id']].get('resume_failure') or launch.get('initial_recoveries', {}).get(row['id'])
        if failure:
            value['resume_failure'] = failure
        budget = report['budgets'][row['id']]
        value.update(maximum_seconds=int(budget['gpu_seconds']) - 1,
                     execution_ledger=dict(root=str(run / 'ledger'), identity=identity, attempt_id=attempt, budget=budget))
        return value
    try:
        while True:
            if SCREEN and time.time()-state['started_unix'] >= 24*3600:
                raise RuntimeError('R8 SCREEN24 wall deadline reached; preserve incomplete jobs')
            if stopped:
                raise RuntimeError('R8 queue interrupted; preserve all reservations')
            ledger.snapshot()
            for gpu, (process, row, output, start, deadline) in list(active.items()):
                if time.monotonic() - start >= deadline:
                    raise RuntimeError('R8 worker deadline reached')
                code = process.poll()
                if code is None:
                    continue
                output.close()
                del active[gpu]
                item = state['tasks'][row['id']]
                receipt = run / 'ledger' / ('attempt-' + item['attempts'][-1] + '.json')
                evidence = json.loads(receipt.read_text()) if receipt.exists() else {}
                if code == 0 and evidence.get('status') == 'COMPLETE':
                    item['status'] = 'COMPLETE'
                else:
                    item.update(status='FAILED', exit_code=code, failure=evidence)
                    write(state_path, state)
                    # Fail closed on unknown/identity failures. An evidenced infrastructure
                    # retry must first pass the journal's full equivalence validation.
                    if (evidence.get('error_type') == 'FloatingPointError' or
                            evidence.get('error') == 'non-finite tensor' or
                            (evidence.get('error_type') in ('RuntimeError', '_LinAlgError') and
                             'not positive-definite' in evidence.get('error', ''))):
                        item['status'] = 'NUMERICAL_FAILED'
                    elif (evidence.get('error_type') in ('OSError', 'TimeoutError', 'ConnectionError', 'InterruptedError') and
                          evidence.get('errno') != 28 and len(item['attempts']) + launch.get('previous_attempt_counts', {}).get(row['id'], 0) == 1 and row['kind'] != 'BASES'):
                        item.update(status='PENDING', resume_failure=dict(
                            **{'class': 'INFRASTRUCTURE'}, reason=evidence.get('error') or evidence['error_type'],
                            evidence=dict(attempt=str(receipt), sha256=digest(receipt), code_sha=launch['code_sha'])))
                    else:
                        raise RuntimeError('R8 worker failure requires classification: ' + row['id'])
                write(state_path, state)
            for key, deps in cpu_deps.items():
                if state['cpu'][key] == 'PENDING' and all(done(dep) for dep in deps):
                    run_cpu(key)
            for gpu in (5, 6, 7):
                if gpu in active:
                    continue
                ready = next((r for r in sorted(work, key=lambda item: -report['budgets'][item['id']]['gpu_seconds']) if state['tasks'][r['id']]['status'] == 'PENDING' and
                              all(done(dep) for dep in r['dependencies'])), None)
                if ready is None:
                    continue
                item = state['tasks'][ready['id']]
                attempt = 'w' + str(work.index(ready)).zfill(4) + '_' + str(len(item['attempts']))
                config = config_for(ready, gpu, attempt)
                path = run / (attempt + '.json')
                write(path, config)
                ledger.reserve(attempt, gpu, config['execution_ledger']['budget'], digest(path))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), R8_WORK_CONFIG=str(path))
                env.pop('R8_QUEUE_CONFIG', None)
                output = (run / (attempt + '.log')).open('xb')
                process = subprocess.Popen([launch['python'], launch['worker_entry']], env=env,
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                item.update(status='RUNNING', pid=process.pid, physical_gpu=gpu, config=str(path),
                            started_unix=time.time())
                item['attempts'].append(attempt)
                active[gpu] = (process, ready, output, time.monotonic(), config['maximum_seconds'] + 2)
                write(state_path, state)
            if not active:
                pending = [k for k,v in state['tasks'].items() if v['status'] == 'PENDING']
                state['status'] = 'BLOCKED_DEPENDENCIES' if pending else 'EXECUTION_COMPLETE'
                write(state_path, state)
                return
            time.sleep(2)
    except BaseException as exc:
        state.update(status='STOPPED', error_type=type(exc).__name__, error=str(exc))
        write(state_path, state)
        try:
            ledger.stop(str(exc))
        finally:
            for process, _, output, _, _ in active.values():
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGTERM)
                output.close()
        raise


if __name__ == '__main__':
    main()
