"""One user-requested GPU withdrawal; preserve frozen science and original budget."""
import concurrent.futures
import os
from pathlib import Path
import time
import traceback

from ..r10_12h_core.run import read, save, sha
from ..r36_incremental_modules import run as retained
from ..r37_cw_orientation import run as reporting
from .method import Host


def pending_jobs(jobs, receipts, interrupted_phase):
    pending = []
    for job in jobs:
        phase = 'online_' + job['id']
        rows = sorted((r for r in receipts if r['phase'] == phase), key=lambda r: r['attempt'])
        if rows and rows[-1]['status'] == 'COMPLETE':
            continue
        if rows:
            if (phase != interrupted_phase or len(rows) != 1 or rows[0]['attempt'] != 0
                    or rows[0]['failure'].get('exit_code') != -15):
                raise ValueError('only the evidenced user-interrupted attempt may restart')
            pending.append((job, 1))
        else:
            pending.append((job, 0))
    return pending


def alive(identity):
    p = Path('/proc') / str(identity['pid']) / 'stat'
    if not p.exists():
        return False
    try:
        parts = p.read_text().split()
    except FileNotFoundError:
        return False
    return parts[21] == identity['start_ticks'] and parts[2] != 'Z'


def supervise():
    base = retained.base
    c = base.config(); root = Path(c['output_root'])
    amendment = read(root/'GPU_RESOURCE_AMENDMENT.json')
    assert amendment['allowed_physical_ids'] == [0, 1]
    assignments = [a for a in c['gpu_assignments'] if a['physical_id'] in (0, 1)]
    assert [a['physical_id'] for a in assignments] == [0, 1]
    interrupted = amendment['interrupted'][0]['phase']
    state = read(root/'RUN_STATE.json')
    def persist():
        save(root/'RUN_STATE.json', state); base.ledger(c, state)
    with retained.lease(root/'supervisor', dict(experiment_id=retained.ID, config_sha256=sha(c))):
        try:
            assert not any(read(p).get('active') for p in (root/'processes').glob('*.json'))
            assert not (root/'REVIEW_RELEASE.json').exists()
            rows = [read(p) for p in (root/'attempts').glob('*.json')]
            pending = pending_jobs(c['jobs'], rows, interrupted)
            estimates = read(root/'PROFILE_ADMISSION.json')['estimates']
            seconds = sum(estimates[j['candidate']['id']] for j, _ in pending)
            assert base.amounts(c)[0]+seconds < c['origin']['gpu_worker_cap_seconds']-120
            assert time.time()+seconds/2+max(estimates.values()) < c['origin']['online_deadline_epoch']-600
            save(root/'RESOURCE_RESUME_ADMISSION.json', dict(admitted=True, at=time.time(),
                 allowed_physical_ids=[0, 1], pending=len(pending), projected_remaining_GPU_seconds=seconds,
                 original_T0=c['origin']['T0'], no_new_profile=True))
            for job, attempt in pending:
                if attempt == 1:
                    archive = root/'interrupted_outputs'/job['id']
                    archive.parent.mkdir(exist_ok=True)
                    assert not archive.exists()
                    (root/'target'/job['id']).rename(archive)
            state.pop('reason', None); state.pop('ended', None)
            state.update(status='RUNNING_FIXED_MATRIX', resource_allowed_physical_ids=[0, 1])
            persist()
            def batch(items, prefix, duration, cpu=False):
                for offset in range(0, len(items), 2):
                    with concurrent.futures.ThreadPoolExecutor(2) as pool:
                        fs = {}
                        for i, (job, attempt) in enumerate(items[offset:offset+2]):
                            if not cpu: state['jobs'][job['id']] = 'RUNNING'
                            fs[pool.submit(base.run_task, c, prefix+job['id'],
                                 None if cpu else assignments[i], duration, job, attempt)] = job
                        persist()
                        failed = False
                        for f in concurrent.futures.as_completed(fs):
                            receipt = f.result()
                            if not cpu: state['jobs'][fs[f]['id']] = receipt['status']
                            failed |= receipt['status'] != 'COMPLETE'
                            persist()
                    if failed: raise RuntimeError('task failed; no further automatic retry')
            batch(pending, 'online_', max(1800, 3*max(estimates.values())))
            assert not any(read(p).get('active') for p in (root/'processes').glob('*.json'))
            save(root/'stages/FINAL.barrier.json', dict(status='ALL_WORKERS_RETIRED', at=time.time()))
            save(root/'FINAL_JOBS.json', c['jobs'])
            save(root/'REVIEW_RELEASE.json', dict(frozen_sha256=sha(read(root/'FROZEN.json')),
                 all_formal_workers_retired=True, at=time.time()))
            state['status'] = 'SCORING'; persist()
            batch([(dict(id=j['id'], job=j, stage='FINAL', final=True), 0) for j in c['jobs']],
                  'score_final_', 7200, cpu=True)
            state.update(status='COMPLETE', ended=time.time())
        except BaseException as e:
            traceback.print_exc(); state.update(status='INCOMPLETE', reason=str(e), ended=time.time())
        finally:
            persist(); reporting.report(c, state)
            save(root/'public/RESOURCE_AMENDMENT.json', dict(
                 reason='User withdrew GPU 2; only GPUs 0 and 1 permitted',
                 allowed_physical_ids=[0, 1], interrupted_attempts=1,
                 maximum_authorized_restarts=1, original_T0=c['origin']['T0'],
                 original_budget_preserved=True, operational_code_sha=amendment['operational_code_sha']))
            with (root/'public/REPORT.md').open('a') as f:
                f.write('\nUser requested withdrawal of GPU 2 during execution. One interrupted '
                        'trajectory was archived and restarted from original initialization on 0/1. '
                        'Both attempt costs are retained; completed trajectories, frozen science '
                        'and the original T0/budget remain unchanged. No other retry is permitted.\n')


def main():
    retained.ID = retained.base.ID = 'R41_CONFLICT_PROJECTION'
    retained.Host = retained.base.Host = Host
    retained.public = reporting.public
    retained.base.filesystem_guard = retained.filesystem_guard
    retained.scoring.metrics = retained.hard_metrics
    if os.environ['RUN_MODE'] == 'watch':
        c = retained.base.config(); root = Path(c['output_root'])
        amendment = read(root/'GPU_RESOURCE_AMENDMENT.json')
        while any(alive(a) for a in amendment['previous_controllers']):
            if time.time() >= c['origin']['online_deadline_epoch']:
                raise TimeoutError('original deadline while awaiting retirement')
            time.sleep(10)
        retained.base.watch()
    elif os.environ['RUN_MODE'] == 'supervise':
        supervise()
    else:
        from .run import main as worker_main
        worker_main()


if __name__ == '__main__':
    main()
