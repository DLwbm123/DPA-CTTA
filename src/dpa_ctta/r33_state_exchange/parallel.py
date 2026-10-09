"""One authorized scheduling handoff; existing online workers keep running."""
import concurrent.futures
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from . import run

base, read, save = run.base, run.read, run.save


def process_state(identity):
    try:
        fields = Path(f'/proc/{identity["pid"]}/stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return fields[0] if fields[19] == str(identity['start_ticks']) else None


def alive(identity):
    return process_state(identity) not in (None, 'Z', 'X')


def retire_legacy(plan):
    # The original parent must reap its children to retain real exit codes.
    sup = plan['legacy_supervisor']
    if alive(sup):
        os.kill(sup['pid'], signal.SIGINT)
        os.kill(sup['pid'], signal.SIGCONT)
    until = time.monotonic() + 60
    while any(alive(plan[k]) for k in ('legacy_supervisor', 'legacy_watchdog')):
        if time.monotonic() > until:
            raise TimeoutError('legacy control retirement')
        time.sleep(.2)


def supervise(c, plan):
    root = Path(c['output_root'])
    state = dict(status='RUNNING_COUNTERFACTUALS', jobs={j['id']: 'RUNNING' for j in c['jobs']}, scheduling='SIX_WORKERS_TWO_PER_GPU')
    def persist():
        save(root/'RUN_STATE.json', state)
        base.ledger(c, state)
    try:
        if process_state(plan['legacy_supervisor']) not in ('T', 't'):
            raise RuntimeError('legacy scheduler must be stopped before admission')
        if [j['id'] for j in c['jobs'][3:]] != plan['new_job_ids']:
            raise ValueError('registered queue changed')
        persist()
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            futures = {pool.submit(base.run_task, c, 'online_'+j['id'], g, 22*3600, j): j
                       for j, g in zip(c['jobs'][3:], c['gpu_assignments'])}
            for f in concurrent.futures.as_completed(futures):
                state['jobs'][futures[f]['id']] = f.result()['status']
                persist()
        while any(alive(p) for p in plan['existing_workers']):
            if time.time() > c['origin']['online_deadline_epoch']:
                raise TimeoutError('original online deadline')
            time.sleep(2)
        retire_legacy(plan)
        save(root/'LEGACY_CONTROL_RETIREMENT.json', dict(at=time.time(), state=read(root/'RUN_STATE.json'), reason='authorized scheduling handoff; original parents reaped their workers'))
        for j in c['jobs']:
            phase = 'online_'+j['id']
            p = read(root/'processes'/f'{phase}.json')
            r = read(root/'attempts'/f'{phase}.0.json')
            if p['active'] or p['exit_code'] != 0 or r['status'] != 'COMPLETE':
                raise RuntimeError('formal worker failed or retirement incomplete: '+j['id'])
            state['jobs'][j['id']] = r['status']
        state['status'] = 'SCORING'; persist()
        result = base.run_task(c, 'score_all', None, 2*3600, {})
        if result['status'] != 'COMPLETE':
            raise RuntimeError('CPU scoring failed')
        state.update(status='COMPLETE', decision=result['result'], ended=time.time(), delivery='PENDING_GITHUB')
    except BaseException as e:
        state.update(status='BLOCKED', reason=str(e)); traceback.print_exc()
    finally:
        persist(); run.report.finish(c, state)


def main():
    base.ID = run.ID
    c = base.config(); root = Path(c['output_root'])
    plan = read(root/'PARALLEL_HANDOFF.json')
    mode = os.environ['RUN_MODE']
    if mode == 'worker':
        if not os.environ['RUN_PHASE'].startswith('score_'):
            if os.environ['RUN_PHASE'] not in ['online_'+j for j in plan['new_job_ids']]:
                raise ValueError('unregistered parallel worker')
            # A job lease prevents duplicate writers while permitting two jobs per GPU.
            original_lease = run.lease
            run.lease = lambda path, identity: original_lease(path/os.environ['RUN_PHASE'], identity)
        sys.exit(0 if run.worker() else 1)
    elif mode == 'supervise':
        supervise(c, plan)
    elif mode == 'watch':
        save(root/'PARALLEL_CONTROL_READY.json', dict(pid=os.getpid(), at=time.time()))
        until = time.monotonic()+60
        while not (root/'PARALLEL_RELEASE.json').exists():
            if time.monotonic()>until:
                raise TimeoutError('parallel admission release absent')
            time.sleep(.1)
        try:
            base.watch()
        finally:
            retire_legacy(plan)
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    main()
