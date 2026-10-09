"""Resume one pre-compute admission failure after the original watchdog exits."""
import json
import os
from pathlib import Path
import time
import traceback


def admission_only(receipt):
    return (receipt['status'] == 'FAILED' and receipt['result'] is None
            and receipt['failure']['reason'] == 'insufficient free GPU memory for measured peak plus margin'
            and not any(receipt['cost'].get(k, 0) for k in ('model_forwards', 'backward_calls', 'optimizer_steps')))


def alive(identity):
    try:
        fields = Path(f'/proc/{identity["pid"]}/stat').read_text().split()
        return fields[21] == identity['start_ticks'] and fields[2] != 'Z'
    except FileNotFoundError:
        return False


def self_check():
    r = dict(status='FAILED', result=None, failure=dict(reason='insufficient free GPU memory for measured peak plus margin'), cost={})
    assert admission_only(r)
    assert not admission_only(dict(r, result={}))
    assert not admission_only(dict(r, status='COMPLETE'))
    for key in ('model_forwards', 'backward_calls', 'optimizer_steps'):
        assert not admission_only(dict(r, cost={key: 1}))
    assert not admission_only(dict(r, failure=dict(reason='replay mismatch')))
    assert not alive(dict(pid=os.getpid(), start_ticks='invalid'))


def main():
    from dpa_ctta.r32_history_origin import run, report
    from dpa_ctta.r10_12h_core.run import read, save
    base = run.base
    base.ID = run.ID
    c = base.config()
    root = Path(c['output_root'])
    request = read(root/'private/RECOVERY_REQUEST.json')
    phase = request['phase']
    state = None
    try:
        assert admission_only(read(root/'attempts'/f'{phase}.0.json'))
        # The original watchdog kills registered active workers on exit.
        while any(alive(p) for p in request['wait_for']):
            if time.time() >= c['origin']['online_deadline_epoch']:
                raise TimeoutError('original online deadline while awaiting retirement')
            save(root/'RECOVERY_STATE.json', dict(status='WAITING_ORIGINAL_RETIREMENT', at=time.time()))
            time.sleep(30)
        state = read(root/'RUN_STATE.json')
        for job in c['jobs']:
            p = read(root/'processes'/f'online_{job["id"]}.json')
            assert not p['active'] and not alive(p)
            if 'online_'+job['id'] != phase:
                assert p['exit_code'] == 0 and state['jobs'][job['id']] == 'COMPLETE'
        failed = read(root/'processes'/f'{phase}.json')
        save(root/'private/FAILED_PROCESS_ATTEMPT0.json', failed)
        assert not (root/'target'/request['job']['id']).exists()
        # Binding/determinism checks belong to the GPU worker spawned by run_task.
        base.available_memory(request['assignment'], 8*1024**3)
        os.environ['ENTRY_MODULE'] = 'dpa_ctta.r32_history_origin.run'
        state.update(status='RUNNING_COUNTERFACTUALS')
        state.pop('reason', None)
        state['jobs'][request['job']['id']] = 'RUNNING'
        save(root/'RUN_STATE.json', state)
        save(root/'RECOVERY_STATE.json', dict(status='RETRYING_ADMISSION_ONLY', at=time.time()))
        result = base.run_task(c, phase, request['assignment'], 22*3600, request['job'], attempt=1)
        state['jobs'][request['job']['id']] = result['status']
        if result['status'] != 'COMPLETE':
            raise RuntimeError('recovery attempt failed; retained for diagnosis')
        state['status'] = 'SCORING'
        save(root/'RUN_STATE.json', state)
        result = base.run_task(c, 'score_all', None, 2*3600, {})
        if result['status'] != 'COMPLETE':
            raise RuntimeError('CPU scoring failed')
        state.update(status='COMPLETE', decision=result['result'], ended=time.time(), delivery='PENDING_GITHUB')
    except BaseException as e:
        traceback.print_exc()
        save(root/'RECOVERY_STATE.json', dict(status='BLOCKED', reason=str(e), at=time.time()))
        if state is not None:
            state.update(status='BLOCKED', reason=str(e))
    else:
        save(root/'RECOVERY_STATE.json', dict(status='COMPLETE', at=time.time()))
    finally:
        if state is not None:
            save(root/'RUN_STATE.json', state)
            base.ledger(c, state)
            report.finish(c, state)


if __name__ == '__main__':
    main()
