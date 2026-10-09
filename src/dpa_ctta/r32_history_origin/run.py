"""Fixed counterfactual matrix; all online jobs retire before scores."""
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import torch
from . import online, report
from ..r20_model_only_search import runtime as base
from ..r10_12h_core.run import read, save
from ..r9_current_first.storage import lease

ID = 'R32_HISTORY_ORIGIN'


def worker():
    c = base.config(); root = Path(c['output_root']); phase = os.environ['RUN_PHASE']; start = float(os.environ['RUN_STARTED'])
    guard = base.Guard(c, phase); cpu = phase.startswith('score_'); result = None; failure = None; cost = {}
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    try:
        if cpu:
            result = report.score(c, guard)
        else:
            assignment = json.loads(os.environ['RUN_ASSIGNMENT'])
            base.gpu_policy(assignment); base.available_memory(assignment, 8*1024**3)
            if subprocess.check_output(['findmnt', '-n', '-o', 'FSTYPE', '-T', str(root)], text=True).strip() not in ('nfs', 'nfs4'):
                raise OSError('required network mount absent')
            if shutil.disk_usage(root).free < 64*1024**3:
                raise OSError('storage reserve')
            permit = online.access_guard(c); job = json.loads(os.environ['RUN_JOB'])
            with lease(root/'leases'/f'gpu{assignment["physical_id"]}', dict(experiment_id=ID, GPU=assignment)), base.Meter(None, guard.observe) as meter:
                guard.meter = meter
                if phase.startswith('profile_'):
                    result = online.profile(c, job, guard, permit)
                else:
                    result = online.stream(c, job, guard, permit)

                cost = meter.cost.copy()
    except BaseException as e:
        failure = dict(type=type(e).__name__, reason=str(e)); traceback.print_exc()
        if guard.meter:
            cost = guard.meter.cost.copy()
    cost.update(gpu_seconds=0 if cpu else time.time()-start, cpu_seconds=time.time()-start if cpu else 0)
    attempt = int(os.environ.get('RUN_ATTEMPT', 0))
    receipt = dict(status='COMPLETE' if failure is None else 'FAILED', phase=phase, attempt=attempt, started=start,
                   ended=time.time(), cost=cost, result=result, failure=failure, code_sha=c['code_sha'])
    save(root/'attempts'/f'{phase}.{attempt}.json', receipt)
    return failure is None


def admission(c, profiles):
    root=Path(c['output_root']);plans=read(root/'private/PLAN.json')
    replay=max(sum(p['replay']['seconds'])/len(p['replay']['seconds']) for p in profiles)
    history=max(sum(p['history_seconds'])/len(p['history_seconds']) for p in profiles)
    query={rule:max(sum(b['seconds'])/len(b['seconds']) for p in profiles for b in p['branches'] if b['rule']==rule) for rule in online.RULES}
    initialization=max(p['initialization_seconds'] for p in profiles)
    durations=[initialization+1951*replay+256*history+sum(len(case['query']) for case in plans[str(j['order'])])*4*sum(query.values()) for j in c['jobs']]
    finish=time.time()+sum(max(durations[k:k+3]) for k in range(0,len(durations),3))/.8
    result=dict(admitted=finish<c['origin']['online_deadline_epoch'],GPU_cap_hours=None,projected_formal_GPU_seconds=sum(durations),
                projected_online_end_with_reserve=finish,measured_replay_seconds_per_arrival=replay,measured_history_seconds_per_update=history,
                measured_query_seconds_per_arrival=query,peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in profiles))
    save(root/'public/PROFILE_ADMISSION.json',result)
    return result['admitted']


def supervise():
    c = base.config(); root = Path(c['output_root'])
    state = dict(status='PROFILING', jobs={j['id']: 'NOT_RUN' for j in c['jobs']})
    def persist():
        save(root/'RUN_STATE.json', state); base.ledger(c, state)
    def batch(jobs, prefix, timeout):
        results = []
        with concurrent.futures.ThreadPoolExecutor(3) as pool:
            ff = {pool.submit(base.run_task, c, prefix+j['id'], g, timeout, j): j for j, g in zip(jobs, c['gpu_assignments'])}
            for f in concurrent.futures.as_completed(ff):
                r = f.result(); results.append(r)
                if prefix == 'online_':
                    state['jobs'][ff[f]['id']] = r['status']; persist()
        return results
    persist()
    try:
        if read(root/'MECHANICAL_TESTS.json')['status'] != 'PASS':
            raise ValueError('mechanical tests not passed')
        profiles = batch([dict(id=f'p{i}') for i in range(3)], 'profile_', 3600)
        if any(r['status'] != 'COMPLETE' for r in profiles):
            raise RuntimeError('profile failed; no automatic retry')
        save(root/'public/STATE_INVENTORY.json', read(root/'target/p0/inventory.json'))
        if not admission(c, [p['result'] for p in profiles]):
            raise RuntimeError('BUDGET_BLOCKED: complete registered matrix cannot be admitted with reserve')
        save(root/'public/FROZEN.json',dict(jobs=6,branches=96,histories=list(online.HISTORIES),rules=list(online.RULES),history_length=64,seeds=c['seeds'],orders=[0,1],all_online_before_scores=True,no_performance_pruning=True))
        for stage,jobs in [('COUNTERFACTUALS',c['jobs'])]:
            state['status'] = 'RUNNING_'+stage; persist()
            for offset in range(0, len(jobs), 3):
                if time.time() >= c['origin']['online_deadline_epoch']:
                    raise TimeoutError('online engineering deadline')
                group = jobs[offset:offset+3]
                for j in group:
                    state['jobs'][j['id']] = 'RUNNING'
                persist(); results = batch(group, 'online_', 22*3600)
                if any(r['status'] != 'COMPLETE' for r in results):
                    raise RuntimeError('formal job failed; no scientific retry or arm removal')
        state['status'] = 'SCORING'; persist()
        result = base.run_task(c, 'score_all', None, 2*3600, {})
        if result['status'] != 'COMPLETE':
            raise RuntimeError('CPU scoring failed')
        state.update(status='COMPLETE', decision=result['result'], ended=time.time(), delivery='PENDING_GITHUB')
    except BaseException as e:
        state.update(status='BLOCKED', reason=str(e)); traceback.print_exc()
    finally:
        persist(); report.finish(c, state)


def main():
    base.ID = ID
    mode = os.environ['RUN_MODE']
    if mode == 'worker':
        sys.exit(0 if worker() else 1)
    elif mode == 'supervise':
        supervise()
    elif mode == 'watch':
        base.watch()
    else:
        raise ValueError('unknown mode')


if __name__ == '__main__':
    main()
