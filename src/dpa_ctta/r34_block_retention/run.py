"""Six concurrent jobs with the existing worker/receipt/watchdog machinery."""
import concurrent.futures
import os
from pathlib import Path
import sys
import time
import traceback
from . import online, report
from ..r33_state_exchange import run as worker_runtime
from ..r10_12h_core.run import read, save

base=worker_runtime.base
ID='R34_BLOCK_RETENTION'


def admission(c,profiles):
    replay=max(sum(p['replay_seconds'])/len(p['replay_seconds']) for p in profiles)
    future=max(sum(p['future_seconds'])/len(p['future_seconds']) for p in profiles)
    duration=max(p['initialization_seconds'] for p in profiles)+1951*replay+3646*future
    finish=time.time()+duration/.8
    result=dict(admitted=finish<c['origin']['online_deadline_epoch'],GPU_cap_hours=None,concurrency=6,workers_per_GPU=2,
        projected_formal_GPU_seconds=6*duration,projected_online_end_with_reserve=finish,
        measured_replay_seconds_per_arrival=replay,measured_future_seconds_per_arrival=future,
        peak_reserved_bytes=max(p['peak_reserved_bytes'] for p in profiles))
    save(Path(c['output_root'])/'public/PROFILE_ADMISSION.json',result)
    return result['admitted']


def supervise():
    c=base.config();root=Path(c['output_root'])
    state=dict(status='PROFILING',jobs={j['id']:'NOT_RUN' for j in c['jobs']})
    def persist(): save(root/'RUN_STATE.json',state);base.ledger(c,state)
    def batch(jobs,prefix,timeout):
        results=[]
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            fs={pool.submit(base.run_task,c,prefix+j['id'],c['gpu_assignments'][i%3],timeout,j):j for i,j in enumerate(jobs)}
            for f in concurrent.futures.as_completed(fs):
                r=f.result();results.append(r)
                if prefix=='online_': state['jobs'][fs[f]['id']]=r['status'];persist()
        return results
    persist()
    try:
        if read(root/'MECHANICAL_TESTS.json')['status']!='PASS': raise ValueError('mechanical checks not passed')
        profiles=batch([dict(id=f'p{i}') for i in range(6)],'profile_',3600)
        if any(x['status']!='COMPLETE' for x in profiles): raise RuntimeError('profile failed; no automatic retry')
        save(root/'public/STATE_INVENTORY.json',read(root/'target/p0/inventory.json'))
        if not admission(c,[p['result'] for p in profiles]): raise RuntimeError('BUDGET_BLOCKED: entire matrix not admitted')
        state.update(status='RUNNING_COUNTERFACTUALS',jobs={j['id']:'RUNNING' for j in c['jobs']});persist()
        results=batch(c['jobs'],'online_',22*3600)
        if any(x['status']!='COMPLETE' for x in results): raise RuntimeError('formal job failed; no scientific retry or pruning')
        state['status']='SCORING';persist();result=base.run_task(c,'score_all',None,2*3600,{})
        if result['status']!='COMPLETE': raise RuntimeError('CPU scoring failed')
        state.update(status='COMPLETE',decision=result['result'],ended=time.time(),delivery='PENDING_GITHUB')
    except BaseException as e:
        state.update(status='BLOCKED',reason=str(e));traceback.print_exc()
    finally: persist();report.finish(c,state)


def main():
    base.ID=ID
    mode=os.environ['RUN_MODE']
    if mode=='worker':
        worker_runtime.ID=ID;worker_runtime.online=online;worker_runtime.report=report
        original=worker_runtime.lease
        worker_runtime.lease=lambda path,identity: original(path/os.environ['RUN_PHASE'],identity)
        sys.exit(0 if worker_runtime.worker() else 1)
    elif mode=='supervise': supervise()
    elif mode=='watch': base.watch()
    else: raise ValueError('unknown mode')


if __name__=='__main__': main()
