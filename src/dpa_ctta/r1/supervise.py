"""Finite parent-owned process groups, one active trajectory per device slot."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from .evidence import write
from .plan import binding,science


class Interrupted(RuntimeError):
    pass


def limits():
    c=science()['runtime_proposed_caps']
    return dict(trajectory_seconds=c['trajectory_hours']*3600,wall_seconds=c['wall_hours']*3600,active_seconds=c['total_gpu_active_hours']*3600,bytes=c['private_output_bytes'])


def stop_owned(record,grace=.5):
    """Called once while this child/group is owned; never signal old PID history."""
    if record.get('cleaned'):return
    p=record['process'];pgid=p.pid
    try:
        os.killpg(pgid,signal.SIGTERM)
        end=time.monotonic()+grace
        while time.monotonic()<end:
            p.poll()
            try:os.killpg(pgid,0)
            except ProcessLookupError:break
            time.sleep(.02)
        try:os.killpg(pgid,signal.SIGKILL)
        except ProcessLookupError:pass
    except ProcessLookupError:pass
    try:p.wait(timeout=max(grace,.1))
    except subprocess.TimeoutExpired:raise RuntimeError('owned process did not reap after SIGKILL')
    record['cleaned']=True


def supervise(out,packet,start_process,caps=None,poll_seconds=.2):
    """start_process(spec, log) must return a child with start_new_session=True.

    Tests supply only short CPU children and small caps. No CLI changes these caps.
    A fresh process per formal trajectory eliminates unmanaged formal grandchildren.
    """
    out=Path(out);caps=limits() if caps is None else caps
    slots=len(packet['devices']);jobs={j['job_id']:j for j in packet['jobs']}
    queues={i:[jobs[a['job_id']] for a in packet['schedule']['assignments'] if a['worker']==i] for i in range(slots)}
    active={};history=[];stopped=False;phase='smoke';smoked=set();started=time.monotonic();last=started;seconds=0.;caught=None
    handlers={};pending_signal=None;stop_reason=None;failed_starts=[]

    def interrupt(signum,frame):
        nonlocal pending_signal
        # Raising inside an interrupted mkstemp/open can orphan a live fd on NFS.
        # Finish the current atomic IO/spawn, then unwind at a safe loop boundary.
        pending_signal=signum
    def halt(reason):
        nonlocal stopped,stop_reason
        if not stopped:stopped=True;stop_reason=reason
    def remember(exc):
        nonlocal caught
        if caught is None:caught=exc
        halt(type(exc).__name__+': '+str(exc))
    def failure(rec,reason,status='INCOMPLETE'):
        # Failure state and ownership never depend on persistence succeeding.
        rec['status']=status;rec['reason']=reason
    def write_failure(rec):
        p=out/rec['key'];p.mkdir(mode=0o700,exist_ok=True)
        path=p/'supervisor.failure.json'
        if not path.exists():write(path,dict(binding=rec['binding'],status=rec['status'],reason=rec['reason'],prefix_preserved=True))
    def finish(slot,reason=None,status='INCOMPLETE'):
        rec=active[slot];p=rec['process']
        code=p.poll()
        if reason is not None:failure(rec,reason,status)
        stop_owned(rec)
        active.pop(slot)
        rec.update(exit_code=p.returncode,ended_seconds=time.monotonic()-started)
        if reason is None:rec['status']='EXITED' if code==0 else 'INCOMPLETE'
        if reason is None and code==0 and rec['phase']=='smoke':smoked.add(slot)

    try:
        for sig in (signal.SIGINT,signal.SIGTERM):handlers[sig]=signal.signal(sig,interrupt)
        while True:
            if pending_signal is not None:raise Interrupted('parent signal '+str(pending_signal))
            now=time.monotonic();seconds+=(now-last)*len(active);last=now
            write(out/'usage.json',dict(seconds=seconds,wall_seconds=now-started,active_workers=len(active)),replace=True)
            if pending_signal is not None:raise Interrupted('parent signal '+str(pending_signal))
            # Poll every owned child before any new dispatch, independent of creation order.
            for slot,rec in list(active.items()):
                code=rec['process'].poll()
                if code is None:continue
                if code:
                    halt('worker nonzero exit')
                    p=out/rec['key'];scope=None
                    for name in ('failure.json','smoke.failure.json'):
                        if (p/name).is_file():scope=json.loads((p/name).read_text()).get('scope')
                    finish(slot,'worker nonzero exit '+str(code))
                    if scope in ('shared_assets','shared_code'):
                        for other in list(active):finish(other,'shared registered assets/code failure')
                        break
                else:finish(slot)
            if now-started>=caps['wall_seconds'] or seconds>=caps['active_seconds']:
                halt('matrix time cap')
                for slot in list(active):finish(slot,'matrix time cap','TIMEOUT')
            for slot,rec in list(active.items()):
                if rec['phase']=='formal' and now-rec['started']>=caps['trajectory_seconds']:
                    halt('trajectory time cap');finish(slot,'trajectory time cap','TIMEOUT')
            if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>caps['bytes']:
                halt('private output cap')
                for slot in list(active):finish(slot,'private output cap')
            if phase=='smoke' and len(smoked)==slots and not active and not stopped:phase='formal'
            if pending_signal is not None:raise Interrupted('parent signal '+str(pending_signal))
            if not stopped:
                for slot in range(slots):
                    if slot in active:continue
                    if any(r['process'].poll() not in (None,0) for r in active.values()):break
                    job=None if phase=='smoke' else (queues[slot].pop(0) if queues[slot] else None)
                    if phase=='smoke' and slot in smoked:continue
                    if phase=='formal' and job is None:continue
                    key='device'+str(slot) if job is None else job['job_id']
                    spec=dict(phase=phase,worker=slot,job=job,key=key,binding=binding(packet,slot,job))
                    try:
                        with (out/(key+'.log')).open('x') as log:
                            p=start_process(spec,log)
                            rec=dict(spec,process=p,pid=p.pid,pgid=p.pid,started=time.monotonic(),started_seconds=time.monotonic()-started,status='RUNNING')
                            history.append(rec);active[slot]=rec
                    except BaseException:
                        failure(spec,'process creation failed');failed_starts.append(spec)
                        raise
                    if pending_signal is not None:raise Interrupted('parent signal '+str(pending_signal))
                    write(out/'processes.started.json',dict(binding=binding(packet),processes=[{k:r[k] for k in ('pid','pgid','binding','phase','key')} for r in history]),replace=True)
            if not active and (stopped or phase=='formal' and not any(queues.values())):break
            time.sleep(poll_seconds)
    except BaseException as exc:
        remember(exc)
    finally:
        # No file IO, mkdir, exists or stat precedes this owned-group cleanup.
        # Attempt every group even if another group's cleanup raises.
        for sig in handlers:signal.signal(sig,signal.SIG_IGN)
        try:
            for slot in list(active):
                try:finish(slot,'parent interrupted or startup/supervision failure')
                except BaseException as exc:remember(exc)
            # All cleanup attempts precede best-effort evidence. Any IO failure
            # still fails the matrix; it is never swallowed to resume dispatch.
            for rec in history+failed_starts:
                if 'reason' in rec:
                    try:write_failure(rec)
                    except BaseException as exc:remember(exc)
            if stopped:
                try:write(out/'dispatch.stopped.json',dict(binding=binding(packet),status='INCOMPLETE',reason=stop_reason))
                except BaseException as exc:remember(exc)
            records=[{k:v for k,v in r.items() if k not in ('process','job','started','cleaned')} for r in history]
            complete=not stopped and caught is None and len(history)==slots+len(jobs) and all(r.get('exit_code')==0 for r in history)
            launched={r['key'] for r in history if r['phase']=='formal'}
            result=dict(binding=binding(packet),status='COMPUTE_COMPLETE' if complete else 'INCOMPLETE',processes=records,exit_codes=[r.get('exit_code') for r in history],active_seconds=seconds,wall_seconds=time.monotonic()-started,unstarted_jobs=[j for j in jobs if j not in launched])
            try:write(out/'matrix.processes.json',result,replace=True)
            except BaseException as exc:
                remember(exc);result['status']='INCOMPLETE'
        finally:
            for sig,handler in handlers.items():signal.signal(sig,handler)
    if caught is not None:
        try:print('supervision failed: '+type(caught).__name__+': '+str(caught),file=sys.stderr)
        except OSError:pass
        raise caught
    if not complete:raise RuntimeError('incomplete finite matrix; no retry')
    return result
