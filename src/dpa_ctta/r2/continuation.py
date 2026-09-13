"""One explicitly authorized continuation of the September 13 IO incident.

The original failed run stays immutable. Complete trajectories keep their original
bindings; the interrupted trajectory restarts from the registered checkpoint.
"""
import json
from pathlib import Path
from ..r1.evidence import output_bytes
from .plan import authorize, allocation, binding, bound, matrix, science, SCIENCE_SHA, registration_digest

SOURCE_RUN = '8ce180e3db2249c19bba9d103ede9cb2'
SOURCE_CODE = '0d515328a6cc42d8e0c6a458265b41e41c454fa6'
PENDING = ('o2a4', 'o3a1', 'o3a2', 'o3a3', 'o3a4')
SMOKE = dict(forwards=112, backwards=14, base_adam=14, perturb=0, restore=0)


def read(path):
    return json.loads(Path(path).read_text())


def inspect_source(source, assets, ids, authorization):
    # ponytail: this IO incident only; unrelated failures require their own recovery review.
    source = Path(source).resolve()
    receipt = read(source/'receipt.json'); packet = read(source/'packet.private.json')
    identity = receipt['binding']; jobs = matrix()['jobs']
    if identity['run_id'] != SOURCE_RUN or identity['code_sha'] != SOURCE_CODE or 'continuation' in packet:
        raise ValueError('not the authorized original incident')
    if identity['science_sha256']!=SCIENCE_SHA or identity['registration_digest']!=registration_digest(assets['registration']) or Path(packet['out']).resolve()!=source:
        raise ValueError('original science/registration/output binding')
    if authorization.get('continuation_source_binding') != identity or authorization.get('continuation_policy') != 'preserve_completed_restart_failed_once':
        raise PermissionError('explicit continuation authorization required')
    bound(packet, identity)
    if authorize(packet['authorization'], assets['registration'], current_sha=SOURCE_CODE) != ids:
        raise ValueError('original authorized devices')
    if packet['assets'] != assets or packet['devices'] != receipt['devices'] or [d['index'] for d in receipt['devices']] != ids:
        raise ValueError('original assets/device binding')
    if any(v['jobs'] != jobs or v['schedule'] != allocation(jobs,len(ids)) for v in (packet,receipt)):
        raise ValueError('original frozen matrix')
    if receipt['formal_budget'] != science()['formal_budget'] or receipt['smoke_budget'] != science()['per_gpu_smoke_budget']:
        raise ValueError('original budgets')
    stopped = read(source/'dispatch.stopped.json'); process = read(source/'matrix.processes.json')
    for value in (stopped,process):bound(value,identity)
    if stopped['status'] != 'INCOMPLETE' or stopped['reason'] != 'worker nonzero exit' or process['status'] != 'INCOMPLETE' or process['unstarted_jobs'] != list(PENDING[1:]):
        raise ValueError('original stop evidence')
    entries = {(e['phase'],e['key']):e for e in process['processes']}
    completed = [j['job_id'] for j in jobs if j['job_id'] not in PENDING]
    expected = {('smoke','device'+str(i)) for i in range(len(ids))} | {('formal',k) for k in completed+['o2a4']}
    if set(entries) != expected or len(entries) != len(process['processes']) or process['exit_codes'] != [e['exit_code'] for e in process['processes']]:
        raise ValueError('original process coverage')
    started = read(source/'processes.started.json');bound(started,identity)
    if started['processes'] != [{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in process['processes']]:
        raise ValueError('original created processes')
    slots = {a['job_id']:a['worker'] for a in receipt['schedule']['assignments']}
    for (phase,key),entry in entries.items():
        job = next((j for j in jobs if j['job_id']==key),None)
        slot = slots[key] if job else int(key.removeprefix('device'))
        bound(entry,binding(receipt,slot,job))
        if entry['pid'] != entry['pgid'] or entry['pid'] <= 0 or entry['exit_code'] != (1 if key=='o2a4' else 0) or entry['status'] != ('INCOMPLETE' if key=='o2a4' else 'EXITED'):
            raise ValueError('original process exit')
        path = source/key
        name = 'smoke.completion.json' if phase=='smoke' else 'failure.json' if key=='o2a4' else 'completion.json'
        done = read(path/name);bound(done,binding(receipt,slot,job))
        if key=='o2a4':
            if done['status']!='INCOMPLETE' or done['records']!=576 or 'No such file or directory' not in done['reason'] or '/.write-' not in done['reason'] or (path/'completion.json').exists():
                raise ValueError('original temporary-file failure')
            if done['physical'] != dict(forwards=4608,backwards=576,base_adam=576,perturb=0,restore=0):
                raise ValueError('failed prefix physical budget')
            with (path/'records.jsonl').open() as log:records=sum(1 for line in log if line.strip())
            if records != 576:
                raise ValueError('failed prefix record coverage')
        else:
            counts=SMOKE if phase=='smoke' else dict(forwards=job['forwards'],backwards=job['backwards'],base_adam=job['adam'],perturb=0,restore=0)
            if list(path.glob('*failure.json')) or done['status'] != ('MECHANICAL_SMOKE_COMPLETE' if phase=='smoke' else 'TRAJECTORY_COMPLETE') or done['physical'] != counts:
                raise ValueError('original completion contradiction')
            if done['backend']['seed']!=20260907 or done['checkpoint_io']['bytes']!=assets['registration']['checkpoint']['bytes']:
                raise ValueError('original seed/checkpoint metadata')
            if job and (done['records']!=job['records'] or not (path/'records.jsonl').is_file()):
                raise ValueError('original complete records')
    if any((source/k).exists() for k in PENDING[1:]):raise ValueError('unstarted job has unexpected output')
    for key in ('active_seconds','wall_seconds'):
        if not 0<=process[key]<86400:raise ValueError('original runtime budget')
    info=dict(source_directory=str(source),source_binding=identity,completed_jobs=completed,pending_jobs=list(PENDING),
                discarded_prefix_records=576,discarded_prefix_physical=dict(forwards=4608,backwards=576,base_adam=576,perturb=0,restore=0),
                prior_smoke_physical={k:v*len(ids) for k,v in SMOKE.items()},prior_active_seconds=process['active_seconds'],
                prior_wall_seconds=process['wall_seconds'],source_bytes=output_bytes(source))
    if authorization.get('abandoned_continuation_directory'):
        include_attempt(info,assets,ids,authorization,'abandoned','a8693fe3eab44ddb802c6de6c0caafda','66eea7e880e16d4d25efa4edabc9d8ad59ff175d',())
    if authorization.get('partial_continuation_directory'):
        include_attempt(info,assets,ids,authorization,'partial','6bb3ccee255243a684c95975ba2671bc','1560f28d41dda5665404ececbefc4f0a7c69d192',('o2a4',))
    return info


def include_attempt(info,assets,ids,authorization,kind,expected_run,expected_code,completed):
    """Validate the two known failed continuations; carry only complete trajectories."""
    from .plan import stream
    from .analyze import validate
    out=Path(authorization[kind+'_continuation_directory']).resolve()
    packet=read(out/'packet.private.json');receipt=read(out/'receipt.json');identity=receipt['binding']
    if identity!=authorization.get(kind+'_continuation_binding') or identity['run_id']!=expected_run or identity['code_sha']!=expected_code:
        raise PermissionError('unapproved abandoned continuation')
    bound(packet,identity)
    if identity['science_sha256']!=SCIENCE_SHA or identity['registration_digest']!=info['source_binding']['registration_digest']:
        raise ValueError('abandoned science/registration')
    if authorize(packet['authorization'],assets['registration'],current_sha=expected_code)!=ids or packet['assets']!=assets or Path(packet['out']).resolve()!=out:
        raise ValueError('abandoned assets/authorization')
    original=read(Path(info['source_directory'])/'receipt.json');jobs=matrix()['jobs']
    if packet['devices']!=receipt['devices'] or [(d['index'],d['uuid']) for d in receipt['devices']]!=[(d['index'],d['uuid']) for d in original['devices']]:
        raise ValueError('abandoned devices')
    if any(v['jobs']!=jobs or v['schedule']!=allocation(jobs,len(ids)) or v['continuation']!=info for v in (packet,receipt)):
        raise ValueError('abandoned original continuation binding')
    process=read(out/'matrix.processes.json');stopped=read(out/'dispatch.stopped.json')
    for value in (process,stopped):bound(value,identity)
    reason_ok=stopped['reason']=='worker nonzero exit' if completed else 'FileNotFoundError' in stopped['reason'] and '/.nfs' in stopped['reason']
    if process['status']!='INCOMPLETE' or stopped['status']!='INCOMPLETE' or not reason_ok:
        raise ValueError('abandoned NFS stop evidence')
    attempted=('o2a4','o3a2');entries=process['processes']
    expected={('smoke','device'+str(i)) for i in range(len(ids))}|{('formal',k) for k in attempted}
    if len(entries)!=len(expected) or {(e['phase'],e['key']) for e in entries}!=expected or process['exit_codes']!=[e['exit_code'] for e in entries] or process['unstarted_jobs']!=[k for k in PENDING if k not in attempted]:
        raise ValueError('abandoned process coverage')
    started=read(out/'processes.started.json');bound(started,identity)
    if started['processes']!=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]:raise ValueError('abandoned started inventory')
    slots={a['job_id']:a['worker'] for a in packet['schedule']['assignments']};prefixes={}
    for e in entries:
        key=e['key'];job=next((j for j in jobs if j['job_id']==key),None)
        slot=slots[key] if job else int(key.removeprefix('device'));b=binding(receipt,slot,job);bound(e,b)
        success=job is None or key in completed
        if e['pid']!=e['pgid'] or e['pid']<=0 or e['exit_code']!=(0 if success else 1 if completed else -15) or e['status']!=('EXITED' if success else 'INCOMPLETE'):
            raise ValueError('abandoned owned exits')
        p=out/key
        if job:
            rows=[json.loads(line) for line in (p/'records.jsonl').read_text().splitlines() if line.strip()]
            if not 0<len(rows)<=job['records']:raise ValueError('prior trajectory coverage')
            validate(rows,stream(assets['registration'],job['order'])[:len(rows)],job['arm'],job['order'],b)
            if success:
                done=read(p/'completion.json');bound(done,b)
                if list(p.glob('*failure.json')) or done['status']!='TRAJECTORY_COMPLETE' or done['records']!=job['records'] or len(rows)!=job['records'] or done['physical']!=dict(forwards=job['forwards'],backwards=job['backwards'],base_adam=job['adam'],perturb=0,restore=0):raise ValueError('prior complete trajectory')
                if done['backend']['seed']!=20260907 or done['checkpoint_io']['bytes']!=assets['registration']['checkpoint']['bytes']:raise ValueError('prior complete checkpoint/seed')
            else:
                failure=read(p/'supervisor.failure.json');bound(failure,b)
                if failure['status']!='INCOMPLETE' or not failure['prefix_preserved'] or (p/'completion.json').exists() or len(rows)>=job['records']:raise ValueError('abandoned prefix failure')
                if completed:
                    failure=read(p/'failure.json');bound(failure,b)
                    if failure['records']!=len(rows) or failure['status']!='INCOMPLETE' or '/usage.json' not in failure['reason'] or 'No such file or directory' not in failure['reason'] or failure['physical']!=dict(forwards=len(rows)*8,backwards=len(rows),base_adam=len(rows),perturb=0,restore=0):raise ValueError('usage scan failure accounting')
                prefixes[key]=len(rows)
        else:
            done=read(p/'smoke.completion.json');bound(done,b)
            if list(p.glob('*failure.json')) or done['status']!='MECHANICAL_SMOKE_COMPLETE' or done['physical']!=SMOKE or done['backend']['seed']!=20260907 or done['checkpoint_io']['bytes']!=assets['registration']['checkpoint']['bytes']:
                raise ValueError('abandoned smoke')
    if any((out/k).exists() for k in [*info['completed_jobs'],*[k for k in PENDING if k not in attempted]]):raise ValueError('unexpected abandoned job')
    n=sum(prefixes.values());info['discarded_prefix_records']+=n
    for k,per_visit in dict(forwards=8,backwards=1,base_adam=1,perturb=0,restore=0).items():info['discarded_prefix_physical'][k]+=n*per_visit
    for k in SMOKE:info['prior_smoke_physical'][k]+=SMOKE[k]*len(ids)
    for key in ('active_seconds','wall_seconds'):
        if not 0<=process[key]<86400:raise ValueError('abandoned runtime budget')
        info['prior_'+key]+=process[key]
    info['source_bytes']+=output_bytes(out)
    info.update({kind+'_directory':str(out),kind+'_binding':identity,kind+'_prefixes':prefixes})
    if not completed:info['unrecorded_inflight_upper_bound']=dict(forwards=8*len(attempted),backwards=len(attempted),base_adam=len(attempted))
    for key in completed:
        info['completed_jobs'].append(key);info['pending_jobs'].remove(key)
        info.setdefault('carried_job_sources',{})[key]=str(out)


def source_for(receipt, out, job):
    continuation = receipt.get('continuation')
    if continuation and job['job_id'] in continuation['completed_jobs']:
        source = Path(continuation.get('carried_job_sources',{}).get(job['job_id'],continuation['source_directory']))
        return source/job['job_id'],read(source/'receipt.json')
    return Path(out)/job['job_id'],receipt


def public_accounting(continuation, physical, smoke):
    prefix = continuation['discarded_prefix_physical']; prior = continuation['prior_smoke_physical']
    result=dict(source_binding=continuation['source_binding'],carried_jobs=continuation['completed_jobs'],executed_jobs=continuation['pending_jobs'],
                discarded_prefix_records=continuation['discarded_prefix_records'],discarded_prefix_physical=prefix,prior_smoke_physical=prior,
                actual_total_physical={k:physical['adam' if k=='base_adam' else k]+prefix[k]+prior[k]+smoke[k] for k in ('forwards','backwards','base_adam')},
                actual_scoring_visits=physical['new_records']+continuation['discarded_prefix_records'],
                note='User-authorized IO continuation; prior failures preserved. Formal table excludes interrupted prefixes. Multiple runtime commits; repeated prefixes and fresh smoke exceed the original compute budget explicitly.')
    if 'abandoned_binding' in continuation:
        lower=result.pop('actual_total_physical');upper=continuation['unrecorded_inflight_upper_bound']
        result.update(abandoned_binding=continuation['abandoned_binding'],abandoned_prefixes=continuation['abandoned_prefixes'],actual_total_physical_lower_bound=lower,
                      actual_total_physical_upper_bound={k:v+upper[k] for k,v in lower.items()},unrecorded_inflight_upper_bound=upper)
        result['recorded_scoring_visits']=result.pop('actual_scoring_visits')
        result['note']+=' Terminated workers may each have one unrecorded in-flight visit; physical totals are bounded, not exact.'
    if 'partial_binding' in continuation:result.update(partial_binding=continuation['partial_binding'],partial_prefixes=continuation['partial_prefixes'],additional_carried_jobs=list(continuation['carried_job_sources']))
    return result
