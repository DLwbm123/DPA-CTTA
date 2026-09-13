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
    # ponytail: one incident only; a later failure needs its own explicit recovery review.
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
    return dict(source_directory=str(source),source_binding=identity,completed_jobs=completed,pending_jobs=list(PENDING),
                discarded_prefix_records=576,discarded_prefix_physical=dict(forwards=4608,backwards=576,base_adam=576,perturb=0,restore=0),
                prior_smoke_physical={k:v*len(ids) for k,v in SMOKE.items()},prior_active_seconds=process['active_seconds'],
                prior_wall_seconds=process['wall_seconds'],source_bytes=output_bytes(source))


def source_for(receipt, out, job):
    continuation = receipt.get('continuation')
    if continuation and job['job_id'] in continuation['completed_jobs']:
        source = Path(continuation['source_directory'])
        return source/job['job_id'],read(source/'receipt.json')
    return Path(out)/job['job_id'],receipt


def public_accounting(continuation, physical, smoke):
    prefix = continuation['discarded_prefix_physical']; prior = continuation['prior_smoke_physical']
    return dict(source_binding=continuation['source_binding'],carried_jobs=continuation['completed_jobs'],executed_jobs=continuation['pending_jobs'],
                discarded_prefix_records=continuation['discarded_prefix_records'],discarded_prefix_physical=prefix,prior_smoke_physical=prior,
                actual_total_physical={k:physical['adam' if k=='base_adam' else k]+prefix[k]+prior[k]+smoke[k] for k in ('forwards','backwards','base_adam')},
                actual_scoring_visits=physical['new_records']+continuation['discarded_prefix_records'],
                note='User-authorized IO continuation; original failure preserved. Formal table excludes the interrupted prefix. Two runtime commits; repeated prefix and fresh smoke exceed the original compute budget explicitly.')
