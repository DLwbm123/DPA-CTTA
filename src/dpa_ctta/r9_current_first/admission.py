"""Bind a proposed launch only after exact-code measured profiles and new authorization."""
import math
import re
from pathlib import Path
from .protocol import SPEC_SHA,CAPS,digest


def require_authorized(config):
    if config.get('execution_authorized') is not True:raise PermissionError('R9 execution is disabled; implementation request is not real-data/GPU authorization')
    if config.get('schema')!='R9_LAUNCH_V1' or config.get('spec_sha256')!=SPEC_SHA or config.get('caps')!=CAPS:
        raise ValueError('R9 frozen launch identity')
    if not re.fullmatch('[0-9a-f]{40}',config.get('code_sha','')):raise ValueError('exact R9 SHA required')
    gpu=config.get('gpu_assignments')
    if not isinstance(gpu,list) or not 1<=len(gpu)<=3 or len({x['physical_id'] for x in gpu})!=len(gpu):raise ValueError('new GPU assignments required')
    if any(type(x['physical_id']) is not int or x['physical_id']<0 or not x['uuid'].startswith('GPU-') for x in gpu):raise ValueError('GPU UUID binding')
    root=Path(config.get('output_root') or '')
    if not root.is_absolute() or 'r9' not in root.name.lower():raise ValueError('independent absolute R9 output root required')
    for key in ('bindings','source_inventory','profile','admission','authorization'):
        if not isinstance(config.get(key),dict) or not config[key]:raise ValueError('missing R9 '+key)
    if config.get('lr_source_policy') not in ('first_two_mean','first_only','per_seed'):raise ValueError('LR source policy unresolved')
    if config.get('score_release_policy')!='sealed_internal_score_release_at_end':raise ValueError('score/storage protocol unresolved')
    choices=dict(lr_source_policy=config['lr_source_policy'],score_release_policy=config['score_release_policy'])
    a=config['authorization']
    if a.get('protocol_choices')!=choices:raise PermissionError('protocol choices need explicit binding')
    if a.get('code_sha')!=config['code_sha'] or a.get('spec_sha256')!=SPEC_SHA or a.get('gpu_assignments')!=gpu or a.get('output_root')!=str(root) or not a.get('user_instruction'):
        raise PermissionError('new explicit execution authorization must bind this launch')
    if config['profile'].get('code_sha')!=config['code_sha'] or config['admission'].get('profile_sha256')!=digest(config['profile']):
        raise ValueError('profile must be measured on same exact runtime')
    if config['admission'].get('status')!='PASS':raise ValueError('full matrix resource admission not passed')
    from .profile import projection
    from .gradient_calibration import POLICIES
    from .protocol import graph
    profile=config['profile']
    if profile.get('bindings_sha256')!=digest(config['bindings']) or profile.get('protocol_choices')!=choices:raise ValueError('profile inputs/choices mismatch')
    if set(profile.get('node_budgets',{}))!={n['id'] for n in graph()['nodes']}:raise ValueError('profile missing task budgets')
    for row in profile['node_budgets'].values():
        if set(row)!=set(CAPS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in row.values()):raise ValueError('node budget')
    proof=projection(profile['measurements'],len(POLICIES[config['lr_source_policy']]),profile['retained_disk_bound'],profile['prior_cost'])
    if proof['status']!='PASS' or proof['upper_bound']!=config['admission']['upper_bound']:raise ValueError('full resource proof mismatch')
    totals=config['admission']['upper_bound']
    if set(totals)!=set(CAPS) or any(not math.isfinite(v) or v<0 or v>CAPS[k] for k,v in totals.items()):raise ValueError('aggregate cap')
    return root


def project(units,measurements,retained_disk,peak_temporary_disk,prior=None,recovery_factor=2.):
    """Worst-case all permitted single retries; no timing-only worker stop follows."""
    if recovery_factor!=2.:raise ValueError('one full failed reservation plus one recovery')
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in (retained_disk,peak_temporary_disk)):raise ValueError('disk bound')
    total=dict.fromkeys(CAPS,0);prior=prior or dict.fromkeys(CAPS,0)
    if set(prior)!=set(CAPS) or any(not math.isfinite(v) or v<0 for v in prior.values()):raise ValueError('prior costs')
    for name,count in units.items():
        row=measurements[name]
        if type(count) is not int or count<0 or not row.get('measured') or not row.get('evidence'):raise ValueError('measured profile units')
        if any(k not in row or type(row[k]) not in (int,float) or not math.isfinite(row[k]) or row[k]<0 for k in CAPS):raise ValueError('finite measured unit costs')
        for key in CAPS:
            if key!='disk_bytes':total[key]+=math.ceil(row[key]*count*recovery_factor)
    total['disk_bytes']=retained_disk+peak_temporary_disk
    total={k:total[k]+prior[k] for k in CAPS}
    return dict(status='PASS' if all(total[k]<=CAPS[k] for k in CAPS) else 'OVER_CAP',upper_bound=total)
