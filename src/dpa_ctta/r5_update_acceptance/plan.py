"""Frozen metadata, stage-specific scopes and future execution authorization."""
import hashlib,json,math,re,subprocess
from pathlib import Path
from ..r1.plan import ROOT,REF,GRATA,digest,registration_digest
from ..r3.plan import stream as old_stream,stream_summary as old_summary
from .rule import ARMS,PHYSICAL

SCIENCE=ROOT/'configs/r5_science_v1.json'
SCIENCE_SHA='89fa1492d196b232d16fe587d30effce9b70ab4a4a087d9d711b95f877f2b8f5'
BASE='b2bfce6cb29cea2df026194120f45b4f7252d53f'


def science():
    if digest(SCIENCE)!=SCIENCE_SHA:raise ValueError('frozen R5 science changed')
    return json.loads(SCIENCE.read_text())


def stream(reg,order):
    if registration_digest(reg)!=science()['registration_digest']:raise ValueError('registration binding')
    return old_stream(reg,order)


def stream_summary(reg):
    s=old_summary(reg)
    if s['stream_digest']!=science()['stream_digest']:raise ValueError('stream binding')
    return s


def fingerprint():
    # Binding reusable C runs includes the actual C path, observation and rule.
    files=['src/dpa_ctta/b1_host.py','src/dpa_ctta/r1/host.py','src/dpa_ctta/r5_update_acceptance/host.py','src/dpa_ctta/r5_update_acceptance/rule.py','src/dpa_ctta/r5_update_acceptance/evaluation.py','configs/r5_science_v1.json']
    return hashlib.sha256(json.dumps({p:digest(ROOT/p) for p in files},sort_keys=True).encode()).hexdigest()


def matrix(scope):
    if scope not in ('A','B_NEW','AB'):raise ValueError('explicit R5 scope')
    jobs=[]
    for order in range(5):
        for i,arm in enumerate(ARMS):
            reused=arm=='C' and order in (0,1,4)
            if scope=='A' and not reused or scope=='B_NEW' and reused:continue
            jobs.append(dict(job_id=f'o{order}a{i}',arm=arm,order=order,records=1951,network_forwards=15608,loss_backward_calls=1951,adam_calls=1951,jacobian_vjp_upper=0,reused_from_A=reused and scope=='AB',selection_exposure='A_diagnostic_stream' if order in (0,1,4) else 'B_new_order_same_contents'))
    b=science()['budgets'][scope]
    if len(jobs)!=b['jobs'] or sum(j['records'] for j in jobs)!=b['records'] or sum(j['network_forwards'] for j in jobs)!=b['network_forwards']:raise ValueError('budget')
    return jobs


def allocation(jobs,workers):
    if type(workers) is not int or not 1<=workers<=3:raise ValueError('worker count')
    return dict(rule='job_index % workers',assignments=[dict(job_id=j['job_id'],worker=i%workers) for i,j in enumerate(jobs)])


def dry_run(reg=None):
    if reg is not None:
        for o in range(5):stream(reg,o)
    return dict(status='METADATA_DRY_RUN',science_sha256=SCIENCE_SHA,c_fingerprint=fingerprint(),matrices={s:matrix(s) for s in ('A','B_NEW','AB')},budgets=science()['budgets'],stream=None if reg is None else stream_summary(reg),p_accept=None,execution_started=False,external_review='NOT_RUN',R5A='NOT_RUN',R5B='NOT_RUN')


def authorize(auth,reg,current_sha=None):
    if auth.get('enabled') is not True:raise PermissionError('R5 execution disabled')
    scope=auth.get('scope')
    if scope not in ('A','B_NEW'):raise PermissionError('A or B_NEW authorization required; no combined automatic stage')
    sha=current_sha or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    wanted=dict(approved_code_sha=sha,approved_science_sha256=SCIENCE_SHA,approved_registration_digest=registration_digest(reg),approved_stream_digest=stream_summary(reg)['stream_digest'],approved_c_fingerprint=fingerprint(),approved_trajectory_count=len(matrix(scope)))
    if not re.fullmatch('[0-9a-f]{40}',sha) or any(auth.get(k)!=v for k,v in wanted.items()):raise PermissionError('R5 exact binding')
    review=auth.get('external_review',{});waiver=auth.get('user_waiver',{})
    reviewed=review.get('status')=='PASS' and bool(review.get('reference'))
    waived=waiver.get('explicit') is True and bool(waiver.get('reference'))
    if not (reviewed or waived):raise PermissionError('new R5 external review or explicit user waiver required')
    for item,active in ((review,reviewed),(waiver,waived)):
        if active and (item.get('code_sha')!=sha or item.get('scope')!=scope):raise PermissionError('review/waiver is different revision or stage')
    ids=auth.get('allowed_physical_gpu_ids');workers=auth.get('max_workers')
    if not isinstance(ids,list) or not 1<=len(ids)<=3 or len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids) or type(workers) is not int or not 1<=workers<=len(ids):raise PermissionError('explicit devices and workers')
    if type(auth.get('background_allowed')) is not bool:raise PermissionError('background policy')
    caps=auth.get('caps')
    if not isinstance(caps,dict) or set(caps)!={'trajectory_seconds','wall_seconds','active_seconds','bytes'} or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in caps.values()):raise PermissionError('freeze resource caps before running')
    recipe=auth.get('smoke_recipe')
    if scope=='A':
        if recipe!=science()['A_smoke_proposal'] or auth.get('calibration') is not None or auth.get('reuse_A') is not None:raise PermissionError('A recipe/scope contamination')
    else:
        # B has its own reviewed recipe: four warm-up visits for C/HALF/RANDOM/VERIFY plus legacy C.
        wanted_recipe=dict(recipe='B_ALL_ARMS4_OLD_C4_V1',network_forwards=160,loss_backward_calls=20,adam_calls=20,jacobian_vjp_calls=0,seed=20260907,pixel_indices=[0,1,2,3],rtol=1e-4,atol=1e-5)
        if recipe!=wanted_recipe or not reviewed:raise PermissionError('separately reviewed B execution recipe required')
        if not auth.get('reuse_A') or not isinstance(auth.get('calibration'),dict):raise PermissionError('complete qualified A and label-free calibration required')
    return ids[:workers]
