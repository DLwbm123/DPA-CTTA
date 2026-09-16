"""Prompt-derived metadata; execution additionally requires the supplied raw proposal."""
import hashlib,json,math,re,subprocess
from ..r1.plan import ROOT,REF,GRATA,digest,registration_digest
from ..r3.plan import stream as old_stream,stream_summary as old_summary
from .loss import ARMS,PHYSICAL

BASE='e271098e2a12baa166fc7b77848af4a2300d9cc3'
SCIENCE=ROOT/'configs/r6_science_v1.json'
SCIENCE_SHA='2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff'
REGISTRATION='8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf'
RECURRENCE='cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db'
SMOKE=dict(recipe='R6_ALL_ARMS4_OLD_C4_V1',network_forwards=160,loss_backward_calls=20,adam_calls=20,jacobian_vjp_calls=0,seed=20260907,pixel_indices=[0,1,2,3],rtol=1e-4,atol=1e-5)
CAPS=dict(trajectory_seconds=21600,wall_seconds=86400,active_seconds=172800,bytes=8589934592)


def science():
    if not SCIENCE.is_file():raise FileNotFoundError('missing supplied R6_SCIENCE_PROPOSAL.json; do not reconstruct its bytes')
    if digest(SCIENCE)!=SCIENCE_SHA:raise ValueError('R6 original proposal byte digest')
    return json.loads(SCIENCE.read_text())


def stream(reg,order):
    if registration_digest(reg)!=REGISTRATION:raise ValueError('registration binding')
    return old_stream(reg,order)


def stream_summary(reg):
    if registration_digest(reg)!=REGISTRATION:raise ValueError('registration binding')
    s=old_summary(reg)
    if s['stream_digest']!=RECURRENCE:raise ValueError('recurrence binding')
    return s


def fingerprint():
    files=['src/dpa_ctta/'+p for p in ('b1_host.py','r1/host.py','source_pilot.py','source_io.py','hosts/vptta.py','integrations/ctta_suite.py','host_diagnostic.py','p1_analysis.py','p2_analysis.py','r5_update_acceptance/analyze.py','r1/assets.py','r1/evidence.py','r1/supervise.py','r1/plan.py','r3/plan.py','r3/execution.py','b3_runtime.py')]
    files += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'src/dpa_ctta/r6_regional_consistency').glob('*.py'))]
    return hashlib.sha256(json.dumps({p:digest(ROOT/p) for p in files},sort_keys=True).encode()).hexdigest()


def matrix(scope):
    if scope not in ('A','B_NEW','AB'):raise ValueError('explicit R6 scope')
    orders={'A':(0,1,4),'B_NEW':(2,3),'AB':range(5)}[scope]
    return [dict(job_id=f'o{o}a{i}',arm=a,order=o,records=1951,network_forwards=15608,loss_backward_calls=1951,adam_calls=1951,jacobian_vjp_upper=0,reused_from_A=scope=='AB' and o in (0,1,4),selection_exposure='A_development_stream' if o in (0,1,4) else 'B_new_order_same_contents') for o in orders for i,a in enumerate(ARMS)]


def allocation(jobs,workers):
    if type(workers) is not int or not 1<=workers<=3:raise ValueError('worker count')
    return dict(rule='job_index % workers',assignments=[dict(job_id=j['job_id'],worker=i%workers) for i,j in enumerate(jobs)])


def dry_run(reg=None):
    if reg is not None:
        for o in range(5):stream(reg,o)
    matrices={s:matrix(s) for s in ('A','B_NEW','AB')}
    budgets={s:dict(jobs=len(js),records=1951*len(js),**{k:v*1951*len(js) for k,v in PHYSICAL.items()}) for s,js in matrices.items()}
    present=SCIENCE.is_file()
    if present:science()
    return dict(status='REGISTERED_METADATA_DRY_RUN' if reg is not None else 'UNREGISTERED_METADATA_ONLY',
        source='provided prompt and plan; original proposal pending' if not present else 'provided prompt and byte-verified proposal',
        science_sha256=SCIENCE_SHA,science_bytes_verified=present,production_fingerprint=fingerprint(),matrices=matrices,budgets=budgets,
        stream=None if reg is None else stream_summary(reg),allocation_examples={s:{str(n):allocation(j,n) for n in (1,2,3)} for s,j in matrices.items()},
        proposed_caps=CAPS,approved_caps=None,devices=None,smoke_proposal=SMOKE,execution_started=False,external_review='NOT_RUN',R6A='NOT_RUN',R6B='NOT_RUN')


def authorize(auth,reg,current_sha=None):
    if auth.get('enabled') is not True:raise PermissionError('R6 execution disabled')
    scope=auth.get('scope')
    if scope not in ('A','B_NEW'):raise PermissionError('A or B_NEW only; AB is metadata')
    science()
    sha=current_sha or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    wanted=dict(approved_code_sha=sha,approved_science_sha256=SCIENCE_SHA,approved_registration_digest=registration_digest(reg),approved_stream_digest=stream_summary(reg)['stream_digest'],approved_production_fingerprint=fingerprint(),approved_trajectory_count=len(matrix(scope)))
    if not re.fullmatch('[0-9a-f]{40}',sha) or any(auth.get(k)!=v for k,v in wanted.items()):raise PermissionError('R6 exact binding')
    review=auth.get('external_review') or {};waiver=auth.get('user_waiver') or {}
    reviewed=review.get('status')=='PASS' and bool(review.get('reference'))
    waived=waiver.get('explicit') is True and bool(waiver.get('reference'))
    if not (reviewed or waived):raise PermissionError('new R6 external review or explicit waiver')
    for item,active in ((review,reviewed),(waiver,waived)):
        if active and (item.get('code_sha')!=sha or item.get('scope')!=scope):raise PermissionError('review or waiver revision/scope')
    ids=auth.get('allowed_physical_gpu_ids');workers=auth.get('max_workers')
    if not isinstance(ids,list) or not 1<=len(ids)<=3 or any(type(i) is not int or i<0 for i in ids) or len(set(ids))!=len(ids) or type(workers) is not int or not 1<=workers<=len(ids):raise PermissionError('explicit devices and workers')
    caps=auth.get('caps')
    if not isinstance(caps,dict) or set(caps)!=set(CAPS) or any(type(v) not in (int,float) or not math.isfinite(v) or v<=0 for v in caps.values()):raise PermissionError('approved finite caps required')
    if type(auth.get('background_allowed')) is not bool or auth.get('smoke_recipe')!=SMOKE:raise PermissionError('background policy / smoke recipe')
    if scope=='A' and auth.get('reuse_A') is not None:raise PermissionError('A must be fresh')
    if scope=='B_NEW' and (not auth.get('reuse_A') or not isinstance(auth.get('reuse_A_binding'),dict)):raise PermissionError('qualified original A binding required')
    return ids[:workers]
