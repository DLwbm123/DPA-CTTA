"""Metadata-only registration, immutable science and explicit execution opt-in."""
import hashlib,json,subprocess
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SCIENCE=ROOT/'configs/r1_science_v1.json'
DEFAULTS=ROOT/'configs/r1_execution.defaults.json'
BASELINE='0ef9d593a196e18f38295edc5b919372e297671b'
REF='dbff0d985c6c95345d9fb78f5b1daef57b392564'
GRATA='33ae20d664f305af34739ec54a5bec7da53ffa0b'
COUNTS=dict(REFUGE=400,ORIGA=650,REFUGE_Valid=800,Drishti_GS=101)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

FROZEN_SCIENCE_SHA256='e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2'

def science():
    if digest(SCIENCE)!=FROZEN_SCIENCE_SHA256:raise ValueError('science bytes differ from frozen implementation')
    return json.loads(SCIENCE.read_text())

def registration_digest(reg):return hashlib.sha256(json.dumps(reg,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def from_b3(path):
    # Never traverse identities/proxy/history or call historical file validators.
    old=json.loads(Path(path).read_text());f=old['tasks']['fundus']
    keys=('sample_id','group_id','domain','subset','image_path','mask_path','image_size','manifest_index','patient_linkage','video_linkage')
    reg=dict(checkpoint={k:f['checkpoint'][k] for k in ('path','sha256','bytes','mtime_ns')},target=[{k:r.get(k,'UNKNOWN') for k in keys} for r in f['target']],metadata_origin='B3_registered_Fundus',checkpoint_only=True)
    reg['historical_scalars']={str(o):dict(A=str(Path(old['p2_directory'])/f'fundus_{o}_A.jsonl') if o<2 else str(Path(path).parent/f'fundus_{o}_A.jsonl'),C0=str(Path(path).parent.parent/'b4_frozen_c_transfer_20260911/fundus_canonical_C0.jsonl')) for o in range(4)}
    for o in range(4):stream(reg,o)
    return reg


def stream(reg,order):
    if order not in range(4) or not reg['checkpoint_only']:raise ValueError('frozen registration')
    rs=reg['target'];cfg=science()
    if len(rs)!=1951 or len({r['group_id'] for r in rs})!=1951 or len({r['sample_id'] for r in rs})!=1951 or Counter(r['domain'] for r in rs)!=COUNTS:raise ValueError('target metadata coverage')
    if Counter(r['subset'] for r in rs)!=Counter(remaining_dev=1695,legacy_dev=128,p1_extension_dev=128):raise ValueError('subset counts')
    if any(r['image_size'] is None or len(r['image_size'])!=2 for r in rs):raise ValueError('grid metadata')
    for d in COUNTS:
        inds=[r['manifest_index'] for r in rs if r['domain']==d]
        if inds!=sorted(inds) or len(set(inds))!=len(inds):raise ValueError('within-domain manifest order')
    return [r for d in cfg['orders'][order] for r in rs if r['domain']==d]


def matrix():
    cfg=science();jobs=[]
    for o,domains in enumerate(cfg['orders']):
        for i,arm in enumerate(cfg['arms']):
            jobs.append(dict(job_id=f'o{o}a{i}',arm=arm,order=o,domains=domains,records=1951,forwards=1951*(11 if arm=='C_SENS' else 8),backwards=1951,adam=1951,periodic_resets=7 if arm=='C_PER256' else None,status='NOT_RUN',gpu=None))
    return dict(status='DRY_RUN_ONLY',science_sha256=digest(SCIENCE),jobs=jobs,formal_budget=cfg['formal_budget'],GPU_requests_issued=0,per_gpu_smoke=cfg['per_gpu_smoke_budget'],allowed_gpu_list=[],resource_plan=dict(max_workers=3,trajectory_hours_cap=2,total_gpu_hours_cap=24,wall_hours_cap=24,private_bytes_cap=2*1024**3,estimate='NOT_MEASURED: actual per-device smoke/memory readiness pending Stage II',reduced_concurrency_preserves_24_jobs=True))


def authorize(auth,reg,current_sha=None,science_sha=None):
    if auth.get('enabled') is not True:raise PermissionError('execution disabled; external review and new user authorization required')
    sha=current_sha or subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if len(sha)!=40 or auth.get('approved_code_sha')!=sha:raise PermissionError('approved full code SHA mismatch')
    if auth.get('approved_science_sha256')!=(science_sha or digest(SCIENCE)):raise PermissionError('approved science digest mismatch')
    if auth.get('approved_registration_digest')!=registration_digest(reg):raise PermissionError('approved registration mismatch')
    if not isinstance(auth.get('external_review_reference'),str) or not auth['external_review_reference'].strip():raise PermissionError('real external review reference required')
    ids=auth.get('allowed_physical_gpu_ids',[]);workers=auth.get('max_workers')
    if not ids or len(ids)>3 or len(set(ids))!=len(ids) or any(type(i) is not int or i<0 for i in ids):raise PermissionError('explicit unique GPU list, max three')
    if type(workers) is not int or not 1<=workers<=len(ids):raise PermissionError('worker count')
    if not isinstance(auth.get('background_allowed'),bool):raise PermissionError('background policy missing')
    return ids[:workers]
