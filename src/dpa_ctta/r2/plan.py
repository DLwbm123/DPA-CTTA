"""Frozen five-arm R2 matrix; registration body remains the R1 registration."""
import json
from ..r1.plan import ROOT, digest, registration_digest, binding, bound, stream, REF, GRATA
from ..r1.plan import authorize as r1_authorize

SCIENCE = ROOT/'configs/r2_science_v1.json'
DEFAULTS = ROOT/'configs/r2_execution.defaults.json'
SCIENCE_SHA = '882323714fc29b439bccb540cfe7e685733da1e7f0012af8a202954ec2e12353'
R1_CODE = '54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b'
R1_SCIENCE = 'e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2'


def science():
    if digest(SCIENCE) != SCIENCE_SHA: raise ValueError('frozen R2 science bytes changed')
    return json.loads(SCIENCE.read_text())


def allocation(jobs,workers):
    if type(workers) is not int or workers not in (1,2,3): raise ValueError('one to three slots')
    arms=[s['name'] for s in science()['arms']]
    assignments=[dict(job_id=j['job_id'],worker=(arms.index(j['arm'])+j['order'])%workers) for j in jobs]
    return dict(rule='(arm_index + order_index) % workers',assignments=assignments,
                loads=[dict(worker=i,jobs=sum(a['worker']==i for a in assignments),forwards=sum(j['forwards'] for j,a in zip(jobs,assignments) if a['worker']==i)) for i in range(workers)])


def matrix():
    c=science();jobs=[dict(job_id=f'o{o}a{i}',arm=s['name'],order=o,domains=ds,records=1951,forwards=15608,backwards=1951,adam=1951,status='NOT_RUN',gpu=None)
        for o,ds in enumerate(c['orders']) for i,s in enumerate(c['arms'])]
    return dict(status='DRY_RUN_ONLY',science_sha256=SCIENCE_SHA,jobs=jobs,formal_budget=c['formal_budget'],
                historical_primary_records=15608,per_gpu_smoke=c['per_gpu_smoke_budget'],
                rotation_examples={str(k):allocation(jobs,k) for k in (1,2,3)},GPU_requests_issued=0)


def authorize(auth,reg,current_sha=None):
    return r1_authorize(auth,reg,current_sha,science_sha=digest(SCIENCE))


def caps():
    c=science()['resource_caps']
    return dict(trajectory_seconds=c['trajectory_hours']*3600,wall_seconds=c['wall_hours']*3600,
                active_seconds=c['total_active_worker_hours']*3600,bytes=c['private_output_bytes'])
