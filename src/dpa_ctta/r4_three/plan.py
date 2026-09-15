"""Frozen R4T science and 70-trajectory metadata plan; no pixel decoding."""
import csv,json
from pathlib import Path
from ..r1.plan import ROOT,digest,registration_digest
from ..r3.plan import stream as inherited_stream,stream_summary as inherited_summary

BASE='2f90a6a0933cec3a25337a0d46ee632f8d772840'
SCIENCE=ROOT/'configs/r4t_science_v1.json'
SCIENCE_SHA='3b4db63b60cb483c91329c87cafd91931f316b4cb94a1d5aeb6c08c236a64096'


def science():
    if digest(SCIENCE)!=SCIENCE_SHA:raise ValueError('frozen science bytes changed')
    return json.loads(SCIENCE.read_text())


def stream(reg,index):
    if registration_digest(reg)!=science()['registration_digest']:raise ValueError('registration binding')
    return inherited_stream(reg,index)


def stream_summary(reg):
    result=inherited_summary(reg)
    if result['stream_digest']!=science()['secondary_stream_digest']:raise ValueError('recurrence changed')
    return result


def allocation(jobs,workers):
    if type(workers) is not int or workers not in (1,2,3):raise ValueError('worker slots')
    arms=[a['name'] for a in science()['arms']]
    rows=[dict(job_id=j['job_id'],worker=(arms.index(j['arm'])+j['order'])%workers) for j in jobs]
    return dict(rule='(arm_index + stream_id) % workers',assignments=rows,
        loads=[dict(worker=w,jobs=sum(r['worker']==w for r in rows),network_forwards=sum(j['network_forwards'] for j,r in zip(jobs,rows) if r['worker']==w)) for w in range(workers)])


def matrix(reg=None):
    c=science();jobs=[]
    for s in range(5):
        for i,a in enumerate(c['arms']):
            jobs.append(dict(job_id=f'o{s}a{i}',arm=a['name'],order=s,stream=s,endpoint='primary' if s<4 else 'secondary',
                records=c['groups'],network_forwards=c['groups']*a['forwards_per_visit'],
                loss_backward_calls=c['groups'],adam_calls=c['groups'],jacobian_vjp_upper=0,
                actual_parameter_replacements_upper=0,
                worker_slot_1=0,worker_slot_2=(i+s)%2,worker_slot_3=(i+s)%3,status='NOT_RUN',gpu=None))
    budget=c['formal_budget']
    for key,b in [('records','scoring_records'),('network_forwards','network_forwards'),('loss_backward_calls','loss_backward_calls'),('adam_calls','adam_calls'),('jacobian_vjp_upper','jacobian_vjp_calls')]:
        if sum(j[key] for j in jobs)!=budget[b]:raise ValueError('budget contradiction')
    if len(jobs)!=70 or len({j['job_id'] for j in jobs})!=70:raise ValueError('matrix coverage')
    return dict(status='DRY_RUN_ONLY',base_commit=BASE,science_sha256=SCIENCE_SHA,jobs=jobs,formal_budget=budget,
        per_gpu_smoke=c['per_actual_gpu_smoke'],rotation_examples={str(w):allocation(jobs,w) for w in (1,2,3)},
        GPU_requests_issued=0,background_tasks_started=0,real_target_reads=0,
        secondary_stream=stream_summary(reg) if reg is not None else dict(status='REGISTRATION_METADATA_REQUIRED'))


def write_plan(destination,reg):
    p=Path(destination);p.mkdir(parents=True,exist_ok=True);m=matrix(reg)
    for name,value in [('DRY_RUN_MATRIX.json',m),('STREAM_SUMMARY.json',m['secondary_stream'])]:
        (p/name).write_text(json.dumps(value,indent=2)+'\n')
    rows=m['jobs']
    with (p/'DRY_RUN_MATRIX.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return m
