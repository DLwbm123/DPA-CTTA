"""Frozen science and metadata-only 85-trajectory plan; no asset decoding."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path
from ..r1.plan import ROOT,digest,registration_digest,stream as primary_stream
from .kernels import recurring_stream

BASE='f52f132e4be576ea871467432a6f4b12337209a5'
SCIENCE=ROOT/'configs/r3_science_v1.json'
SCIENCE_SHA='73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e'


def science():
    if digest(SCIENCE)!=SCIENCE_SHA:raise ValueError('frozen science bytes changed')
    return json.loads(SCIENCE.read_text())


def stream(reg,index):
    if type(index) is not int or index not in range(5):raise ValueError('stream index')
    if registration_digest(reg)!=science()['registration_digest']:raise ValueError('registered metadata binding')
    original=primary_stream(reg,0)
    if index<4:return primary_stream(reg,index)
    base=science()['secondary_stream']['base_order']
    mapping={d:[r['group_id'] for r in original if r['domain']==d] for d in base}
    sequence=recurring_stream(mapping,base,64);lookup={r['group_id']:r for r in original}
    return [lookup[i] for i in sequence]


def stream_summary(reg):
    cfg=science();rs=stream(reg,4);base=stream(reg,0)
    if len(rs)!=1951 or len({r['group_id'] for r in rs})!=1951:raise ValueError('coverage')
    for d in cfg['orders'][0]:
        if [r['group_id'] for r in rs if r['domain']==d]!=[r['group_id'] for r in base if r['domain']==d]:raise ValueError('within-domain order changed')
    payload=dict(schema='R3_STREAM_V1',registration_digest=registration_digest(reg),stream_id=4,
        rule=cfg['secondary_stream'],sequence=[r['group_id'] for r in rs])
    sha=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    blocks=[]
    for r in rs:
        if not blocks or blocks[-1]['domain']!=r['domain']:blocks.append(dict(domain=r['domain'],first_visit=sum(b['count'] for b in blocks)+1,count=0))
        blocks[-1]['count']+=1
    chunks=[];positions={d:0 for d in cfg['orders'][0]};round_index=0;cursor=1
    while any(positions[d]<sum(r['domain']==d for r in base) for d in positions):
        order=cfg['orders'][0];shift=round_index%4
        for d in order[shift:]+order[:shift]:
            count=min(64,sum(r['domain']==d for r in base)-positions[d])
            if count:
                chunks.append(dict(round=round_index,domain=d,within_domain_start=positions[d],first_visit=cursor,count=count))
                positions[d]+=count;cursor+=count
        round_index+=1
    if cursor!=1952:raise ValueError('chunk coverage')
    return dict(status='METADATA_VERIFIED_NO_PIXELS',registration_digest=registration_digest(reg),stream_digest=sha,
        digest_encoding='SHA256 canonical JSON: schema, registration_digest, stream_id, frozen rule, private ordered group_id sequence',
        records=len(rs),unique_contents=1951,duplicates=0,within_domain_order_unchanged=True,base_registration_unchanged=True,
        domain_counts=dict(Counter(r['domain'] for r in rs)),subset_counts=dict(Counter(r['subset'] for r in rs)),
        chunk_schedule=chunks,contiguous_domain_segments=blocks,primary_streams_unchanged=True,raw_image_replay=False)


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
                loss_backward_calls=c['groups'],adam_calls=c['groups'],jacobian_vjp_upper=c['groups']*a['jacobian_vjp_per_visit_upper'],
                actual_parameter_replacements_upper=c['groups'] if a['name'].startswith(('U_','S_')) else 0,
                worker_slot_1=0,worker_slot_2=(i+s)%2,worker_slot_3=(i+s)%3,status='NOT_RUN',gpu=None))
    budget=c['formal_budget']
    for key,b in [('records','scoring_records'),('network_forwards','network_forwards'),('loss_backward_calls','loss_backward_calls'),('adam_calls','adam_calls'),('jacobian_vjp_upper','jacobian_vjp_upper')]:
        if sum(j[key] for j in jobs)!=budget[b]:raise ValueError('budget contradiction')
    if len(jobs)!=85 or len({j['job_id'] for j in jobs})!=85:raise ValueError('matrix coverage')
    return dict(status='DRY_RUN_ONLY',base_commit=BASE,science_sha256=SCIENCE_SHA,jobs=jobs,formal_budget=budget,
        per_gpu_smoke=c['per_gpu_smoke'],rotation_examples={str(w):allocation(jobs,w) for w in (1,2,3)},
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
