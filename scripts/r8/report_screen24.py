"""Read sealed private outputs; emit aggregate-only Screen24 results on stdout.

Run with ROOT RUN_ID SPEC_JSON arguments. Never emits content identifiers or paths.
No model execution, dataset reads, or target-based checkpoint selection.
"""
import collections
import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def macro(rows):
    cells = collections.defaultdict(list)
    for domain, channel, value in rows:
        cells[domain, channel].append(value)
    assert len(cells) == 8
    return st.mean(st.mean(v) for v in cells.values())


def describe(values):
    values = sorted(values)
    n = len(values)
    pos = (n - 1) * .05
    lo = math.floor(pos)
    return dict(n=n, mean=st.mean(values), median=st.median(values),
                negative=sum(v < 0 for v in values), zero=values.count(0),
                positive=sum(v > 0 for v in values),
                q05=values[lo] + (values[min(lo + 1, n - 1)]-values[lo])*(pos-lo),
                worst_decile_mean=st.mean(values[:max(1, math.ceil(n*.1))]))


def report(root, run, spec):
    source, target, work = (root / x / run for x in ('source', 'target', 'runs'))
    queue, ledger = read(work/'queue.json'), read(work/'ledger/state.json')
    assert queue['status'] == 'EXECUTION_COMPLETE' and ledger['stop'] is None
    assert len(queue['tasks']) == 96 and all(t['status']=='COMPLETE' for t in queue['tasks'].values())
    assert all(s=='COMPLETE' for s in queue['cpu'].values())
    for job in spec['source_jobs']:
        p = source / job['id']
        fit, done = read(p/'fit_complete.json'), read(p/'worker_complete.json')
        assert fit['steps'] == 4000 and done['schema']=='R8_SOURCE_JOB_WORK_COMPLETE_V1'
        assert done['fit_receipt_sha256'] == digest(p/'fit_complete.json')
    selection, index = read(source/'selection.json'), read(source/'source_index.json')
    expected = {k[:-7] for k in queue['tasks'] if k.endswith('_online')}
    assert len(expected) == 40 and expected == {p.name for p in target.iterdir() if p.is_dir()}
    by_method = collections.defaultdict(list)
    private = collections.defaultdict(dict)
    jobs, domains, coverage = [], [], collections.Counter()
    for job_id in sorted(expected):
        p = target/job_id
        done, sealed = read(p/'worker_complete.json'), read(p/'score_complete.json')
        job = done['job']
        assert done['score_receipt_sha256'] == digest(p/'score_complete.json')
        assert done['online_worker_sha256'] == digest(p/'online_worker_complete.json')
        assert sealed['scalar_sha256'] == digest(p/'scalars.private.jsonl')
        assert sealed['visits'] == 1951
        records = [json.loads(s) for s in (p/'scalars.private.jsonl').read_text().splitlines()]
        assert len(records) == 1951 and len({r['content'] for r in records}) == 1951
        assert [r['visit'] for r in records] == list(range(1,1952))
        rows = [r for r in records if r['subset']=='remaining_dev']
        assert len(rows)==1695
        coverage.update(arrivals=len(records), principal=len(rows))
        method = job['arm'] + ('_'+job['mode'] if job['mode'] else '')
        seed = job['source_seed'] or job['target_seed']
        cells = collections.defaultdict(list)
        for row in rows:
            assert {m['channel'] for m in row['metrics']} == {'OD','OC'}
            for m in row['metrics']:
                assert math.isfinite(m['dice']) and 0 <= m['dice'] <= 1
                key = (row['domain'], m['channel'])
                cells[key].append(m)
                private[(method,seed,job['order'])][(row['domain'],row['content'],m['channel'])] = m
        assert len(cells)==8
        vals = {k:st.mean(m['dice'] for m in v) for k,v in cells.items()}
        for (domain,channel), ms in sorted(cells.items()):
            assd = [m['assd'] for m in ms if m['assd'] is not None and math.isfinite(m['assd'])]
            domains.append(dict(job=job_id,method=method,seed=seed,order=job['order'],domain=domain,
                                channel=channel,n=len(ms),dice=vals[domain,channel],
                                assd=st.mean(assd) if assd else None,assd_valid=len(assd),
                                assd_undefined=len(ms)-len(assd)))
        row = dict(job=job_id, method=method, seed=seed, order=job['order'],
                   dice=st.mean(vals.values()),
                   OD=st.mean(v for (d,c),v in vals.items() if c=='OD'),
                   OC=st.mean(v for (d,c),v in vals.items() if c=='OC'),
                   worst_domain=min(st.mean(vals[d,c] for c in ('OD','OC')) for d in {k[0] for k in vals}))
        jobs.append(row)
        by_method[method].append(row)
    assert coverage == dict(arrivals=78040,principal=67800)
    main=[]
    for method, rows in sorted(by_method.items()):
        seed_scores={str(s):st.mean(r['dice'] for r in rows if r['seed']==s) for s in {r['seed'] for r in rows}}
        domain_order=collections.defaultdict(list)
        for r in domains:
            if r['method']==method:
                domain_order[r['domain'],r['order']].append(r['dice'])
        main.append(dict(method=method,jobs=len(rows),dice=st.mean(r['dice'] for r in rows),
                         OD=st.mean(r['OD'] for r in rows),OC=st.mean(r['OC'] for r in rows),
                         seed_scores=seed_scores,seed_sd=st.stdev(seed_scores.values()) if len(seed_scores)>1 else None,
                         worst_domain_order=min(st.mean(v) for v in domain_order.values())))
    comparisons=[]
    for full in ('A_FULL','B_FULL'):
        for control in (full.replace('FULL','STATIC'),'CURRENT_MLP','C0_CURRENT_STATS','VPTTA_NATIVE','C_CTTA_FIXED_LR','G_CTTA_RELEASE_TRANSFER'):
            # Native baseline seed numbers differ: pair by predeclared seed ordinal.
            fs=sorted({s for m,s,o in private if m==full})
            cs=sorted({s for m,s,o in private if m==control},key=str)
            deltas=[]; per_seed=[]; common_assd=[]; undefined=0; domain_order=collections.defaultdict(list)
            for i,seed in enumerate(fs):
                order_scores=[]
                for order in (0,1):
                    a=private[full,seed,order]; b=private[control,cs[min(i,len(cs)-1)],order]
                    assert a.keys()==b.keys()
                    changes=[]
                    for (d,content,c),ma in a.items():
                        mb=b[d,content,c]; v=ma['dice']-mb['dice']
                        changes.append((d,c,v));deltas.append(v);domain_order[d,order].append(v)
                        x,y=ma['assd'],mb['assd']
                        if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):common_assd.append(x-y)
                        else:undefined+=1
                    order_scores.append(macro(changes))
                per_seed.append(st.mean(order_scores))
            comparisons.append(dict(full=full,control=control,macro_delta=st.mean(per_seed),per_seed=per_seed,
                                    content_channel_delta=describe(deltas),
                                    worst_domain_order_delta=min(st.mean(v) for v in domain_order.values()),
                                    assd_common_valid=len(common_assd),assd_undefined=undefined,
                                    assd_pooled_delta=st.mean(common_assd) if common_assd else None))
    cost=dict(ledger['prior_cost']); current=collections.Counter(); failed=collections.Counter()
    for task in queue['tasks'].values():
        assert task['physical_gpu'] in (5,6,7)
        receipt=read(work/'ledger'/('attempt-'+task['attempts'][-1]+'.json'))
        assert receipt['status']=='COMPLETE' and receipt['error'] is None
    for a in ledger['attempts'].values():
        used=a.get('actual') or a['reserved']
        for k,v in used.items():cost[k]=cost.get(k,0)+v
        (current if a.get('actual') else failed).update(used)
    assert all(cost[k]<=ledger['caps'][k] for k in cost)
    capacity=read(source/'capacity_0.3/results.json')
    assert len(capacity)==128
    capacity_summary={}
    for space in ('A32','B64'):
        values=[r['result']['query_soft_Dice'] for r in capacity if r['item']['space']==space]
        assert len(values)==64
        capacity_summary[space]={k:st.mean(v[k] for v in values) for k in values[0]}
    last=max((target/j/'worker_complete.json').stat().st_mtime for j in expected)
    return dict(schema='R8_SCREEN24_PUBLIC_AGGREGATES_V1',run=run,
                scope='Reduced Screen24; not full R8 performance envelope',identity=queue['identity'],
                runtime_code_sha=queue['runtime_code_sha'],coverage=dict(coverage,source_jobs=10,target_jobs=40),
                completion_receipt_latest_utc=datetime.datetime.fromtimestamp(last,datetime.timezone.utc).isoformat(),
                source_selection=selection['per_config_mode'],mlp_selection=index['mlp_selection'],
                main=main,jobs=jobs,domains=domains,comparisons=comparisons,capacity=capacity_summary,
                cost=dict(charged_including_prior_and_failed_reservations=cost,
                          current_completed_actual=dict(current),current_failed_reservations=dict(failed),caps=ledger['caps']))


if __name__=='__main__':
    result=report(Path(sys.argv[1]),sys.argv[2],read(Path(sys.argv[3])))
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))
