"""One predetermined score-blind permutation of the frozen M2 query/transform rows."""
from collections import Counter
import hashlib

STAGES=[['clean','gamma_0.7','gamma_1.5','blur_5_sigma_1'],
 ['gamma_0.7','blur_5_sigma_1','clean','gamma_1.5'],
 ['gamma_1.5','clean','blur_5_sigma_1','gamma_0.7'],
 ['blur_5_sigma_1','gamma_1.5','gamma_0.7','clean'],
 ['clean','blur_5_sigma_1','gamma_1.5','gamma_0.7']]


def make_sequences(episodes,task):
    if task not in ('fundus','polyp'):raise ValueError('task')
    buckets={t:[] for t in STAGES[0]}
    for row in episodes:buckets[row['transform']].append(row)
    if len(episodes)!=600 or any(len(b)!=150 for b in buckets.values()):raise ValueError('M2 transform exposure')
    for b in buckets.values():b.sort(key=lambda r:hashlib.sha256(f"20260907/{task}/{r['episode']}".encode()).digest())
    rows=[]
    for stream,stages in enumerate(STAGES):
        for transform in stages:
            for _ in range(30):
                position=len(rows)%120
                used={r['query_index'] for r in rows[len(rows)-position%4:]}
                bucket=buckets[transform]
                i=next((i for i,r in enumerate(bucket) if r['query_index'] not in used),0)
                old=bucket.pop(i)
                rows.append(dict(visit=len(rows)+1,stream=stream,position=position+1,window=len(rows)//4+1,
                    original_episode=old['episode'],query_index=old['query_index'],transform=old['transform']))
    validate_sequences(episodes,rows)
    return rows


def validate_sequences(old,new):
    assert len(new)==600 and Counter(r['episode'] for r in old)==Counter(r['original_episode'] for r in new)
    assert Counter((r['query_index'],r['transform']) for r in old)==Counter((r['query_index'],r['transform']) for r in new)
    for i,r in enumerate(new):
        assert (r['visit'],r['stream'],r['position'],r['window'])==(i+1,i//120,i%120+1,i//4+1)
        assert r['transform']==STAGES[i//120][i%120//30]
    return True


def target_order(rows,order):
    if order==0:return list(rows)
    if order!=1:raise ValueError('registered orders only')
    domains=list(dict.fromkeys(r['domain'] for r in rows))
    return [r for d in reversed(domains) for r in rows if r['domain']==d]
