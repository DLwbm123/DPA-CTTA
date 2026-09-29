"""Real production units, complete R10 graph, finite per-resource recovery reserve."""
import math
from .protocol import CAPS,PROFILE_CAPS,SPEC,graph,digest


def node_units(node):
    kind=node['kind'];u={}
    if kind=='train':
        j=node['job'];method='WARM' if j['kind']=='warmup' else j['method']
        u['setup/source']=1;u['train/'+method]=2000 if method=='WARM' else 4000
        u['source_checkpoint']=42 if method=='WARM' else 86
        u['source_log']=2000 if method=='WARM' else 4000
        if method!='WARM':u['validation_episode']=4*64;u['gap_episode']=16*5
        u['source_finalize']=1
    elif kind=='d0':u={'setup/source':1,'d0_context':64}
    elif kind=='online':
        arm=node['job']['arm'];n=node['job']['arrivals']
        if arm not in ('N_SOURCE_EVAL','C0','VPTTA_NATIVE','C_CTTA','G_CTTA','B_CARRIER_FULL','B_CARRIER_RESET','B_CARRIER_STATIC'):arm='POLICY'
        u={'setup/'+arm:1,'online/'+arm:n,'target_checkpoint/'+arm:1+n//50,'target_append':n}
    elif kind=='score':u={'score_visit':node['job']['arrivals'],'score_seal':1}
    else:u['control/'+kind]=1
    return u


def units():
    from collections import Counter
    c=Counter()
    for n in graph():c.update(node_units(n))
    return dict(c)


def projection(profile):
    rows=profile['measurements'];nodes=graph();budgets=profile['node_budgets']
    if set(rows)!=set(units()) or set(budgets)!={n['id'] for n in nodes}:raise ValueError('complete R10 measurement/node coverage')
    total={k:profile['prior_cost'][k] for k in CAPS};recovery=[];retained=0
    for n in nodes:
        b=budgets[n['id']]
        if set(b)!=set(CAPS) or any(not isinstance(v,(int,float)) or not math.isfinite(v) or v<0 for v in b.values()):raise ValueError('node budget')
        need=dict.fromkeys(CAPS,0)
        for unit,count in node_units(n).items():
            row=rows[unit]
            if not row.get('measured') or not row.get('evidence'):raise ValueError('real measured profile required')
            for k in CAPS:
                if not math.isfinite(row[k]) or row[k]<0:raise ValueError('measurement cost')
                if k!='disk_bytes':need[k]+=math.ceil(row[k]*count)
        if any(b[k]<need[k] for k in CAPS):raise ValueError('budget below production measurement')
        keep=b['disk_bytes']-(n['job']['arrivals']*2*512*512*4 if n['kind']=='online' else 0)
        if keep<0:raise ValueError('probability disk budget')
        retained+=keep;recovery.append(dict(b,disk_bytes=keep))
        for k in CAPS:
            if k!='disk_bytes':total[k]+=b[k]
    reserve={k:sum(sorted((r[k] for r in recovery),reverse=True)[:3]) for k in CAPS}
    total['disk_bytes']+=retained+19510*2*512*512*4
    total={k:v+reserve[k] for k,v in total.items()}
    return dict(status='PASS' if all(total[k]<=CAPS[k] for k in CAPS) else 'OVER_CAP',upper_bound=total,recovery_reserve=reserve,profile_sha256=digest(profile))
