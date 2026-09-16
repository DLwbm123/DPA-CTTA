# Verbatim replay()/join() excerpts from r5_update_acceptance/analyze.py at pinned SHA.
def replay(traces,ordered,job,identity,p_accept=None):
    if len(traces)!=len(ordered) or len(traces)!=job['records'] or len({r['group_id'] for r in traces})!=len(traces):raise ValueError('trace coverage/duplicates')
    history=deque(maxlen=128);rng=random.Random(20260908);committed=forced=eligible_count=restores=0
    for t,(row,entry) in enumerate(zip(traces,ordered),1):
        bound(row,identity)
        if any(row[k]!=entry[k] for k in ('group_id','sample_id','domain','subset')):raise ValueError('ordered trace identity')
        z=row['trace'];json.dumps(z,allow_nan=False)
        if z['visit']!=t or z['arm']!=job['arm'] or z['counts']!=PHYSICAL or z['source_unchanged'] is not True or z['transaction_complete'] is not True:raise ValueError('visit/physical/causality')
        for name in ('e_pre','e_trial'):
            x=z[name];n=x['count'];s=x['sse']
            if type(n) is not int or n!=524288 or not math.isfinite(s) or not 0<=s<=n or x['mean']!=s/n:raise ValueError('error SSE/count')
        if [r['region'] for r in z['regions']]!=['OD_fg','OD_bg','OC_fg','OC_bg']:raise ValueError('four regions')
        valid=[]
        for r in z['regions']:
            n,s=r['count'],r['sse']
            if type(n) is not int or not 0<=n<=262144 or not math.isfinite(s) or not 0<=s<=n or r['mean']!=(s/n if n else None):raise ValueError('regional SSE/count/null')
            if n:valid.append(s/n)
        if any(sum(r['count'] for r in z['regions'][i:i+2])>262144 for i in (0,2)):raise ValueError('region overlap')
        risk=max(valid) if valid else None
        values=sorted(history);h=(len(values)-1)*.90
        if not values:q90=None
        else:
            low=int(h);q90=values[low]+(h-low)*(values[min(low+1,len(values)-1)]-values[low])
        # Independent implementation, with the specified float64 arithmetic order.
        if z['r']!=risk or z['past_count']!=len(history) or z['q90_past']!=q90:raise ValueError('past window/quantile')
        reason='warmup' if t<=32 else 'empty_reliable_regions' if risk is None else 'insufficient_history' if len(history)<32 else 'eligible'
        eligible=reason=='eligible';shadow=not eligible or (z['e_trial']['mean']<=z['e_pre']['mean']+1e-12 and risk<=q90+1e-12)
        arm=job['arm'];draw=rng.random() if arm=='C_RANDOM' else None
        if arm=='C_RANDOM' and (type(p_accept) not in (int,float) or not 0<=p_accept<=1):raise ValueError('calibration missing')
        accept=True if arm in ('C','C_HALF') else shadow if arm=='C_VERIFY' else (not eligible or draw<p_accept)
        expected=dict(eligible=eligible,forced=not eligible,reason=reason,shadow_accept=shadow,accept=accept,random_draw=draw,p_accept=p_accept if arm=='C_RANDOM' else None)
        if any(z[k]!=v for k,v in expected.items()):raise ValueError('decision replay')
        if any(type(z[k]) is not bool for k in ('eligible','forced','shadow_accept','accept')):raise ValueError('boolean decision types')
        committed+=int(accept);forced+=int(not eligible);eligible_count+=int(eligible);restores+=int(not accept)
        totals=dict(n_visits=t,n_candidate_adam=t,n_committed=committed,n_rejected=t-committed,n_forced=forced,n_eligible=eligible_count,parameter_restorations=restores)
        if z['totals']!=totals or z['adam_committed_step']!=committed:raise ValueError('transaction counters')
        if risk is not None:history.append(risk)
        if z['buffer_count_after']!=len(history):raise ValueError('append timing')
    return totals


def join(traces,evaluations,ordered,job,identity,p_accept=None):
    replay(traces,ordered,job,identity,p_accept)
    if len(evaluations)!=len(traces):raise ValueError('evaluation coverage')
    result=[]
    for t,e in zip(traces,evaluations):
        bound(e,identity)
        if any(e[k]!=t[k] for k in ('sample_id','group_id','domain','subset','visit')) or e['visit']!=t['trace']['visit'] or e['evaluation']['transaction_before_GT'] is not True:raise ValueError('evaluation join/causality')
        metrics=e['evaluation']['metrics']
        if set(metrics)!={'pre','q','trial','emit'}:raise ValueError('prediction coverage')
        for name,ms in metrics.items():
            if [v['channel'] for v in ms]!=['OD','OC']:raise ValueError('channels')
            for m in ms:validate_metric(m)
            for m,p in zip(ms,metrics['pre']):
                if any(m[k]!=p[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) or m['total_pixels']!=262144:raise ValueError('GT identity')
        if metrics['emit']!=metrics['trial' if t['trace']['accept'] else 'pre']:raise ValueError('emitted branch metrics')
        json.dumps(e,allow_nan=False)
        result.append(dict(**t,metrics=metrics,L=[100*(q['dice']-p['dice']) for p,q in zip(metrics['pre'],metrics['trial'])]))
    return result
