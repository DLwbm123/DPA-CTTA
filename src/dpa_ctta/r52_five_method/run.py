"""Twelve new streams plus eighteen audited historical streams; no search/retry."""
import copy
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from .host import Host
from ..r36_incremental_modules import run as retained
from ..r20_model_only_search.report import aggregate, bootstrap_pair, csvout
from ..r20_model_only_search.score import summary
from ..r10_12h_core.run import read, save

ID = 'R52_FIVE_METHOD_COMPARISON'
METHODS = ('C', 'CW_LSO', 'G', 'VPTTA', 'MGIPT')


def validate_rows(rows, seeds, complete=True):
    groups=defaultdict(list)
    for r in rows: groups[r['condition'], r['seed'], r['order']].append(r)
    if complete:
        expected={(m,s,o) for m in METHODS for s in seeds for o in (0,1)}
        if set(groups)!=expected: raise ValueError('five-method matrix incomplete')
    reference={}
    for (method,seed,order),rs in groups.items():
        rs.sort(key=lambda r:r['visit'])
        identity=[(r['visit'],r['content'],r['domain'],r['subset'],r['role']) for r in rs]
        if len(rs)!=1951 or len({r['content'] for r in rs})!=1951 or [r['visit'] for r in rs]!=list(range(1,1952)):
            raise ValueError('trajectory coverage or identity')
        if order in reference and reference[order]!=identity: raise ValueError('unpaired streams')
        reference[order]=identity
        if sum(r['subset']=='remaining_dev' for r in rs)!=1695: raise ValueError('primary denominator')
    return len(groups)


def simultaneous_interval(rows, baseline, repeats=20000):
    """Bonferroni two-sided 98.75% content-cluster interval for four contrasts."""
    index={(r['condition'],r['seed'],r['order'],r['content']):r for r in rows}
    domains=defaultdict(list)
    contents=sorted({r['content'] for r in rows if r['condition']=='CW_LSO'})
    for content in contents:
        values=[]
        for seed in (20260907,17011,29009):
            for order in (0,1):
                a=index['CW_LSO',seed,order,content];b=index[baseline,seed,order,content]
                values.append(st.mean(100*(x['dice']-y['dice']) for x,y in zip(a['metrics'],b['metrics'])))
        domains[a['domain']].append(st.mean(values))
    rng=np.random.default_rng(20261016);draws=np.zeros(repeats)
    for values in domains.values():
        a=np.array(values)
        for offset in range(0,repeats,200):
            n=min(200,repeats-offset)
            draws[offset:offset+n]+=a[rng.integers(len(a),size=(n,len(a)))].mean(1)/len(domains)
    return dict(simultaneous_CI_low_pp=float(np.quantile(draws,.00625)),
                simultaneous_CI_high_pp=float(np.quantile(draws,.99375)),
                correction='Bonferroni four two-sided intervals',bootstrap_repeats=repeats)


def report(c,state):
    root=Path(c['output_root']);out=root/'public';rows=[]
    for binding in c['reused_jobs']:
        rows.extend(json.loads(line) for line in Path(binding['scalar_path']).read_text().splitlines())
    jobs=read(root/'FINAL_JOBS.json') if (root/'FINAL_JOBS.json').exists() else []
    for job in jobs:
        p=root/'scores/final'/(job['id']+'.private.jsonl')
        if (p.with_name(job['id']+'.complete.json')).exists():
            rows.extend(json.loads(line) for line in p.read_text().splitlines())
    complete=state['status']=='COMPLETE'
    trajectories=validate_rows(rows,c['seeds'],complete)
    main,domains=aggregate(rows);csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domains)
    allrows=[]
    for method in METHODS:
        for seed in c['seeds']:
            for order in (0,1):
                rs=[r for r in rows if (r['condition'],r['seed'],r['order'])==(method,seed,order)]
                if rs:allrows.append(dict(condition=method,seed=seed,order=order,observations=len(rs),**summary(rs)))
    for row in allrows:row.pop('cells')
    csvout(out/'ALL_1951_SECONDARY.csv',allrows)
    pairs=[];intervals=[];negative=[]
    if complete:
        primary=[dict(r,role='PRIMARY_ALL') for r in rows if r['subset']=='remaining_dev']
        for baseline in ('C','G','VPTTA','MGIPT'):
            for role,selected in [('PRIMARY_ALL',primary),('SEARCH',rows),('SEALED_REVIEW',rows)]:
                pair,ci,neg=bootstrap_pair(selected,'CW_LSO',baseline,c['seeds'],role=role)
                if role=='PRIMARY_ALL':
                    pair.update(simultaneous_interval(primary,baseline))
                    pair['practical_gain_pass']=(pair['delta_pp']>=.1 and min(pair['order_delta_pp'])>0
                        and pair['positive_trajectories']>=5 and pair['imageweighted_delta_pp']>=0
                        and pair['worst_seed_averaged_cell_pp']>=-2 and pair['simultaneous_CI_low_pp']>0)
                pairs.append(pair);intervals.extend(ci);negative.extend(neg)
    csvout(out/'PAIRED_SUMMARY.csv',pairs);csvout(out/'CONTENT_BOOTSTRAP_CI.csv',intervals);csvout(out/'NEGATIVE_CELLS.csv',negative)
    ledger=retained.base.ledger(c,state)
    csvout(out/'NEW_RUN_COST.csv',[dict(phase=a['phase'],status=a['status'],wall_seconds=a['wall_seconds'],**a['cost']) for a in ledger['attempts']])
    costs=list(c.get('historical_costs',[]))
    for attempt in ledger['attempts']:
        if attempt['phase'].startswith('online_') and attempt['status']=='COMPLETE':
            method=attempt['phase'].split('_s')[0].removeprefix('online_')
            profile=read(root/'profiles'/(method+'.json'))
            costs.append(dict(condition=method,job=attempt['phase'].removeprefix('online_'),
                origin='R52_NEW',arrivals=1951,seconds_per_image=attempt['wall_seconds']/1951,
                peak_reserved_bytes=profile['peak_reserved_bytes'],**attempt['cost']))
    csvout(out/'TRAJECTORY_COST.csv',costs)
    profile_rows=[dict(condition=p.stem,**read(p)) for p in (root/'profiles').glob('*.json')]
    csvout(out/'PROFILE_COST.csv',profile_rows)
    for name,value in [('RUN_STATE.json',state),('RESOURCE_LEDGER.json',ledger),('REUSE_AUDIT.json',read(root/'REUSE_AUDIT.json'))]:
        save(out/name,retained.public(value))
    save(out/'COMPLETION_AUDIT.json',dict(status=state['status'],planned_trajectories=30,
        trajectories=trajectories,reused_trajectories=18,new_trajectories=trajectories-18,scored_rows=len(rows),
        expected_rows=58530,online_target_label_reads=0,independent_generalization=False))
    verdict=dict(status='COMPLETE' if complete else 'NOT_EVALUABLE',
        all_four_pass=complete and all(p['practical_gain_pass'] for p in pairs if p['role']=='PRIMARY_ALL'),
        independent_validation=False,automatic_followup=False)
    save(out/'FINAL_DECISION.json',verdict)
    lines=['# R52 five-method unified development comparison','',f"Status: **{state['status']}**.",'',
        'Primary: remaining_dev 1695 images; equal domains and OD/OC, then equal three seeds and two orders. Every adaptation stream has 1951 arrivals.',
        'C and CW_LSO reuse R50; GraTa reuses R20. VPTTA and MGIPT are new. Historical cost is not a controlled speed comparison.',
        'MGIPT is the previously registered paper_equation implementation on the shared checkpoint, not a new claim of exact released-code reproduction.',
        'All subsets have historical development exposure. No independent generalization claim. No tuning, retry, additional seeds or next experiment.', '',
        '| Method | Primary macro Dice (%) |','|---|---:|']
    for method in METHODS:
        values=[r['macro_Dice_percent'] for r in main if r['condition']==method and r['role']=='PRIMARY_ALL']
        value=f'{st.mean(values):.6f}' if len(values)==6 else 'NA'
        lines.append(f'| {method} | {value} |')
    lines+=['',f'Decision: {json.dumps(verdict)}.',
        'Four primary contrasts report ordinary 95% and Bonferroni-adjusted 98.75% content-bootstrap intervals; all seed/order observations of each content move together. Patient dependence remains unknown.',
        'All four require >=0.1pp, positive both orders, >=5/6 positive trajectories, imageweighted >=0, worst seed-mean domain/channel/order cell >=-2pp, and adjusted lower bound >0.',
        'Only anonymous aggregates and own code/protocol are public; private images, masks, model weights, individual content records, predictions and third-party code remain private.']
    if state.get('reason'):lines+=['',retained.public(state['reason'])]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    (out/'PROTOCOL.md').write_text(Path(c['protocol_path']).read_text())


def main():
    retained.ID=ID;retained.Host=Host;retained.report=report
    retained.main()


if __name__=='__main__':main()
