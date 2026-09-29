"""End-of-round anonymous aggregates; never expose a live target leaderboard."""
import hashlib,json,statistics as st
from collections import defaultdict
from pathlib import Path
from .protocol import SPEC,digest
from .ledger import Ledger
from ..r9_current_first.report import summarize,paired


def full_report(root):
    root=Path(root);state=json.loads((root/'queue.json').read_text())
    if state.get('status') not in ('COMPLETE','INCOMPLETE') or any(x['status'] in ('RUNNING','RETRY') for x in state['nodes'].values()):raise PermissionError('target scores remain sealed during the finite round')
    slots=SPEC['target_core_slots']+SPEC['target_final4000_slots_max'];results={};missing=[]
    def load(slot):
        p=root/'target'/slot;seal=json.loads((p/'score_complete.json').read_text());raw=(p/'scalars.private.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest()!=seal['scalar_sha256']:raise ValueError('score receipt')
        return [json.loads(x) for x in raw.splitlines()]
    for s in slots:
        if state['nodes'].get(s['id']+'__score',{}).get('status')!='COMPLETE':missing.append(s['id']);continue
        rows=load(s['id'])
        if len(rows)!=s['arrivals'] or sum(r['subset']=='remaining_dev' for r in rows)!=s['principal_visits']:raise ValueError('target coverage')
        results[s['id']]=dict(slot=s,aggregate=summarize(rows))
    groups=defaultdict(list)
    for r in results.values():
        s=r['slot'];groups[s['arm'],s.get('checkpoint','fixed')].append(r)
    means={}
    for (arm,checkpoint),rs in groups.items():
        seeds=defaultdict(dict)
        for r in rs:
            if r['slot']['order'] in (0,1):seeds[str(r['slot']['seed'])][r['slot']['order']]=r['aggregate']
        values={s:{m:st.mean(d[o][m] for o in (0,1)) for m in ('dice','soft_dice')} for s,d in seeds.items() if set(d)=={0,1}}
        means[arm+'|'+checkpoint]=dict(per_seed=values,equal_seed_mean={m:st.mean(v[m] for v in values.values()) if values else None for m in ('dice','soft_dice')})
    selection=json.loads((root/'selection.json').read_text()) if (root/'selection.json').exists() else None
    source={}
    for j in SPEC['source_jobs']:
        p=root/'source'/j['id']/'complete.json'
        if p.exists():
            r=json.loads(p.read_text());source[j['id']]={k:r[k] for k in ('method','steps','selection','points','deployment_gap') if k in r}
    comparisons={};long10={}
    for id_,r in results.items():
        s=r['slot'];targets=['C0','VPTTA_NATIVE','C_CTTA','G_CTTA','B_CARRIER_FULL']
        if s['arm']=='SELECTED_GR':targets+=['SELECTED_SUP','SELECTED_GR_CONST_HALF','SELECTED_GR_RESET_ALL','SELECTED_GR_FORCE_WRITE']
        if s['arm']=='GR_SEQ':targets+=['GR_CUR']
        if s['arm']=='GR_RET':targets+=['GR_SEQ','SUP_RET']
        if s['arm']=='GR_RET_EMA':targets+=['GR_RET']
        for arm in targets:
            seed=s['seed']
            if arm in ('C0','B_CARRIER_FULL'):seed=None
            elif arm in ('VPTTA_NATIVE','C_CTTA','G_CTTA') and seed in range(20260924,20260929):seed-=17
            candidates=[k for k,x in results.items() if k!=id_ and x['slot']['arm']==arm and x['slot']['seed']==seed and x['slot']['order']==s['order'] and x['slot'].get('checkpoint','fixed') in ('fixed',s.get('checkpoint','fixed'))]
            if len(candidates)==1:comparisons[id_+' vs '+candidates[0]]=paired(load(id_),load(candidates[0]))
        if s['order']=='LONG10':
            rows=load(id_);long10[id_]=paired([dict(x,cycle=1) for x in rows if x['cycle']==10],[dict(x,cycle=1) for x in rows if x['cycle']==1])
    d0_results={str(seed):json.loads((root/f'D0_{seed}.json').read_text()) for seed in (20260924,20260925) if (root/f'D0_{seed}.json').exists()}
    diagnostics={}
    for j in SPEC['source_jobs']:
        p=root/'source'/j['id']/'fit/physical.jsonl'
        if p.exists():
            rows=[json.loads(x) for x in p.read_text().splitlines()]
            diagnostics[j['id']]=dict(physical_rows=len(rows),last=rows[-1] if rows else None)
    ledger=json.loads((root/'ledger/state.json').read_text())
    return dict(schema='R10_PUBLIC_AGGREGATE_V1',status=state['status'],completed_sources=len(source),completed_slots=len(results),missing_slots=missing,results=results,method_groups=means,paired=comparisons,long10_first_last=long10,source_selection=selection,source=source,d0=d0_results,source_diagnostics=diagnostics,cost=Ledger.total(ledger),interpretation='Five policy seeds share one frozen carrier; first two selected families. Repeated visits/groups are not independent patients. SUP uses soft objectives and extra recomputation; RL uses detached hard rewards. Development target pool, not a new blind clinical evaluation.')
