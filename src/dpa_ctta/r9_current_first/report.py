"""Anonymous aggregate report only; never export content keys or per-image rows."""
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from .protocol import SPEC,digest
from .metrics import domain_macro


def summarize(rows):
    result={key:domain_macro(rows,key) for key in ('dice','soft_dice')}
    cells=defaultdict(list)
    for r in rows:
        if r['subset']=='remaining_dev':
            for m in r['metrics']:cells[r['domain'],m['channel']].append(m)
    result['domains']={}
    for (domain,ch),ms in sorted(cells.items()):
        cell=dict(n=len(ms))
        for key in ('dice','soft_dice','FP','FN','precision','recall','mean_probability','predicted_foreground_fraction','assd'):
            xs=[m[key] for m in ms if m[key] is not None]
            cell[key]=dict(mean=st.mean(xs) if xs else None,finite=len(xs),undefined=len(ms)-len(xs))
        result['domains'].setdefault(domain,{})[ch]=cell
    result['cycles']={str(c):{k:domain_macro([r for r in rows if r['cycle']==c],k) for k in ('dice','soft_dice')} for c in sorted({r['cycle'] for r in rows})}
    return result


def collect(root):
    root=Path(root);slots=SPEC['target_core_slots']+SPEC['target_final16k_slots_max'];results={};missing=[]
    state=json.loads((root/'queue.json').read_text())
    for slot in slots:
        if state['nodes'].get(slot['id']+'__score',{}).get('status')!='COMPLETE':missing.append(slot['id']);continue
        p=root/'target'/slot['id'];alias=p/'reuse.json'
        if alias.exists():p=root/'target'/json.loads(alias.read_text())['original_slot']
        receipt=p/'score_complete.json'
        if not receipt.exists():missing.append(slot['id']);continue
        raw=(p/'scalars.private.jsonl').read_bytes()
        import hashlib
        seal=json.loads(receipt.read_text())
        if hashlib.sha256(raw).hexdigest()!=seal['scalar_sha256']:raise ValueError('score output changed')
        rows=[json.loads(line) for line in raw.splitlines()]
        expected=19510 if slot['order']=='LONG10' else 1951
        if len(rows)!=expected or [r['visit'] for r in rows]!=list(range(1,expected+1)):raise ValueError('score coverage')
        results[slot['id']]=dict(slot=slot,aggregate=summarize(rows),score_receipt_sha256=digest(seal))
    return dict(schema='R9_AGGREGATE_V1',status='COMPLETE' if not missing else 'INCOMPLETE',completed_slots=len(results),missing_slots=missing,results=results,
                interpretation='Development data, repeated seeds/orders/cycles are not new patients. No automatic winner or external validation.')


def paired(a,b):
    """Paired content/visit summaries are descriptive, without independence claims."""
    def keys(rows):
        selected=[r for r in rows if r['subset']=='remaining_dev']
        out={(r['domain'],r['content'],r['cycle']):r for r in selected}
        if len(out)!=len(selected):raise ValueError('duplicate paired visit key')
        return out
    left,right=keys(a),keys(b)
    if set(left)!=set(right):raise ValueError('paired content/cycle coverage differs')
    delta=[];domains=defaultdict(list);assd={c:[] for c in ('OD','OC')};undefined={c:0 for c in assd}
    for key in left:
        lm={m['channel']:m for m in left[key]['metrics']};rm={m['channel']:m for m in right[key]['metrics']}
        d=st.mean(lm[c]['dice']-rm[c]['dice'] for c in lm);delta.append(d);domains[key[0]].append(d)
        for c in assd:
            x,y=lm[c]['assd'],rm[c]['assd']
            if x is None or y is None:undefined[c]+=1
            else:assd[c].append(x-y)
    ordered=sorted(delta);n=max(1,(len(delta)+9)//10)
    return dict(content_visits=len(delta),equal_domain_delta=st.mean(st.mean(v) for v in domains.values()),
                pooled_descriptive=dict(mean=st.mean(delta),worst=min(delta),worst_decile_mean=st.mean(ordered[:n]),negative=sum(x<0 for x in delta),positive=sum(x>0 for x in delta)),
                ASSD={c:dict(common=len(v),undefined=undefined[c],mean_delta=st.mean(v) if v else None) for c,v in assd.items()})


def full_report(root):
    root=Path(root)
    state=json.loads((root/'queue.json').read_text())
    if state.get('status') not in ('COMPLETE','INCOMPLETE') or any(v['status'] in ('RUNNING','RETRY') for v in state['nodes'].values()):raise PermissionError('scores remain sealed until the finite run ends')
    result=collect(root);groups=defaultdict(list)
    for r in result['results'].values():
        s=r['slot'];groups[(s['phase'],s['arm'],str(s['checkpoint']))].append(r)
    result['method_groups']={}
    for (phase,arm,checkpoint),rs in groups.items():
        seeds=defaultdict(list)
        for r in rs:
            if r['slot']['order'] in (0,1):seeds[str(r['slot']['seed'])].append(r)
        perseed={s:st.mean(r['aggregate']['dice'] for r in rows) for s,rows in seeds.items() if {r['slot']['order'] for r in rows}=={0,1}}
        result['method_groups']['|'.join((phase,arm,checkpoint))]=dict(primary_equal_seed_order_mean=st.mean(perseed.values()) if perseed else None,primary_seed_means=perseed,
            orders={str(order):st.mean(r['aggregate']['dice'] for r in rs if r['slot']['order']==order) for order in {r['slot']['order'] for r in rs}},worst_domain_channel=min(v['dice']['mean'] for r in rs for cs in r['aggregate']['domains'].values() for v in cs.values()))
    result['source_selection']={}
    for j in SPEC['source_tasks']:
        path=root/'source'/j['id']/'complete.json'
        if path.exists() and state['nodes'].get(j['id'],{}).get('status')=='COMPLETE':
            r=json.loads(path.read_text());result['source_selection'][j['id']]=dict(job=j,selection=r['selection'],fit_steps=r['fit_steps'])
    for name in ('recipe_selection','gradient_selection'):
        p=root/(name+'.json');result[name]=json.loads(p.read_text()) if p.exists() else None
    def rows(slot):
        p=root/'target'/slot
        if (p/'reuse.json').exists():p=root/'target'/json.loads((p/'reuse.json').read_text())['original_slot']
        return [json.loads(line) for line in (p/'scalars.private.jsonl').read_text().splitlines()]
    complete=result['results'];result['paired']={};result['long10_first_last']={}
    for id_,r in complete.items():
        s=r['slot'];comparators=['C0_CURRENT_STATS','VPTTA_NATIVE']
        if s['arm'].endswith('_FULL'):comparators.append(s['arm'].removesuffix('_FULL')+'_STATIC')
        if s['arm'] in ('A_SELECTED_FULL','B_SELECTED_FULL'):comparators.append(s['arm'][0]+'_RESET')
        if '_SELF_' in s['arm']:comparators.append(s['arm'].replace('_SELF_','_LEGACY_'))
        if '_SELF_TASK_' in s['arm']:comparators.append(s['arm'].replace('_SELF_TASK_','_SELF_'))
        if s['arm'] in ('B_G1','B_G3','B_RESET_G3','COLD_G1','COLD_G3','BN_RESET_G1'):comparators.append('B_SELECTED_FULL')
        a=None
        for arm in comparators:
            seed=None if arm=='C0_CURRENT_STATS' else (20260907+s['seed']-20260924 if arm=='VPTTA_NATIVE' and s['seed'] in range(20260924,20260929) else s['seed'])
            candidates=[k for k,v in complete.items() if v['slot']['arm']==arm and v['slot']['order']==s['order'] and v['slot']['seed']==seed and k!=id_ and (v['slot']['checkpoint']==s['checkpoint'] or arm in ('C0_CURRENT_STATS','VPTTA_NATIVE'))]
            if len(candidates)!=1:continue
            a=rows(id_) if a is None else a;result['paired'][id_+' vs '+candidates[0]]=paired(a,rows(candidates[0]))
        if s['order']=='LONG10':
            a=rows(id_) if a is None else a
            first=[dict(x,cycle=1) for x in a if x['cycle']==1];last=[dict(x,cycle=1) for x in a if x['cycle']==10]
            result['long10_first_last'][id_]=paired(last,first)
    ledger=root/'ledger'/'state.json'
    if ledger.exists():
        from .ledger import Ledger
        d=json.loads(ledger.read_text());known=[];unknown=[]
        for name,a in d['attempts'].items():
            receipt=root/'attempts'/(name+'.json')
            if receipt.exists():known.append(json.loads(receipt.read_text())['actual'])
            else:unknown.append(name)
        result['cost']=dict(charged=Ledger.total(d),attempts=len(d['attempts']),failed=sum(a['status']=='FAILED' for a in d['attempts'].values()),
            new_physical_recorded={k:sum(a[k] for a in known) for k in ('model_forwards','backward_calls','optimizer_steps','vjp_calls')},unclosed_attempts=unknown)
    return result
