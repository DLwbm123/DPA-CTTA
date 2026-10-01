"""Terminal-only gate mechanism report; original baselines and all adverse effects retained."""
import json,statistics,time
from collections import defaultdict
from pathlib import Path
from ..r10_12h_core.run import read,save,sha,epoch,OPS
from ..r10_attribution.report import means,paired,write_csv
from ..r13_full_coverage.coverage import scalar_rows,compose
from .view import NAME
from .run import sealed

def report(c,state):
    if state['status']=='RUNNING' or any(v=='RUNNING' for v in state['jobs'].values()):raise ValueError('target scores embargoed')
    root=Path(c['output_root']);allrows={};table=[];cells=[];contrasts=[];seals=[];soft=[];eligibility=[]
    for x in c['reused_results']:
        parts=[sealed(p) for p in x['parts']];full=read(c['manifests'][x['order']]['path']);rs=compose(full,[scalar_rows(Path(p['path'])/'scalars.private.jsonl') for p in parts]);allrows[x['condition'],x['order']]=rs;seals.append(dict(condition=x['condition'],order=x['order'],origin='REUSED',parts=[{k:p[k] for k in ('online','score')} for p in parts]))
    for j in c['jobs']:
        if state['jobs'][j['id']]=='COMPLETE':
            p=root/'target'/j['id'];sc=read(p/'score_complete.json');on=read(p/'online_complete.json');rs=scalar_rows(p/'scalars.private.jsonl');full=read(c['manifests'][j['order']]['path'])
            if sc['visits']!=1951 or sc['principal']!=1695:raise ValueError('full gated coverage')
            sealed(dict(path=str(p),online=on,score=sc));traces=scalar_rows(p/'visits.jsonl')
            if len(traces)!=len(rs):raise ValueError('gated eligibility trace coverage')
            for domain in ['ALL']+sorted({r['domain'] for r in rs}):
                values=[t['uncertain_fraction'] for r,t in zip(rs,traces) if r['subset']=='remaining_dev' and (domain=='ALL' or r['domain']==domain)]
                eligibility.append(dict(condition=NAME,order=j['order'],domain=domain,n=len(values),eligible_pixel_fraction=statistics.mean(values),protected_pixel_fraction=1-statistics.mean(values)))
            allrows[NAME,j['order']]=compose(full,[rs]);seals.append(dict(condition=NAME,order=j['order'],origin='NEW',parts=[dict(online=on,score=sc,retirement=read(p/'probabilities_retired.json'))]))
        else:table.append(dict(condition=NAME,order=j['order'],origin='NEW',status=state['jobs'][j['id']],visits=None,principal=None,Dice_OD=None,Dice_OC=None,Dice_macro=None))
    for (name,o),rs in sorted(allrows.items()):
        table.append(dict(condition=name,order=o,origin='NEW' if name==NAME else 'REUSED',status='COMPLETE',visits=1951,principal=1695,**means(rs)));groups=defaultdict(list)
        for r in rs:
            if r['subset']=='remaining_dev':
                for m in r['metrics']:groups[r['domain'],m['channel']].append(m['dice'])
        for (domain,ch),vs in sorted(groups.items()):cells.append(dict(condition=name,order=o,domain=domain,channel=ch,n=len(vs),Dice=statistics.mean(vs)))
    for o in (0,1):
        a=allrows.get((NAME,o))
        if a is None:continue
        for name in ('C0','CV_H025','G'):
            b=allrows[name,o]
            for d in ['ALL']+sorted({r['domain'] for r in a}):
                for ch in ('OD','OC','macro'):
                    p=paired(a,b,d,ch);contrasts.append(dict(left=NAME,right=name,order=o,domain=d,channel=ch,delta=p['domain_equal_delta'],n=p['n'],**{'paired_'+k:v for k,v in p['paired_image_distribution'].items() if k!='n'},tail_last_quarter_n=p['tail_last_quarter_n'],tail_last_quarter_delta=p['tail_last_quarter_delta']))
            for r,y in zip(a,b):
                if (r['content'],r['domain'],r['subset'])!=(y['content'],y['domain'],y['subset']):raise ValueError('soft paired identity')
            for d in ['ALL']+sorted({r['domain'] for r in a}):
                for ch in ('OD','OC'):
                    g=defaultdict(list)
                    for r,y in zip(a,b):
                        if r['subset']=='remaining_dev' and (d=='ALL' or r['domain']==d):
                            v=next(m for m in r['metrics'] if m['channel']==ch);w=next(m for m in y['metrics'] if m['channel']==ch)
                            if 'soft_dice' in w:g[r['domain']].append(v['soft_dice']-w['soft_dice'])
                    soft.append(dict(left=NAME,right=name,order=o,domain=d,channel=ch,status='COMPLETE' if g else 'NA_LEGACY_BITMASK',n=sum(map(len,g.values())) if g else None,soft_Dice_delta=statistics.mean(statistics.mean(v) for v in g.values()) if g else None))
    for name,rs in [('RESULTS.csv',table),('DOMAIN_CHANNEL.csv',cells),('TARGET_CONTRASTS.csv',contrasts),('TARGET_SOFT_CONTRASTS.csv',soft),('TARGET_ELIGIBILITY.csv',eligibility)]:write_csv(root/name,rs,fields=None if rs else ['status'])
    src=read(root/'SOURCE_COMPARISON.json') if (root/'SOURCE_COMPARISON.json').exists() else dict(new_rows=[],status='NOT_RUN');rows=c['source_reference']+src.get('new_rows',[]);write_csv(root/'SOURCE_EPISODES.csv',rows);summary=[]
    for name in ('C0','CV_H025',NAME):
        for mode in ['ALL']+sorted({r['mode'] for r in rows}):
            selected=[r for r in rows if r['condition']==name and (mode=='ALL' or r['mode']==mode)];vals={}
            for k in ('hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice'):
                g=defaultdict(list)
                for r in selected:g[r['mode']].append(r[k])
                vals[k]=statistics.mean(statistics.mean(v) for v in g.values()) if g else None
            summary.append(dict(condition=name,mode=mode,episodes=len(selected),**vals))
    write_csv(root/'SOURCE_COMPARISON.csv',summary)
    delta=[next((r['delta'] for r in contrasts if r['right']=='C0' and r['order']==o and r['domain']=='ALL' and r['channel']=='macro'),None) for o in (0,1)];quarter=[next((r['delta'] for r in contrasts if r['right']=='CV_H025' and r['order']==o and r['domain']=='ALL' and r['channel']=='macro'),None) for o in (0,1)];worst=min((r['delta'] for r in contrasts if r['right']=='C0' and r['domain']!='ALL' and r['channel'] in ('OD','OC')),default=None);ss={r['condition']:r for r in summary if r['mode']=='ALL'};source_signal=None
    if src['status']=='COMPLETE':source_signal=dict(soft_half_gap_recovered=ss[NAME]['soft_Dice']>=(ss['C0']['soft_Dice']+ss['CV_H025']['soft_Dice'])/2,hard_loss_vs_quarter_within_0_5pp=ss[NAME]['hard_Dice']>=ss['CV_H025']['hard_Dice']-.005,selection=False)
    decision=dict(matrix_complete=state['status']=='COMPLETE',delta_vs_C0=delta,delta_vs_constant_quarter=quarter,meets_original_priority=all(v is not None and v>0 for v in delta) and statistics.mean(delta)>=.005 if state['status']=='COMPLETE' else None,beats_quarter_both_orders=all(v is not None and v>0 for v in quarter) if state['status']=='COMPLETE' else None,worst_domain_channel_delta=worst,risk_over_2pp=worst is not None and worst<-.02,source_mechanism=source_signal,independent_confirmation=False)
    save(root/'DECISION.json',decision);save(root/'RESULT_SEALS.json',dict(results=seals,constant_quarter_full_online_seal=False));attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))];ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(status=state['status'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),operations={k:sum(a['cost'].get(k,0) for a in attempts) for k in OPS},attempts=attempts,recovery_used=state['recovery_used'],disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()));save(root/'RESOURCE_LEDGER.json',ledger)
    text=[f'# {c["experiment_id"]}',f'Status {state["status"]}; execution {c["code_sha"]}.','Fixed native probability[.4,.6] eligibility with quarter flip mixing; confident original logits exact. Two new full1951/1695 gated trajectories only; complete prior C0/constantquarter/nativeGraTa baselines reused. New scores reviewed only after terminal matrix.','| condition | order | OD % | OC % | domain Dice % |','|---|---:|---:|---:|---:|']
    for r in table:text.append(f'| {r["condition"]} | {r["order"]} | '+' | '.join('MISSING' if r.get(k) is None else f'{100*r[k]:.6f}' for k in ('Dice_OD','Dice_OC','Dice_macro'))+' |')
    text+=['```json',json.dumps(decision,indent=2),'```','Source and target signs retained, no score-based tuning. Original+.5pp priority and >2pp domain-risk threshold unchanged. Legacy probability/softNA; new gated-versus-constantquarter soft contrasts explicit. All data are already exposed development evidence, two correlated deterministic orders, unknown patients; no independent confirmation/new CTTA/clinical/RL claim. Physical cost and all failure/recovery expenses retained.']
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
