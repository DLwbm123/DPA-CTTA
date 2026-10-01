"""Unblind all prescribed conditions only when the fixed matrix is terminal."""
import json,statistics,time
from collections import defaultdict
from pathlib import Path
from ..r10_12h_core.run import read,save,epoch,OPS
from ..r10_attribution.report import means,paired,write_csv

def report(c,state):
    if state['status']=='RUNNING' or any(v=='RUNNING' for v in state['jobs'].values()):raise ValueError('target embargo')
    root=Path(c['output_root']);allrows={};table=[];cells=[];contrasts=[]
    jobs=[(j,root,'NEW',state['jobs'][j['id']]) for j in c['jobs']]+[(x['job'],Path(x['root']),'REUSED','REUSED_COMPLETE') for x in c['reused_results']]
    for j,base,origin,status in jobs:
        name={'C0_CURRENT_STATS':'C0'}.get(j['arm'],j['arm']);row=dict(condition=name,order=j['order'],origin=origin,status=status,visits=None,principal=None,Dice_OD=None,Dice_OC=None,Dice_macro=None)
        if status in ('COMPLETE','REUSED_COMPLETE'):
            p=base/'target'/j['id'];rs=[json.loads(x) for x in (p/'scalars.private.jsonl').read_text().splitlines()];sc=read(p/'score_complete.json')
            if len(rs)!=1024 or sc['principal']!=888:raise ValueError('terminal score coverage')
            allrows[name,j['order']]=rs;row.update(visits=1024,principal=888,**means(rs));groups=defaultdict(list)
            for r in rs:
                if r['subset']=='remaining_dev':
                    for m in r['metrics']:groups[r['domain'],m['channel']].append(m['dice'])
            for (domain,ch),vs in sorted(groups.items()):cells.append(dict(condition=name,order=j['order'],domain=domain,channel=ch,n=len(vs),Dice=statistics.mean(vs)))
        table.append(row)
    for a in c['candidate_conditions']:
        for b in ('C0','G','CV_H2'):
            for order in (0,1):
                if (a,order) not in allrows:continue
                for domain in ['ALL']+sorted({r['domain'] for r in allrows[a,order]}):
                    for ch in ('OD','OC','macro'):
                        p=paired(allrows[a,order],allrows[b,order],domain,ch)
                        contrasts.append(dict(left=a,right=b,order=order,domain=domain,channel=ch,delta=p['domain_equal_delta'],n=p['n'],**{'paired_'+k:v for k,v in p['paired_image_distribution'].items() if k!='n'},tail_last_quarter_n=p['tail_last_quarter_n'],tail_last_quarter_delta=p['tail_last_quarter_delta']))
    for file,rs in [('RESULTS.csv',table),('DOMAIN_CHANNEL.csv',cells),('TARGET_CONTRASTS.csv',contrasts)]:write_csv(root/file,rs)
    src=read(root/'SOURCE_COMPARISON.json') if (root/'SOURCE_COMPARISON.json').exists() else dict(rows=[],status='NOT_RUN');source_rows=src['rows']+c['reused_source_C0'];write_csv(root/'SOURCE_EPISODES.csv',source_rows)
    sr=[]
    for name in dict.fromkeys(('C0',*c['source_conditions'])):
        for mode in ['ALL']+sorted({r['mode'] for r in source_rows}):
            selected=[r for r in source_rows if r['condition']==name and (mode=='ALL' or r['mode']==mode)];values={}
            for k in ('hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice'):
                groups=defaultdict(list)
                for r in selected:groups[r['mode']].append(r[k])
                values[k]=statistics.mean(statistics.mean(v) for v in groups.values()) if groups else None
            sr.append(dict(condition=name,mode=mode,episodes=len(selected),**values))
    write_csv(root/'SOURCE_COMPARISON.csv',sr)
    complete=state['status']=='COMPLETE';signals={}
    for a in c['candidate_conditions']:
        ds=[next((r['delta'] for r in contrasts if r['left']==a and r['right']=='C0' and r['order']==o and r['domain']=='ALL' and r['channel']=='macro'),None) for o in (0,1)]
        worst=min((r['delta'] for r in contrasts if r['left']==a and r['right']=='C0' and r['domain']!='ALL' and r['channel'] in ('OD','OC')),default=None)
        signals[a]=dict(delta_vs_C0=ds,worst_domain_channel_delta=worst,meets_priority_scale=all(v is not None and v>0 for v in ds) and statistics.mean(ds)>=.005 if complete else None)
    decision=dict(matrix_complete=complete,signals=signals,automatic_followon_authorized=False,meaning='Prespecified exploratory flip-bias/weight diagnostic after negative R11, not an independent confirmation or new CTTA method. Two orders share contents; no independent replication or clinical/general RL conclusion.')
    save(root/'DECISION.json',decision)
    attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))];ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(status=state['status'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),operations={k:sum(a['cost'].get(k,0) for a in attempts) for k in OPS},attempts=attempts,recovery_used=state['recovery_used'],disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()));save(root/'RESOURCE_LEDGER.json',ledger)
    lines=[f'# {c["experiment_id"]}',f'Status: {state["status"]}; execution SHA {c["code_sha"]}.','Fixed horizontal flip weights .25/1, zero parameter updates; target weights 0/.5 and GraTa reused with matching seals. All target scores unblinded after terminal matrix. No condition selection or tuning from source/target scores.','| condition | order | origin/status | domain Dice % |','|---|---:|---|---:|']
    for r in table:lines.append(f'| {r["condition"]} | {r["order"]} | {r["origin"]}/{r["status"]} | '+('MISSING' if r['Dice_macro'] is None else f'{100*r["Dice_macro"]:.6f}')+' |')
    lines+=['```json',json.dumps(decision,indent=2),'```','Priority scale is +0.5 percentage point vs C0, both orders positive; descriptive development rule, not significance. Preserved domain/OD/OC degradations and paired distributions accompany all means. Unknown patient dependence; source simulator validation is development evidence only.','No RL, B, history, training, view search, seed expansion or automatic follow-on. Anonymous summaries exclude private identities, images, labels, predictions, host paths and credentials.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
