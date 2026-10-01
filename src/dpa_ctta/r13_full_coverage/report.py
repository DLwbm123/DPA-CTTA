"""Full hard-score comparisons with explicit partition seals and history provenance."""
import json,statistics,time
from pathlib import Path
from collections import defaultdict
from ..r10_12h_core.run import read,save,sha,epoch,OPS
from ..r10_attribution.report import means,paired,write_csv
from ..r8_ba.journal import _digest
from .coverage import scalar_rows,compose

def report(c,state):
    if state['status']=='RUNNING' or any(v=='RUNNING' for v in state['jobs'].values()):raise ValueError('target score embargo')
    root=Path(c['output_root']);allrows={};table=[];cells=[];contrasts=[];parts=[]
    for x in c['reused_results']:
        p=Path(x['path']);sc=read(p/'score_complete.json')
        if sc!=x['score'] or read(p/'online_complete.json')!=x['online'] or _digest(p/'scalars.private.jsonl')!=sc['scalar_sha256']:raise ValueError('reused seal changed')
        rs=scalar_rows(p/'scalars.private.jsonl');allrows['SHORT' if x['role']=='SHORT_PARTITION' else 'FULL',x['condition'],x['order']]=rs
        parts.append(dict(condition=x['condition'],order=x['order'],role=x['role'],online=x['online'],score=x['score']))
    for j in c['jobs']:
        o=j['order'];p=root/'target'/j['id'];full=read(c['full_manifests'][o]['path']);complement=read(c['manifests'][o]['path'])
        if state['jobs'][j['id']]=='COMPLETE':
            sc=read(p/'score_complete.json');rs=scalar_rows(p/'scalars.private.jsonl')
            if len(rs)!=927 or sc['principal']!=807 or sc['scalar_sha256']!=_digest(p/'scalars.private.jsonl'):raise ValueError('complement terminal coverage')
            old=allrows['SHORT','CV_H025',o];allrows['FULL','CV_H025',o]=compose(full,[old,rs]);allrows['COMPLEMENT','CV_H025',o]=compose(complement,[rs]);parts.append(dict(condition=j['arm'],order=o,role='NEW_COMPLEMENT',online=read(p/'online_complete.json'),score=sc,retirement=read(p/'probabilities_retired.json')))
        else:table.append(dict(scope='FULL',condition='CV_H025',order=o,status='MISSING',visits=None,principal=None,Dice_OD=None,Dice_OC=None,Dice_macro=None))
        c0=allrows['FULL','C0',o];keep={r['group_id'] for r in complement};short_manifest=read(read(Path(c['previous_root'])/'RESOLVED_CONFIG.json')['manifests'][o]['path'])
        allrows['COMPLEMENT','C0',o]=compose(complement,[[r for r in c0 if r['content'] in keep]])
        allrows['SHORT','C0',o]=compose(short_manifest,[[r for r in c0 if r['content'] not in keep]])
    for (scope,name,o),rs in sorted(allrows.items()):
        m=means(rs);n=sum(r['subset']=='remaining_dev' for r in rs);table.append(dict(scope=scope,condition=name,order=o,status='COMPLETE',visits=len(rs),principal=n,**m));groups=defaultdict(list)
        for r in rs:
            if r['subset']=='remaining_dev':
                for metric in r['metrics']:groups[r['domain'],metric['channel']].append(metric['dice'])
        for (domain,ch),values in sorted(groups.items()):cells.append(dict(scope=scope,condition=name,order=o,domain=domain,channel=ch,n=len(values),Dice=statistics.mean(values)))
    for scope in ('FULL','COMPLEMENT','SHORT'):
        for o in (0,1):
            a=allrows.get((scope,'CV_H025',o))
            if a is None:continue
            for name in ('C0','G') if scope=='FULL' else ('C0',):
                b=allrows[scope,name,o]
                for domain in ['ALL']+sorted({r['domain'] for r in a}):
                    for ch in ('OD','OC','macro'):
                        p=paired(a,b,domain,ch);contrasts.append(dict(scope=scope,left='CV_H025',right=name,order=o,domain=domain,channel=ch,delta=p['domain_equal_delta'],n=p['n'],**{'paired_'+k:v for k,v in p['paired_image_distribution'].items() if k!='n'},tail_last_quarter_n=p['tail_last_quarter_n'],tail_last_quarter_delta=p['tail_last_quarter_delta']))
    for name,rows in [('RESULTS.csv',table),('DOMAIN_CHANNEL.csv',cells),('TARGET_CONTRASTS.csv',contrasts)]:write_csv(root/name,rows)
    full_delta=[next((r['delta'] for r in contrasts if r['scope']=='FULL' and r['right']=='C0' and r['order']==o and r['domain']=='ALL' and r['channel']=='macro'),None) for o in (0,1)]
    worst=min((r['delta'] for r in contrasts if r['scope']=='FULL' and r['right']=='C0' and r['domain']!='ALL' and r['channel'] in ('OD','OC')),default=None)
    decision=dict(matrix_complete=state['status']=='COMPLETE',full_delta_vs_C0=full_delta,meets_priority_scale=all(v is not None and v>0 for v in full_delta) and statistics.mean(full_delta)>=.005 if state['status']=='COMPLETE' else None,worst_domain_channel_delta=worst,risk_over_2pp=worst is not None and worst<-.02,independent_confirmation=False,meaning='Broader already exposed development coverage, correlated deterministic orders; original priority threshold unchanged')
    save(root/'DECISION.json',decision);save(root/'PARTITION_SEALS.json',dict(full_probability_seal=False,composition='Exact disjoint immutable per-content score union reordered by full registered arrivals; native full GraTa trajectories reused intact',parts=parts))
    src=read(Path(c['previous_root'])/'SOURCE_COMPARISON.json')
    if sha(src)!=c['reused_source_sha256']:raise ValueError('source qualification changed')
    save(root/'SOURCE_COMPARISON.reused.json',src)
    for name in ('SOURCE_EPISODES.csv','SOURCE_COMPARISON.csv'):(root/name).write_text((Path(c['previous_root'])/name).read_text())
    attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))];ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(status=state['status'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),operations={k:sum(a['cost'].get(k,0) for a in attempts) for k in OPS},attempts=attempts,recovery_used=state['recovery_used'],disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()));save(root/'RESOURCE_LEDGER.json',ledger)
    lines=[f'# {c["experiment_id"]}',f'Status {state["status"]}; execution {c["code_sha"]}.','Only the fixed quarter-weight complementary927 arrivals/order were newly inferred; full1951 coverage is composed from disjoint sealed short1024 and complement927 partitions. Primary1695 =888+807. No invented full online seal or training seed replication. C0/GraTa complete1951 historical native trajectories reused with hard-metric compatibility qualification. Legacy probability/soft metrics are unavailable (NA).','| scope | condition | order | OD % | OC % | domain Dice % |','|---|---|---:|---:|---:|---:|']
    for r in table:lines.append(f'| {r["scope"]} | {r["condition"]} | {r["order"]} | '+' | '.join('MISSING' if r.get(k) is None else f'{100*r[k]:.6f}' for k in ('Dice_OD','Dice_OC','Dice_macro'))+' |')
    lines+=['```json',json.dumps(decision,indent=2),'```','Full old development cohort was exposed during R7/R8. Orders share content, unknown patient dependence. No independent confirmation, blind-test, new CTTA method, clinical or RL claim. Source quarter hard gain and large soft loss are retained without tuning. All target scores reviewed only after terminal matrix. See full domain/channel, paired distribution, source and cost tables.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
