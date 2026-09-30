"""Terminal-only fixed factorial contrasts, source/target direction and seal parity."""
import csv,json,statistics,time
from collections import defaultdict
from pathlib import Path
from ..r10_12h_core.run import read,save,epoch,OPS
from ..r10_attribution.report import means,paired,values,write_csv
from ..r10_attribution.audit import distribution
from ..r9_current_first.storage import load_torch
from .diagnostic import equal_state

PAIRS=(('B_RESET_1','B_FULL_1'),('B_FULL_READOUT_025','B_FULL_1'),('B_RESET_READOUT_025','B_RESET_1'))
LEVELS=('B_ZERO','B_RESET_1','B_FULL_READOUT_025','B_RESET_READOUT_025')

def interaction(rows,order,domain='ALL',channel='macro'):
    a,b,d,e=(rows[k,order] for k in ('B_FULL_READOUT_025','B_FULL_1','B_RESET_READOUT_025','B_RESET_1'))
    paired(a,b,domain,channel);paired(d,e,domain,channel);paired(a,d,domain,channel)
    vals=[];cells=defaultdict(list)
    for aa,bb,dd,ee in zip(a,b,d,e):
        if aa['subset']=='remaining_dev' and (domain=='ALL' or aa['domain']==domain):
            v=(values(aa,channel)-values(bb,channel))-(values(dd,channel)-values(ee,channel));vals.append(v);cells[aa['domain']].append(v)
    return dict(delta=statistics.mean(statistics.mean(v) for v in cells.values()),n=len(vals),**{'paired_'+k:v for k,v in distribution(vals).items() if k!='n'})

def latest_snapshot(p):
    candidates=[]
    for meta in p.glob('checkpoint.*.json'):
        m=read(meta);candidates.append((m['visits'],meta,m))
    _,meta,m=max(candidates,key=lambda x:x[0]);return load_torch(meta.with_suffix('.pt'),m['sha256'])['host']

def report(c,state):
    if state['status']=='RUNNING' or any(s=='RUNNING' for s in state['jobs'].values()):raise ValueError('target embargo')
    root=Path(c['output_root']);allrows={};table=[];domains=[];diag=[];traces={};receipts={}
    jobs=[(j,root,'NEW',state['jobs'][j['id']]) for j in c['jobs']]+[(x['job'],Path(x['root']),'REUSED','REUSED_COMPLETE') for x in c['reused_results']]
    for j,base,origin,status in jobs:
        name={'C0_CURRENT_STATS':'C0','B_PARENT_FULL':'B_FULL_1'}.get(j['arm'],j['arm']);row=dict(condition=name,order=j['order'],origin=origin,status=status,visits=None,principal=None,Dice_OD=None,Dice_OC=None,Dice_macro=None,Dice_pooled=None,worst_domain=None);p=base/'target'/j['id']
        if status in ('COMPLETE','REUSED_COMPLETE'):
            rs=[json.loads(x) for x in (p/'scalars.private.jsonl').read_text().splitlines()];sc=read(p/'score_complete.json');on=read(p/'online_complete.json')
            if len(rs)!=1024 or sc['principal']!=sum(x['subset']=='remaining_dev' for x in rs):raise ValueError('report coverage')
            allrows[name,j['order']]=rs;receipts[name,j['order']]=on;row.update(visits=len(rs),principal=sc['principal'],**means(rs))
            cells=defaultdict(list)
            for r in rs:
                if r['subset']=='remaining_dev':
                    for m in r['metrics']:cells[r['domain'],m['channel']].append(m['dice'])
            for (d,ch),vs in sorted(cells.items()):domains.append(dict(condition=name,order=j['order'],domain=d,channel=ch,n=len(vs),Dice=statistics.mean(vs)))
            ts=[json.loads(x) for x in (p/'visits.jsonl').read_text().splitlines()];traces[name,j['order']]=ts
            switch=[];remaining=0;last=None
            for r,t in zip(rs,ts):
                if last is not None and last!=r['domain']:remaining=32
                switch.append('first32_after_switch' if remaining else 'other');remaining=max(0,remaining-1);last=r['domain']
            for d in sorted({r['domain'] for r in rs}):
                for part in ('ALL','first32_after_switch','other'):
                    selected=[t for r,t,window in zip(rs,ts,switch) if r['domain']==d and (part=='ALL' or window==part)]
                    keys=('code_norm','ambient_norm','tanh_saturation','B_z_norm','B_d_norm','up1_original_increment_ratio','up3_original_increment_ratio','up1_readout_increment_ratio','up3_readout_increment_ratio','up1_feature_rms','up3_feature_rms')
                    for k in keys:diag.append(dict(origin='TARGET',condition=name,order=j['order'],domain=d,window=part,quantity=k,**distribution([t[k] for t in selected if k in t])))
        table.append(row)
    write_csv(root/'RESULTS.csv',table);write_csv(root/'DOMAIN_CHANNEL.csv',domains)
    pairs=(*PAIRS,*((k,other) for k in LEVELS for other in ('C0','G')))
    contrasts=[]
    for a,b in pairs:
        for o in (0,1):
            if (a,o) not in allrows or (b,o) not in allrows:
                contrasts.append(dict(left=a,right=b,order=o,domain='ALL',channel='macro',status='MISSING',delta=None));continue
            for d in ['ALL']+sorted({r['domain'] for r in allrows[a,o]}):
                for ch in ('OD','OC','macro'):
                    p=paired(allrows[a,o],allrows[b,o],d,ch);contrasts.append(dict(left=a,right=b,order=o,domain=d,channel=ch,status='COMPLETE',delta=p['domain_equal_delta'],n=p['n'],**{'paired_'+k:v for k,v in p['paired_image_distribution'].items() if k!='n'},tail_last_quarter_n=p['tail_last_quarter_n'],tail_last_quarter_delta=p['tail_last_quarter_delta']))
    for o in (0,1):
        if all((k,o) in allrows for k in ('B_FULL_READOUT_025','B_FULL_1','B_RESET_READOUT_025','B_RESET_1')):
            for d in ['ALL']+sorted({r['domain'] for r in allrows['B_FULL_1',o]}):
                for ch in ('OD','OC','macro'):contrasts.append(dict(left='SCALE_FULL_MINUS_SCALE_RESET',right='interaction',order=o,domain=d,channel=ch,status='COMPLETE',**interaction(allrows,o,d,ch)))
    write_csv(root/'TARGET_FACTORIAL.csv',contrasts)
    parity=[];stateproof=[]
    for o in (0,1):
        if ('B_ZERO',o) in receipts:
            a,b=receipts['B_ZERO',o],receipts['C0',o];parity.append(dict(order=o,compatible_dtype='little endian float32 sigmoid probabilities',same_predictions=a['prediction_sha256']==b['prediction_sha256'],same_bytes=a['prediction_bytes']==b['prediction_bytes'],zero_prediction_sha256=a['prediction_sha256'],C0_prediction_sha256=b['prediction_sha256']))
        for a,b in (('B_ZERO','B_FULL_READOUT_025'),('B_RESET_1','B_RESET_READOUT_025')):
            if (a,o) in traces and (b,o) in traces:
                aa,bb=traces[a,o],traces[b,o];ok=len(aa)==len(bb)==1024 and all(x['B_state_sha256']==y['B_state_sha256'] and x['B_counter']==y['B_counter'] for x,y in zip(aa,bb));stateproof.append(dict(order=o,left=a,right=b,visits=1024,state_trajectory_exact=ok))
                if not ok:raise ValueError('target readout intervention changed recurrence')
        if state['jobs'].get(f'B_FULL_READOUT_025_o{o}')=='COMPLETE':
            old=next(x for x in c['reused_results'] if x['job']['arm']=='B_PARENT_FULL' and x['job']['order']==o);a=latest_snapshot(root/'target'/f'B_FULL_READOUT_025_o{o}');b=latest_snapshot(Path(old['root'])/'target'/old['job']['id']);ok=a['visits']==b['visits']==1000 and equal_state(a['state'],b['state']);stateproof.append(dict(order=o,left='B_FULL_READOUT_025',right='old B_FULL_1',last_available_checkpoint_visit=a['visits'],state_checkpoint_exact=ok,note='Old target journal retains checkpoints through visit1000, not a visit1024 final-state snapshot.'))
            if not ok:raise ValueError('target endpoint does not match original native B history')
    z=read(root/'ZERO_PARITY.json');z.update(target_compatible_probability_seals=parity,target_internal_state_proofs=stateproof,target_direct_pixel_parity='Identical compatible whole-trajectory float32 probability digests' if len(parity)==2 and all(p['same_predictions'] and p['same_bytes'] for p in parity) else 'Unavailable or different; inspect parity records');save(root/'ZERO_PARITY.json',z)
    src=read(root/'SOURCE_COMPARISON.json') if (root/'SOURCE_COMPARISON.json').exists() else dict(status='NOT_RUN',rows=[],diagnostics=[])
    sr=[]
    for k in c['source_conditions']:
        for mode in ['ALL']+sorted({r['mode'] for r in src['rows']}):
            rs=[r for r in src['rows'] if r['condition']==k and (mode=='ALL' or r['mode']==mode)]
            for ch in ('OD','OC','Dice'):
                for metric in ('hard','soft'):
                    by=defaultdict(list)
                    for r in rs:by[r['mode']].append(r[metric+'_'+ch])
                    value=statistics.mean(statistics.mean(v) for v in by.values()) if by else None;sr.append(dict(condition=k,mode=mode,metric=metric,channel='macro' if ch=='Dice' else ch,episodes=len(rs),visits=32*len(rs),Dice=value))
    write_csv(root/'SOURCE_COMPARISON.csv',sr)
    for k in c['source_conditions']:
        for mode in sorted({r['mode'] for r in src['rows']}):
            rs=[r for r in src['diagnostics'] if r['condition']==k and r['mode']==mode]
            for key in ('code_norm','ambient_norm','tanh_saturation','B_z_norm','B_d_norm','up1_original_increment_ratio','up3_original_increment_ratio','up1_readout_increment_ratio','up3_readout_increment_ratio','up1_feature_rms','up3_feature_rms'):diag.append(dict(origin='SOURCE',condition=k,mode=mode,quantity=key,**distribution([r[key] for r in rs if key in r])))
    write_csv(root/'DIAGNOSTICS.csv',diag)
    source_target=[]
    for a,b in (*PAIRS,*((k,'C0') for k in ('B_FULL_1','B_RESET_1','B_FULL_READOUT_025','B_RESET_READOUT_025'))):
        for ch in ('OD','OC','macro'):
            def sv(k):return next((r['Dice'] for r in sr if r['condition']==k and r['mode']=='ALL' and r['metric']=='hard' and r['channel']==ch),None)
            va,vb=sv(a),sv(b);ds=None if va is None or vb is None else va-vb
            for o in (0,1):
                dt=paired(allrows[a,o],allrows[b,o],channel=ch)['domain_equal_delta'] if (a,o) in allrows and (b,o) in allrows else None
                source_target.append(dict(left=a,right=b,channel=ch,order=o,source_four_mode_delta=ds,target_four_domain_delta=dt,opposite_sign=None if ds is None or dt is None else ds*dt<0))
    write_csv(root/'SOURCE_TARGET_DIRECTION.csv',source_target)
    # Read saved R10 traces only; absent residual diagnostics stay unavailable.
    core=Path(read(Path(c['previous_root'])/'RESOLVED_CONFIG.json')['previous_root']);oldr10=[]
    for o in (0,1):
        rs=[json.loads(x) for x in (core/'target'/f'GR_RET_EMA_o{o}'/'visits.jsonl').read_text().splitlines()]
        oldr10.append(dict(order=o,gain=distribution([r['gain'] for r in rs if 'gain' in r]),residual_relative_candidate=distribution([r['residual_relative_candidate'] for r in rs if 'residual_relative_candidate' in r]),note='Saved R10 logs only; no old target rerun. Residual support is the first eight basis coordinates, verified from execution-version Controller.act; this is an action-space limitation, not proof of the score gap root cause.'))
    save(root/'OLD_R10_DIAGNOSTICS.json',oldr10)
    attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))];ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(status=state['status'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),operations={k:sum(a['cost'].get(k,0) for a in attempts) for k in OPS},attempts=attempts,recovery_used=state['recovery_used'],disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()));save(root/'RESOURCE_LEDGER.json',ledger)
    get=lambda a,b:[next((r['delta'] for r in contrasts if r['left']==a and r['right']==b and r['order']==o and r['domain']=='ALL' and r['channel']=='macro'),None) for o in (0,1)]
    signals={k:dict(delta_vs_C0=get(k,'C0'),meets_retest_scale=False) for k in LEVELS if k!='B_ZERO'}
    for x in signals.values():xs=x['delta_vs_C0'];x['meets_retest_scale']=all(v is not None and v>0 for v in xs) and statistics.mean(xs)>=.005
    eligible=[k for k,x in signals.items() if x['meets_retest_scale']]
    complete=state['status']=='COMPLETE' and all(v=='COMPLETE' for v in state['jobs'].values())
    if not complete:
        for x in signals.values():x['meets_retest_scale']=None
    decision='Unavailable: fixed source/target matrix incomplete; no carrier decision or negative performance finding.' if not complete else ('Fixed nonzero intervention is a development hypothesis for separate frozen replication.' if eligible else 'No fixed nonzero condition meets the predeclared +0.5pp vs C0 retest scale; retain zero-update C0 and investigate carrier/readout validity before adding actor complexity.')
    conclusion=dict(matrix_complete=complete,zero_parity_source=z['passed'],zero_target_seals=parity,history_original_scale=get('B_RESET_1','B_FULL_1'),scale_FULL=get('B_FULL_READOUT_025','B_FULL_1'),scale_RESET=get('B_RESET_READOUT_025','B_RESET_1'),signals=signals,source_target_opposite_sign=[r for r in source_target if r['opposite_sign']],decision=decision,automatic_followon_authorized=False)
    save(root/'DECISION.json',conclusion)
    lines=[f'# {c["experiment_id"]}',f'Status: {state["status"]}; execution SHA {c["code_sha"]}.',f'New wall including preflight: {ledger["actual_wall_seconds"]:.2f}s; GPU-worker {ledger["gpu_worker_seconds"]:.2f}s; recovery {state["recovery_used"]}.','No training, optimizer update or VJP/JVP. Prior baselines are exact paired reuse. All new target scores unblinded only after terminal state.','One seed; two correlated orders, same contents. Unknown patient dependence; no independent-seed confidence interval. Source val is development evidence.','\n| condition | order | status/origin | OD % | OC % | domain macro % |','|---|---:|---|---:|---:|---:|']
    for r in table:lines.append('| '+r['condition']+' | '+str(r['order'])+' | '+r['status']+'/'+r['origin']+' | '+' | '.join('MISSING' if r[k] is None else f'{100*r[k]:.6f}' for k in ('Dice_OD','Dice_OC','Dice_macro'))+' |')
    lines+=['\n## Three bounded questions','1. Zero FiLM parity: source same-path repeat criterion, BN/parameter checks, compatible target probability seals and limits are in ZERO_PARITY.json.','2. Scale and history: fixed paired contrasts and the full-minus-reset scaling interaction are in TARGET_FACTORIAL.csv, including every domain and OD/OC. Norms describe actual injected increments; near-zero feature norms are flagged in private source/target traces. Scaling improvements below C0 are reduced modulation harm, not additional adaptation benefit.','3. Source/target direction: SOURCE_TARGET_DIRECTION.csv flags sign reversal without claiming which observer/basis/calibration/simulator component caused it. Source mode and target domain means are different populations.','\n```json',json.dumps(conclusion,indent=2),'```','The +0.5pp relative-C0 priority threshold is descriptive R&D, not statistical significance or clinical benefit. No extra alpha, training, seed or monitoring is authorized.','Old attribution/SUP/writer evidence is retained; its small future numerical effect did not establish positive target benefit. SUP_STATIC still uses the frozen B action/basis path; RET-versus-STATIC is not a writer-only experiment.','Last-quarter and switch-first32 records are composition-sensitive descriptions, not revisit/forgetting claims. Missing diagnostic quantities are NOT_RECORDED, never zero.','Anonymous summaries are intended for local commit only; private raw images/masks/probabilities, content IDs, credentials and host paths are not public artifacts.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
