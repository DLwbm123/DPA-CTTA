"""Unblind only terminal jobs; paired scalar analysis never reads target images."""
import csv,json,statistics,time
from collections import defaultdict
from pathlib import Path
from ..r10_12h_core.run import read,save,epoch,OPS
from .audit import distribution,gates
from .run import A_ARMS,B_ARMS

PAIRS=(('GR_RET_EMA','C0_CURRENT_STATS'),('GR_RET_EMA','B_PARENT_FULL'),('GR_RET_EMA_CONST_HALF','WARM_CONST_HALF'),('GR_RET_EMA','R10_RESET_ALL'),('GR_RET_EMA','GR_RET_EMA_CONST_HALF'),('GR_RET_EMA','R10_FORCE_WRITE'),('SUP_RET','SUP_RET_CONST_HALF'),('SUP_STATIC','GR_RET_EMA'),('SUP_RET','GR_RET_EMA'),('SUP_STATIC','G'),('SUP_RET','G'),('SUP_RET','SUP_STATIC'))

def write_csv(path,rows,fields=None):
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def primary(rows):return [r for r in rows if r['subset']=='remaining_dev']
def values(row,channel):
    m={x['channel']:x['dice'] for x in row['metrics']}
    return statistics.mean(m.values()) if channel=='macro' else m[channel]

def means(rows):
    cells=defaultdict(list)
    for r in primary(rows):
        for m in r['metrics']:cells[r['domain'],m['channel']].append(m['dice'])
    if len(cells)!=8:raise ValueError('four-domain OD/OC coverage')
    by_channel={ch:statistics.mean(statistics.mean(v) for (domain,k),v in cells.items() if k==ch) for ch in ('OD','OC')}
    return dict(Dice_OD=by_channel['OD'],Dice_OC=by_channel['OC'],Dice_macro=statistics.mean(by_channel.values()),Dice_pooled=statistics.mean(values(r,'macro') for r in primary(rows)),worst_domain=min(statistics.mean(statistics.mean(v) for (d,ch),v in cells.items() if d==domain) for domain in {d for d,ch in cells}))

def paired(a,b,domain='ALL',channel='macro'):
    if len(a)!=len(b) or any((x['visit'],x['content'],x['domain'],x['subset'])!=(y['visit'],y['content'],y['domain'],y['subset']) for x,y in zip(a,b)):raise ValueError('paired identity/visit/eligibility mismatch')
    pairs=[(x,y) for x,y in zip(a,b) if x['subset']=='remaining_dev' and (domain=='ALL' or x['domain']==domain)]
    diff=[values(x,channel)-values(y,channel) for x,y in pairs];by=defaultdict(list)
    for (x,y),d in zip(pairs,diff):by[x['domain']].append(d)
    main=statistics.mean(statistics.mean(v) for v in by.values())
    # Fixed last quarter of the original arrival sequence; composition is reported, not a causal forgetting measure.
    tail=[values(x,channel)-values(y,channel) for x,y in pairs if x['visit']>3*len(a)//4]
    return dict(n=len(diff),domain_equal_delta=main,paired_image_distribution=distribution(diff),positive_fraction=statistics.mean(v>0 for v in diff),negative_fraction=statistics.mean(v<0 for v in diff),zero_fraction=statistics.mean(v==0 for v in diff),tail_last_quarter_n=len(tail),tail_last_quarter_delta=statistics.mean(tail) if tail else None)

def report(c,state):
    if state['status']=='RUNNING' or any(v=='RUNNING' for v in state['jobs'].values()):raise ValueError('target scores still embargoed')
    root=Path(c['output_root']);old=Path(c['previous_root']);allrows={};table=[];cost=[];behavior=[]
    for reused,jobs,base in ((True,c['old_jobs'],old),(False,c['jobs'],root)):
        for j in jobs:
            status='REUSED_COMPLETE' if reused else state['jobs'][j['id']];p=base/'target'/j['id'];row=dict(method=j['arm'],order=j['order'],origin='REUSED' if reused else 'NEW',status=status,visits=None,principal=None,Dice_OD=None,Dice_OC=None,Dice_macro=None,Dice_pooled=None,worst_domain=None)
            if status in ('COMPLETE','REUSED_COMPLETE'):
                rows=[json.loads(line) for line in (p/'scalars.private.jsonl').read_text().splitlines()];receipt=read(p/'score_complete.json');m=c['manifests'][j['order']]
                if len(rows)!=m['visits'] or receipt['principal']!=sum(x['subset']=='remaining_dev' for x in rows):raise ValueError('report sealed coverage')
                allrows[j['arm'],j['order']]=rows;row.update(visits=len(rows),principal=receipt['principal'],**means(rows))
                traces=[json.loads(line) for line in (p/'visits.jsonl').read_text().splitlines()]
                for key in ('write','gain','state_norm','mass'):
                    vs=[r[key] for r in traces if key in r]
                    if vs:behavior.append(dict(kind='TARGET_'+key,method=j['arm'],order=j['order'],**(gates(vs) if key=='write' else distribution(vs))))
            table.append(row)
            history=read(base/'RESOURCE_LEDGER.json').get('attempts',[]) if reused else []
            attempts=[a for a in history if a['phase'] in ('online:'+j['id'],'score:'+j['id'])] if reused else [read(p) for p in (root/'attempts').glob('*') if p.name.startswith(('online_'+j['id']+'.','score_'+j['id']+'.'))]
            cost.append(dict(task=j['id'],origin='REUSED' if reused else 'NEW',status=status,wall_seconds=sum(a['wall_seconds'] for a in attempts if a.get('wall_seconds') is not None) if attempts else None,gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts) if attempts else None,attempts=len(attempts) if attempts else None,note='historical execution cost; no new trajectory' if reused else 'new online plus independent scoring'))
    write_csv(root/'RESULTS.csv',table)
    comparisons=[]
    for a,b in PAIRS:
        for order in (0,1):
            if (a,order) not in allrows or (b,order) not in allrows:
                comparisons.append(dict(left=a,right=b,order=order,domain='ALL',channel='macro',status='MISSING',delta=None));continue
            domains=['ALL']+sorted({r['domain'] for r in allrows[a,order]})
            for domain in domains:
                for channel in ('OD','OC','macro'):
                    p=paired(allrows[a,order],allrows[b,order],domain,channel)
                    comparisons.append(dict(left=a,right=b,order=order,domain=domain,channel=channel,status='COMPLETE',delta=p['domain_equal_delta'],n=p['n'],**{f'paired_{k}':v for k,v in p['paired_image_distribution'].items() if k!='n'},positive_fraction=p['positive_fraction'],negative_fraction=p['negative_fraction'],zero_fraction=p['zero_fraction'],tail_last_quarter_n=p['tail_last_quarter_n'],tail_last_quarter_delta=p['tail_last_quarter_delta']))
    write_csv(root/'ATTRIBUTION.csv',comparisons)
    audit=read(root/'WRITER_AUDIT.private.json')
    for branch,item in audit['parameter_changes'].items():behavior.append(dict(kind='OLD_WARM_TO_POST_PARAMETERS',method=branch,**item))
    for name,item in audit['historical_training'].items():
        if isinstance(item,dict):behavior.append(dict(kind='OLD_RL_SIGNAL',method=name,**item))
        elif isinstance(item,list):
            for i,value in enumerate(item):behavior.append(dict(kind='OLD_RL_SIGNAL',method=name,epoch=i,**value))
        else:behavior.append(dict(kind='OLD_RL_SIGNAL',method=name,status=item))
    for row in audit['source_supplement']:
        for i,g in enumerate(row['row']['branch_gradients']):
            for branch,v in g.items():behavior.append(dict(kind='NEW_SOURCE_COPY_GRADIENT_NO_UPDATE',method=branch,round=row['schedule_round_zero_based'],epoch=i,**v))
        for a in row['advantage']:behavior.append(dict(kind='NEW_SOURCE_COPY_ADVANTAGE',round=row['schedule_round_zero_based'],L2=a['L2'],max_abs=a['max_abs']))
    for method in c['training']:
        path=root/'source'/method/'fit/physical.jsonl'
        if path.exists():
            rs=[json.loads(s) for s in path.read_text().splitlines()]
            for branch in ('use','write'):behavior.append(dict(kind='NEW_SUP_GRADIENT',method=method,branch=branch,**distribution([g[branch]['grad_norm'] for r in rs for g in r['branch_gradients']])))
            endpoint=root/'source'/method/'complete.json'
            if endpoint.exists():
                import torch
                from ..r9_current_first.storage import load_torch
                warm=c['endpoints']['WARM']['artifact'];new=read(endpoint)['artifact']
                w=load_torch(old/'source/WARM'/warm['file'],warm['sha256'])['actor'];n=load_torch(root/'source'/method/new['file'],new['sha256'])['actor']
                for branch in ('use','write'):
                    before=torch.cat([v.flatten().double() for k,v in w.items() if k.startswith(branch+'.')]);after=torch.cat([v.flatten().double() for k,v in n.items() if k.startswith(branch+'.')]);d=after-before
                    behavior.append(dict(kind='NEW_WARM_TO_SUP_PARAMETERS',method=method,branch=branch,L2=float(d.norm()),max_abs=float(d.abs().max()),changed_elements=int((d!=0).sum())))

    write_csv(root/'WRITER_DIAGNOSTICS.csv',behavior);save(root/'WRITER_DIAGNOSTICS.json',dict(audit=audit,rows=behavior))
    counter=read(root/'SOURCE_COUNTERFACTUAL.private.json') if (root/'SOURCE_COUNTERFACTUAL.private.json').exists() else dict(status='NOT_RUN',rows=[])
    cf=[]
    for r in counter['rows']:
        for horizon,hr in r['horizons'].items():
            for w,changes in hr['delta_vs_half'].items():
                for i,ch in enumerate(('OD','OC')):cf.append(dict(episode=r['episode'],mode=r['mode'],horizon=int(horizon),current_writer=float(w),reference_writer=.5,channel=ch,hard_Dice_delta=changes['hard'][i],soft_Dice_delta=changes['soft'][i],probability_MAE=hr['probability_and_masks'][w]['probability_MAE'][i],mask_flip_fraction=hr['probability_and_masks'][w]['mask_flip_fraction'][i],commit_m_L2=r['commit_delta'][w]['m'],commit_q_L2=r['commit_delta'][w]['q'],commit_h_abs=r['commit_delta'][w]['h'],current_use_max_error=r['current_use_max_error'],finite_candidate_best_gain=hr['finite_candidate_best_gain']))
    write_csv(root/'SOURCE_COUNTERFACTUAL.csv',cf,fields=None if cf else ['status']);save(root/'SOURCE_COUNTERFACTUAL.json',counter)
    attempts=[read(p) for p in sorted((root/'attempts').glob('*.json'))]
    for a in attempts:
        if not a['phase'].startswith(('online:','score:')):cost.append(dict(task=a['phase'],origin='NEW',status=a['status'],wall_seconds=a.get('wall_seconds'),gpu_worker_seconds=a['cost'].get('gpu_seconds',0),attempts=1,note='preflight probes are not formal training' if a['phase']=='preflight' else None))
    for m,s in state['training'].items():
        if s!='COMPLETE' and not any(a['phase']=='train:'+m for a in attempts):cost.append(dict(task='train:'+m,origin='NEW',status=s,wall_seconds=None,gpu_worker_seconds=None,attempts=None))
    write_csv(root/'COST_AND_STATUS.csv',cost)
    ledger=read(root/'RESOURCE_LEDGER.json');ledger.update(status=state['status'],actual_wall_seconds=time.time()-epoch(c['origin']['T0']),gpu_worker_seconds=sum(a['cost'].get('gpu_seconds',0) for a in attempts),operations={k:sum(a['cost'].get(k,0) for a in attempts) for k in OPS},attempts=attempts,recovery_used=state['recovery_used'],disk_bytes=sum(p.stat().st_size for p in root.rglob('*') if p.is_file()));save(root/'RESOURCE_LEDGER.json',ledger)
    def delta(a,b):return [next((r['delta'] for r in comparisons if r['left']==a and r['right']==b and r['order']==o and r['domain']=='ALL' and r['channel']=='macro'),None) for o in (0,1)]
    answers={}
    d=delta('GR_RET_EMA','C0_CURRENT_STATS');parent=delta('GR_RET_EMA','B_PARENT_FULL');use=delta('GR_RET_EMA_CONST_HALF','WARM_CONST_HALF')
    answers['where_improvement_comes_from']=dict(R10_minus_C0=d,R10_minus_B_PARENT=parent,post_minus_WARM_fixed_writer=use,interpretation='Compare matched normalization, carrier and post-use controls; do not attribute R10-N to RL without these controls.')
    answers['does_writer_change_future']=dict(source_probability_change_max=max((r['probability_MAE'] for r in cf),default=None),source_mask_flip_max=max((r['mask_flip_fraction'] for r in cf),default=None),learned_SUP_writer_delta=delta('SUP_RET','SUP_RET_CONST_HALF'),caveat='Source intervention consequences do not establish deployable target benefit.')
    candidates=[(r['method'],r['Dice_macro']) for r in table if r['origin']=='NEW' and r['Dice_macro'] is not None];best={m:statistics.mean(v for k,v in candidates if k==m) for m in {k for k,v in candidates}}
    writer=delta('SUP_RET','SUP_RET_CONST_HALF');sup=delta('SUP_RET','GR_RET_EMA');static=delta('SUP_STATIC','GR_RET_EMA')
    def show(xs):return ', '.join('MISSING' if v is None else f'{v*100:+.5f}pp' for v in xs)
    normalization_sufficient=all(v is not None and v<=0 for v in d)
    answers['where_improvement_comes_from']['answer']=f'R10-C0: {show(d)}; R10-parent B: {show(parent)}; post-WARM at fixed writer: {show(use)}. '+('Matched current-statistics C0 already matches or exceeds R10, so R10-N does not establish a policy/RL gain.' if normalization_sufficient else 'Read these matched controls together; effects are not an additive causal decomposition.')
    prob=answers['does_writer_change_future']['source_probability_change_max'];flip=answers['does_writer_change_future']['source_mask_flip_max']
    answers['does_writer_change_future']['answer']='INSUFFICIENT SOURCE EVIDENCE' if prob is None else f'Fixed-source writer interventions {"do" if prob>0 else "do not"} change future probabilities (max recorded channel-mean absolute difference {prob:.8g}; mask flip fraction {flip:.8g}). Learned SUP writer target advantage is {show(writer)}; numerical change alone is not useful target performance.'
    writer_signal=all(v is not None and v>=.002 for v in writer)
    recipe_signal=all(v is not None and v>=.005 for v in sup)
    static_signal=all(v is not None and v>=.005 for v in static)
    available_sup=[m for m in ('SUP_STATIC','SUP_RET') if all((m,o) in allrows for o in (0,1))]
    c0_dominates=bool(available_sup) and all(means(allrows[m,o])['Dice_macro']<means(allrows['C0_CURRENT_STATS',o])['Dice_macro'] for m in available_sup for o in (0,1)) if all(('C0_CURRENT_STATS',o) in allrows for o in (0,1)) else False
    best_method=max(best,key=best.get) if best else None
    if c0_dominates and normalization_sufficient:direction='Prioritize current-image normalization/carrier validity; this run does not justify adding further policy complexity on frozen B.'
    elif best_method=='SUP_STATIC':direction='Prioritize the simpler supervised static use controller; do not claim persistent memory/writer contribution.'
    elif writer_signal:direction='SUP_RET learned writer meets the predeclared descriptive signal; prioritize independent replication before a separate RL credit-assignment study.'
    elif recipe_signal:direction='Prioritize supervised use learning and a fixed writer; supervised learned-writer gain did not meet the predeclared two-order criterion.'
    elif static_signal:direction='Prioritize supervised static use learning for replication; no learned-writer signal is established.'
    else:direction='Retain the negative or incomplete evidence and analyze the matched current-image/carrier controls; do not blindly extend the same RL recipe.'
    answers['next_research_line']=dict(answer=direction,best_new_two_order_mean=best_method,SUP_RET_research_signal=recipe_signal,SUP_STATIC_research_signal=static_signal,SUP_writer_retest_signal=writer_signal,automatic_followon_authorized=False)
    save(root/'INTERPRETATION.json',answers)
    lines=[f'# {c["experiment_id"]}',f'Status: {state["status"]}; execution SHA {c["code_sha"]}.',f'Sealed comparisons were unblinded only after terminal run state. New training branches: {c["training"]}.',f'Actual new wall: {ledger["actual_wall_seconds"]:.2f}s; GPU-worker: {ledger["gpu_worker_seconds"]:.2f}s; recovery used: {state["recovery_used"]}.','One policy seed; two correlated orders with the same contents. Patient dependence is unknown; no independent-seed confidence interval or target-selected checkpoint.','SUP versus GR differs in objective and compute, so it is a training-recipe comparison, not a pure estimator causal experiment.','Last-quarter summaries use the fixed final 256 arrivals and may differ in domain composition. No revisits are invented.','\n| method | order | origin/status | OD % | OC % | domain macro % | pooled % |','|---|---:|---|---:|---:|---:|---:|']
    for r in table:lines.append('| '+r['method']+' | '+str(r['order'])+' | '+r['origin']+'/'+r['status']+' | '+' | '.join('MISSING' if r[k] is None else f'{100*r[k]:.5f}' for k in ('Dice_OD','Dice_OC','Dice_macro','Dice_pooled'))+' |')
    lines+=['\n## Attribution (percentage points)','| comparison | order | macro delta pp |','|---|---:|---:|']
    for r in comparisons:
        if r['domain']=='ALL' and r['channel']=='macro':lines.append(f'| {r["left"]} - {r["right"]} | {r["order"]} | '+('MISSING' if r['delta'] is None else f'{100*r["delta"]:+.6f}')+' |')
    lines+=['\n## Three bounded answers','```json',json.dumps(answers,indent=2),'```','The +0.5pp recipe and +0.2pp learned-writer thresholds are descriptive R&D signals, not statistical significance or clinical standards. Decisions are post-run only; no additional run is authorized.','Raw permissible scalar records and checkpoints remain in the private output root. Four requested tables: ATTRIBUTION.csv, WRITER_DIAGNOSTICS.csv, SOURCE_COUNTERFACTUAL.csv, COST_AND_STATUS.csv.']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
