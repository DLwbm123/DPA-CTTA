"""Independent CPU scalar closeout; no real image or source reader import."""
import json,math
from pathlib import Path
import numpy as np
from .plan import science,stream,matrix
from ..p2_analysis import validate_metric
from ..m1_analysis import paired
from ..host_diagnostic_analysis import channels,distribution

PAIRS=(('C_SENS','C'),('C_PER256','C'),('C_SENS','C_PER256'),('C_PCA_GLOBAL','C'),('C_PCA_REGION','C'),('C_PCA_REGION','C_PCA_GLOBAL'),('C_PCA_REGION','C_PCA_SHUFFLED'))
SUBSETS=('remaining_dev','legacy_dev','p1_extension_dev','all_dev')


def validate(rows,ordered,arm,order):
    if len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('coverage/duplicate')
    resets=0;age=0
    for i,(r,e) in enumerate(zip(rows,ordered),1):
        if any(r[k]!=e[k] for k in ('group_id','sample_id','domain','subset')) or r['global_visit']!=i or r['order']!=order or r['arm']!=arm:raise ValueError('ordered identity')
        if r['reset_before_current']:resets+=1;age=0
        age+=1
        if r['reset_count']!=resets or r['total_adam_calls']!=i or r['optimizer_steps_since_reset']!=age or r['segment_age']!=age:raise ValueError('reset/Adam lifecycle')
        if arm=='C_PER256' and r['reset_before_current']!=(i>1 and (i-1)%256==0):raise ValueError('period boundary')
        if arm not in ('C_PER256','C_SENS') and resets:raise ValueError('unexpected recovery')
        if r['counts']!=dict(forwards=11 if arm=='C_SENS' else 8,backwards=1,base_adam=1,perturb=0,restore=0):raise ValueError('physical calls')
        if r['prediction_fixed_before_label'] is not True:raise ValueError('evaluator isolation')
        if [m['channel'] for m in r['metrics']]!=['OD','OC']:raise ValueError('channels')
        for m in r['metrics']:validate_metric(m)
        p=r['pca']
        if arm.startswith('C_PCA_'):
            if p is None or any(v is not None and v>=i for v in p['basis_versions_used']):raise ValueError('PCA time')
            if sum(b['state_bytes'] for b in p['banks'])>=1024**2:raise ValueError('PCA memory')
        elif p is not None:raise ValueError('unexpected PCA')
        json.dumps(r,allow_nan=False)


def summarize(arms):
    result={}
    for d in dict.fromkeys(r['domain'] for r in arms['C']):
        selected={a:[r for r in rs if r['domain']==d] for a,rs in arms.items()}
        result[d]=dict(arms={a:{c:dict(dice_percent=distribution([100*m['dice'] for m in channels(rs,c)]),**({} if c=='macro' else dict(assd_defined=sum(m['assd'] is not None for m in channels(rs,c)),assd_undefined=sum(m['assd'] is None for m in channels(rs,c)),**{k:sum(m[k] for m in channels(rs,c)) for k in ('pred_empty','pred_full','gt_empty','gt_full')}))) for c in ('OD','OC','macro')} for a,rs in selected.items()},paired={a+'-'+b:{c:paired(channels(selected[a],c),channels(selected[b],c),c=='macro') for c in ('OD','OC','macro')} for a,b in PAIRS})
    means={a:float(np.mean([v['arms'][a]['macro']['dice_percent']['mean'] for v in result.values()])) for a in arms}
    return dict(domains=result,task_domain_macro_dice_percent=means,comparisons_pp={a+'-'+b:means[a]-means[b] for a,b in PAIRS})


def recompute(out,reg):
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU closeout must not initialize GPU')
    out=Path(out);all_rows={};physical=dict(records=0,forwards=0,backwards=0,adam=0);mechanism={};target={}
    for job in matrix()['jobs']:
        p=out/job['job_id'];done=json.loads((p/'completion.json').read_text())
        if done['status']!='TRAJECTORY_COMPLETE':raise ValueError('incomplete trajectory')
        rows=[json.loads(line) for line in (p/'records.jsonl').read_text().splitlines()];validate(rows,stream(reg,job['order']),job['arm'],job['order']);all_rows[(job['order'],job['arm'])]=rows
        physical['records']+=len(rows)
        for k,v in [('forwards','forwards'),('backwards','backwards'),('adam','base_adam')]:physical[k]+=sum(r['counts'][v] for r in rows)
        mechanism[job['job_id']]=dict(resets=rows[-1]['reset_count'],reset_visits=[r['global_visit'] for r in rows if r['reset_before_current']],host_seconds=sum(r['host_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),pca_ready_visits=sum(any(v is not None for v in r['pca']['basis_versions_used']) for r in rows if r['pca']),pca_last=None if not rows[-1]['pca'] else rows[-1]['pca']['banks'],subloss_mean=None if not rows[-1]['pca'] else float(np.mean([r['pca']['subloss'] for r in rows])))
    if physical!={k:science()['formal_budget'][k] for k in physical}:raise ValueError('matrix counts')
    for o in range(4):
        arms={a:all_rows[o,a] for a in science()['arms']}
        reference=arms['C']
        for arm,rows in arms.items():
            if any(any(m[k]!=n[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) for r,c in zip(rows,reference) for m,n in zip(r['metrics'],c['metrics'])):raise ValueError('cross-arm GT binding')
        target[str(o)]={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()}) for s in SUBSETS}
    pairs=[a+'-'+b for a,b in PAIRS];desc={p:float(np.mean([target[str(o)]['remaining_dev']['comparisons_pp'][p] for o in range(4)])) for p in pairs}
    secondary=secondary_controls(all_rows,reg)
    decision=assess(target,mechanism)
    result=dict(status='R1_EXPERIMENT_COMPLETE',physical=physical,target=target,descriptive_four_order_comparisons_pp=desc,mechanism=mechanism,secondary_A_C0=secondary,scientific_assessment=decision,scientific_status=decision['status'],limitations=['Exposed development data; orders share contents.','ASSD geometry not recomputed; recorded scalar cohorts only.','No automatic next run or combination.'])
    (out/'public_aggregate.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    (out/'R1_EXPERIMENT_REPORT.md').write_text('# R1 results\n\nR1_EXPERIMENT_COMPLETE\n\n'+json.dumps(dict(physical=physical,descriptive=desc,scientific_status=result['scientific_status']),indent=2)+'\n\nAll domain/channel/subset pairs and mechanism coverage are in public_aggregate.json. Historical A/C0 are identity-validated when present; unavailable controls are explicit. Matched-control and risk assessments are in the aggregate. No automatic next experiment.\n')
    return result


def secondary_controls(all_rows,reg):
    result={}
    for o in range(4):
        result[str(o)]={}
        for name,path in reg.get('historical_scalars',{}).get(str(o),{}).items():
            p=Path(path)
            if not p.is_file():result[str(o)][name]=dict(status='UNAVAILABLE');continue
            old=[json.loads(line) for line in p.read_text().splitlines()];lookup={r['group_id']:r for r in old};reference=all_rows[o,'C']
            if len(lookup)!=len(old) or set(lookup)!={r['group_id'] for r in reference}:raise ValueError('historical identity coverage')
            mapped=[lookup[r['group_id']] for r in reference]
            for r,v in zip(reference,mapped):
                if any(r[k]!=v[k] for k in ('sample_id','domain','subset')):raise ValueError('historical identity')
                for a,b in zip(r['metrics'],v['metrics']):
                    validate_metric(b)
                    if any(a[k]!=b[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')):raise ValueError('historical GT')
            result[str(o)][name]=dict(status='AVAILABLE',comparisons={arm:{subset:{d:{c:paired(channels([r for r in rows if r['domain']==d and (subset=='all_dev' or r['subset']==subset)],c),channels([r for r in mapped if r['domain']==d and (subset=='all_dev' or r['subset']==subset)],c),c=='macro') for c in ('OD','OC','macro')} for d in science()['orders'][o]} for subset in SUBSETS} for (order,arm),rows in all_rows.items() if order==o})
    return result


def assess(target,mechanism):
    cfg=science();s={a:[target[str(o)]['remaining_dev']['task_domain_macro_dice_percent'][a] for o in range(4)] for a in cfg['arms']};items={}
    for arm in cfg['arms'][1:]:
        delta=[a-b for a,b in zip(s[arm],s['C'])];domain={d:float(np.mean([target[str(o)]['remaining_dev']['domains'][d]['arms'][arm]['macro']['dice_percent']['mean']-target[str(o)]['remaining_dev']['domains'][d]['arms']['C']['macro']['dice_percent']['mean'] for o in range(4)])) for d in cfg['orders'][0]}
        job_ids=[j['job_id'] for j in matrix()['jobs'] if j['arm']==arm];active=any(mechanism[k]['resets']>0 for k in job_ids) if arm=='C_SENS' else any(mechanism[k]['pca_ready_visits']>0 for k in job_ids) if 'PCA' in arm else True
        practical=float(np.mean(delta))>=.5 and sum(v>0 for v in delta)>=3;risk=any(v < -2 for v in domain.values()) or min(s[arm])-min(s['C']) < -.5
        items[arm]=dict(mean_gain_pp=float(np.mean(delta)),positive_orders=sum(v>0 for v in delta),domain_mean_deltas=domain,worst_order_score_delta_pp=min(s[arm])-min(s['C']),active=active,practical_signal=practical,risk_warning=risk)
    mean=lambda a:float(np.mean(s[a]));routes=[]
    if items['C_SENS']['active'] and items['C_SENS']['practical_signal'] and mean('C_SENS')>mean('C_PER256'):routes.append('C_SENS')
    elif items['C_PER256']['practical_signal']:routes.append('C_PER256')
    if items['C_PCA_REGION']['active'] and items['C_PCA_REGION']['practical_signal'] and mean('C_PCA_REGION')>max(mean('C_PCA_GLOBAL'),mean('C_PCA_SHUFFLED')):routes.append('C_PCA_REGION')
    elif items['C_PCA_GLOBAL']['active'] and items['C_PCA_GLOBAL']['practical_signal']:routes.append('C_PCA_GLOBAL')
    status='SUPPORTED' if routes and not any(items[a]['risk_warning'] for a in routes) else 'MIXED' if routes or any(v['mean_gain_pp']>0 for v in items.values()) else 'NO_GAIN'
    return dict(status=status,arms=items,candidate_routes=routes,recommendation='Retain C; no sufficient matched evidence' if not routes else 'Independent candidates only; tail review and independent data remain necessary',combination_run_authorized=False,note='Frozen descriptive resource-allocation thresholds, not significance, clinical criteria, or execution gates. INACTIVE mechanisms are identified by active=false; no retuning.')
