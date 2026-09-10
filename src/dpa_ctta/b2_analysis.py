"""B2 scalar reconstruction, matched frozen P2 controls, and descriptive reports."""
import json
from pathlib import Path
import numpy as np
import torch
from .p2_analysis import validate_metric,validate_bundle
from .p2_data import ARMS as P2_ARMS,SUBSETS,expected
from .p1_analysis import evaluate
from .m1_analysis import paired,read_lines
from .host_diagnostic_analysis import channels,distribution
from .host_diagnostic_run import private_json
from .b1_host import expected_counts
from .b1_analysis import old_records as p2_records,validate_rows as validate_b1

OLD=('N','A','EA','O2','D4');ARMS=OLD+('C','G','U','S','I')
PAIRS=(('I','C'),('I','U'),('I','S'),('U','C'),('S','C'),('I','A'),('I','G'),('I','O2'))
COMPLETE='B2_INTERVAL_CONSISTENCY_COMPLETE'


def old_records(reg,order):
    arms=p2_records(reg,order);stream=expected(reg,'fundus',order)
    for arm in ('C','G'):
        rows=arms[arm]=read_lines(Path(reg['b1_directory'])/f'fundus_{order}_{arm}.jsonl')
        validate_b1(rows,stream,order,arm,arms['N'])
    return arms


def validate_rows(rows,stream,order,arm,old):
    if len(rows)!=len(stream) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('B2 coverage')
    for i,(r,e,n) in enumerate(zip(rows,stream,old)):
        values=dict(task='fundus',order=order,arm=arm,visit=i+1,adam_step=i+1,domain=e['domain'],subset=e['subset'],sample_id=e['sample_id'],group_id=e['group_id'])
        if any(r[k]!=v for k,v in values.items()):raise ValueError('B2 identity/order/state')
        if r['counts']!=expected_counts('C') or not r['prediction_fixed_before_label'] or not r['frozen_parameters_checked']:raise ValueError('B2 counters/invariants')
        if r['lr']!=1e-4:raise ValueError('B2 learning rate')
        d=r['diagnostics']
        if d['bn_gradient_l2']<0 or d['adam_update_l2']<0 or d['zero_gradient']!=(d['bn_gradient_l2']==0):raise ValueError('B2 gradient/update diagnostic')
        if [c['channel'] for c in d['channels']]!=['OD','OC']:raise ValueError('B2 diagnostic channels')
        for c in d['channels']:
            if sum(v['denominator'] for v in c['partitions'].values())!=c['pixels'] or c['partitions']['1']['denominator']!=c['q_foreground']:raise ValueError('B2 partitions')
            if any(not 0<=v['active']<=v['denominator'] for v in c['partitions'].values()):raise ValueError('B2 active counts')
            if not 0<=c['crosses_half']<=c['pixels'] or not 0<=c['interval_width_mean']<=1:raise ValueError('B2 interval width')
            for key in ('s','r'):
                v=c[key]
                if not (0<=v['mean']<=v['max']<=.500001 and 0<=v['p50']<=v['p90']<=v['max']):raise ValueError('B2 radius stats')
        if [m['channel'] for m in r['metrics']]!=['OD','OC']:raise ValueError('B2 channels')
        for m,ref in zip(r['metrics'],n['metrics']):
            validate_metric(m)
            if any(m[k]!=ref[k] for k in ['gt_pixels','total_pixels','gt_empty','gt_full']):raise ValueError('B2 evaluator/GT identity')
        json.dumps(r,allow_nan=False)
        if any(c['pixels']!=m['total_pixels'] for c,m in zip(d['channels'],r['metrics'])):raise ValueError('B2 diagnostic grid')
    return True


def summarize(arms):
    domains=list(dict.fromkeys(r['domain'] for r in arms['N']));out=dict(groups=len(arms['N']),domains={},task_domain_macro_dice_percent={},task_comparisons_pp={})
    for domain in domains:
        rows={a:[r for r in rs if r['domain']==domain] for a,rs in arms.items()};item=out['domains'][domain]=dict(arms={},paired={})
        for a,rs in rows.items():
            item['arms'][a]={}
            for c in ['OD','OC','macro']:
                ms=channels(rs,c);v=dict(dice_percent=distribution([100*m['dice'] for m in ms]))
                if c!='macro':
                    valid=[m['assd'] for m in ms if m['assd'] is not None]
                    v.update(assd_conditional_mean_px=float(np.mean(valid)) if valid else None,assd_defined=len(valid),assd_undefined=len(ms)-len(valid),**{k+'_count':sum(m[k] for m in ms) for k in ['gt_empty','gt_full','pred_empty','pred_full']})
                item['arms'][a][c]=v
        item['interval_diagnostics']={}
        for a in ('U','S','I'):
            item['interval_diagnostics'][a]={}
            for c in ('OD','OC'):
                ds=[next(v for v in r['diagnostics']['channels'] if v['channel']==c) for r in rows[a]]
                item['interval_diagnostics'][a][c]=dict(active_fraction=distribution([sum(v['active'] for v in d['partitions'].values())/d['pixels'] for d in ds]),width_mean=distribution([d['interval_width_mean'] for d in ds]),s_mean=distribution([d['s']['mean'] for d in ds]),r_mean=distribution([d['r']['mean'] for d in ds]),pseudo_foreground_fraction=distribution([d['q_foreground']/d['pixels'] for d in ds]))
        item['paired']={a+'-'+b:{c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in ['OD','OC','macro']} for a,b in PAIRS}
    out['task_domain_macro_dice_percent']={a:float(np.mean([out['domains'][d]['arms'][a]['macro']['dice_percent']['mean'] for d in domains])) if domains else None for a in arms}
    v=out['task_domain_macro_dice_percent'];out['task_comparisons_pp']={a+'-'+b:v[a]-v[b] if domains else None for a,b in PAIRS}
    return out


def recompute(out,reg,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='B2_RUN_COMPLETE':raise ValueError('incomplete run')
    result=dict(status=COMPLETE,execution_commit=receipt['commit'],coverage=reg['tasks']['fundus']['counts'],target={},limitations=reg['limitations'])
    totals=dict(records=0,forwards=0,backwards=0,base_adam=0,perturb=0,restore=0);cost={}
    for order in [0,1]:
        stream=expected(reg,'fundus',order);arms=old_records(reg,order)
        for arm in ['U','S','I']:
            rows=arms[arm]=read_lines(out/f'fundus_{order}_{arm}.jsonl');validate_rows(rows,stream,order,arm,arms['N'])
            totals['records']+=len(rows)
            for k in totals:
                if k!='records':totals[k]+=sum(r['counts'][k] for r in rows)
            cost[f'{arm}_{order}']=dict(records=len(rows),counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},host_seconds=sum(r['host_seconds'] for r in rows),pipeline_seconds=sum(r['pipeline_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),lr=distribution([r['lr'] for r in rows]),zero_gradient=sum(r['diagnostics']['zero_gradient'] for r in rows),bn_gradient_l2=distribution([r['diagnostics']['bn_gradient_l2'] for r in rows]),adam_update_l2=distribution([r['diagnostics']['adam_update_l2'] for r in rows]),interval_loss=distribution([r['diagnostics']['interval_loss'] for r in rows]),point_bce=distribution([r['diagnostics']['point_bce'] for r in rows]))
        result['target']['order'+str(order)]={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()}) for s in SUBSETS}
    if totals!=reg['formal_budget'] or totals!=done['progress']:raise ValueError('B2 independent total counts')
    primary=[result['target']['order'+str(o)]['remaining_dev'] for o in (0,1)]
    deltas=[r['task_comparisons_pp']['I-C'] for r in primary]
    overall=sum(deltas)/2
    refuge=[r['domains']['REFUGE_Valid']['paired']['I-C'] for r in primary]
    origa=[r['domains']['ORIGA']['paired']['I-C']['macro']['dice_delta_pp']['mean'] for r in primary]
    # ASSD is a common-valid descriptive cohort, never a fabricated zero for undefined maps.
    assd=[r['OC']['assd_delta_mean_px'] for r in refuge]
    robust=all(x>=-.25 for x in deltas) and all(r['macro']['dice_delta_pp']['mean']>=2 for r in refuge) and all(x is not None and x<=0 for x in assd) and all(x>=-2 for x in origa)
    spatial=all(r['task_comparisons_pp'][pair]>0 for r in primary for pair in ('I-U','I-S'))
    result['prespecified_interpretation']=dict(two_order_mean_I_minus_C_pp=overall,overall_gain_scale_met=overall>=.5,robustness_tradeoff_scale_met=robust,I_above_U_and_S_both_orders=spatial,spatial_necessity_established=False,reason='Descriptive development comparison only; positive matched comparisons can support a candidate, not establish necessity.',REFUGE_Valid_OC_common_valid_ASSD_delta_px=assd,ORIGA_I_minus_C_pp=origa)
    smoke=json.loads((out/'smoke.completion.json').read_text());cpu=json.loads((out/'CPU_validation.json').read_text())
    old_cost=json.loads((Path(reg['p2_directory'])/'execution_audit.json').read_text())['cost']
    audit=dict(status=COMPLETE,execution_commit=receipt['commit'],formal=totals,base_adam_including_smoke=totals['base_adam']+20,new_source_DD_outer_training=0,smoke={k:smoke[k] for k in ['status','base_adam','evidence','gpu_seconds','physical','exit_code']},cpu=cpu,cost=cost,old_B1_cost=json.loads((Path(reg['b1_directory'])/'execution_audit.json').read_text())['cost'],forwards_including_smoke=totals['forwards']+160,backwards_including_smoke=totals['backwards']+20,old_P2_cost={k:v for k,v in old_cost.items() if k.startswith('fundus_') and k.rsplit('_',1)[1] in OLD},trainable=done['trainable'],gpu_seconds=done['gpu_seconds'],scalar_recompute=True,ASSD_pixel_recomputed=False,exit_code=0)
    private_json(out/'public_aggregate.json',result);private_json(out/'execution_audit.json',audit)
    lines=['# B2 interval consistency experiment','',COMPLETE,'','Execution commit: '+receipt['commit'],'','U/S/I use partition-mean, partition-shuffled, and spatial six-view population standard-deviation radii. C/G and P2 controls are reused, identity-matched historical trajectories.']
    lines+=['','Prespecified descriptive scales: '+json.dumps(result['prespecified_interpretation'],ensure_ascii=False),'']
    for s in SUBSETS:
        lines+=['','## '+s,'','| Order | Groups | '+' | '.join(ARMS)+' |','|---|---:|'+'---:|'*len(ARMS)]
        for order,subsets in result['target'].items():
            r=subsets[s];lines+=['| '+order+' | '+str(r['groups'])+' | '+' | '.join(f'{r["task_domain_macro_dice_percent"][a]:.6f}' for a in ARMS)+' |']
        lines+=['','| Order | '+' | '.join(a+'-'+b for a,b in PAIRS)+' |','|---|'+'---:|'*len(PAIRS)]
        for order,subsets in result['target'].items():lines+=['| '+order+' | '+' | '.join(f'{subsets[s]["task_comparisons_pp"][a+"-"+b]:+.6f}' for a,b in PAIRS)+' |']
    lines+=['','All domain/channel paired distributions, signs, worst decile/single and ASSD common-valid cohorts are in public_aggregate.json. No macro ASSD.','All subsets are development data, including the historically named remaining_dev. Orders and subsets are not independent replicates. No significance, conformal coverage, clinical safety, novelty or SOTA claim.','U/S/I retain 41 BN layers and 19,136 affine scalars; eight forwards, one backward and one native Adam step per image. Zero new gradient may still move parameters due to Adam history.','CPU reconstruction checks every scalar record and Dice pixel counts; discarded ASSD maps cannot be independently recomputed. Optional q-error dispersion bins were not collected; no extra inference is authorized for them.','A/N/G/O2/D4 are historical context, not teachers or output fusion. No new source/DD/selector training, no radius/LR search, and no automatic next experiment.','Engineering completion does not establish the interval hypothesis. Compare I-C, I-U and I-S before any candidate decision.','']
    (out/'B2_EXPERIMENT_REPORT.md').write_text('\n'.join(lines));private_json(out/'verification.json',dict(status=COMPLETE,formal=totals,exit_code=0))
    if torch.cuda.is_initialized():raise ValueError('CPU recompute initialized CUDA')
