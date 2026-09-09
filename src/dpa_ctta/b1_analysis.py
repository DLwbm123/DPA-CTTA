"""B1 scalar reconstruction, matched frozen P2 controls, and descriptive reports."""
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

OLD=('N','A','EA','O2','D4');ARMS=OLD+('C','G')
PAIRS=(('G','C'),('C','A'),('G','A'),('C','N'),('G','N'),('G','O2'),('G','D4'))
COMPLETE='B1_MATCHED_BASELINE_COMPARISON_COMPLETE'


def old_records(reg,order):
    old=Path(reg['p2_directory']);stream=expected(reg,'fundus',order)
    arms={a:read_lines(old/f'fundus_{order}_{a}.jsonl') for a in P2_ARMS}
    validate_bundle(arms,stream,'fundus',order)
    return {a:arms[a] for a in OLD}


def validate_rows(rows,stream,order,arm,old):
    if len(rows)!=len(stream) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('B1 coverage')
    for i,(r,e,n) in enumerate(zip(rows,stream,old)):
        values=dict(task='fundus',order=order,arm=arm,visit=i+1,adam_step=i+1,domain=e['domain'],subset=e['subset'],sample_id=e['sample_id'],group_id=e['group_id'])
        if any(r[k]!=v for k,v in values.items()):raise ValueError('B1 identity/order/state')
        if r['counts']!=expected_counts(arm) or not r['prediction_fixed_before_label'] or not r['frozen_parameters_checked']:raise ValueError('B1 counters/invariants')
        if not 0<=r['lr']<=.00010001 or (arm=='C' and r['lr']!=1e-4):raise ValueError('B1 learning rate')
        if arm=='G':
            c=r['diagnostics']['cosine']
            if not -1.00001<=c<=1.00001 or abs(r['lr']-1e-4*(c+1)**2/4)>1e-10:raise ValueError('B1 cosine LR mapping')
        if [m['channel'] for m in r['metrics']]!=['OD','OC']:raise ValueError('B1 channels')
        for m,ref in zip(r['metrics'],n['metrics']):
            validate_metric(m)
            if any(m[k]!=ref[k] for k in ['gt_pixels','total_pixels','gt_empty','gt_full']):raise ValueError('B1 evaluator/GT identity')
        json.dumps(r,allow_nan=False)
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
        item['paired']={a+'-'+b:{c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in ['OD','OC','macro']} for a,b in PAIRS}
    out['task_domain_macro_dice_percent']={a:float(np.mean([out['domains'][d]['arms'][a]['macro']['dice_percent']['mean'] for d in domains])) if domains else None for a in arms}
    v=out['task_domain_macro_dice_percent'];out['task_comparisons_pp']={a+'-'+b:v[a]-v[b] if domains else None for a,b in PAIRS}
    return out


def recompute(out,reg,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='B1_RUN_COMPLETE':raise ValueError('incomplete run')
    result=dict(status=COMPLETE,execution_commit=receipt['commit'],coverage=reg['tasks']['fundus']['counts'],target={},limitations=reg['limitations'])
    totals=dict(records=0,forwards=0,backwards=0,base_adam=0,perturb=0,restore=0);cost={}
    for order in [0,1]:
        stream=expected(reg,'fundus',order);arms=old_records(reg,order)
        for arm in ['C','G']:
            rows=arms[arm]=read_lines(out/f'fundus_{order}_{arm}.jsonl');validate_rows(rows,stream,order,arm,arms['N'])
            totals['records']+=len(rows)
            for k in totals:
                if k!='records':totals[k]+=sum(r['counts'][k] for r in rows)
            cost[f'{arm}_{order}']=dict(records=len(rows),counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},host_seconds=sum(r['host_seconds'] for r in rows),pipeline_seconds=sum(r['pipeline_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),lr=distribution([r['lr'] for r in rows]),zero_lr=sum(r['lr']==0 for r in rows))
        result['target']['order'+str(order)]={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()}) for s in SUBSETS}
    if totals!=reg['formal_budget'] or totals!=done['progress']:raise ValueError('B1 independent total counts')
    smoke=json.loads((out/'smoke.completion.json').read_text());cpu=json.loads((out/'CPU_validation.json').read_text())
    old_cost=json.loads((Path(reg['p2_directory'])/'execution_audit.json').read_text())['cost']
    audit=dict(status=COMPLETE,execution_commit=receipt['commit'],formal=totals,base_adam_including_smoke=totals['base_adam']+16,new_source_DD_outer_training=0,smoke={k:smoke[k] for k in ['status','base_adam','evidence','gpu_seconds','source_parity','extra_source_parity_forwards','exit_code']},cpu=cpu,cost=cost,old_P2_cost={k:v for k,v in old_cost.items() if k.startswith('fundus_') and k.rsplit('_',1)[1] in OLD},trainable=done['trainable'],gpu_seconds=done['gpu_seconds'],scalar_recompute=True,ASSD_pixel_recomputed=False,exit_code=0)
    private_json(out/'public_aggregate.json',result);private_json(out/'execution_audit.json',audit)
    lines=['# B1 matched GraTa continuous-stream baseline comparison','',COMPLETE,'','Execution commit: '+receipt['commit'],'','C/G are C-CTTA/G-CTTA on the P2 segmentation checkpoint and continuous protocol, not a complete original-paper reproduction.']
    for s in SUBSETS:
        lines+=['','## '+s,'','| Order | Groups | '+' | '.join(ARMS)+' |','|---|---:|'+'---:|'*len(ARMS)]
        for order,subsets in result['target'].items():
            r=subsets[s];lines+=['| '+order+' | '+str(r['groups'])+' | '+' | '.join(f'{r["task_domain_macro_dice_percent"][a]:.6f}' for a in ARMS)+' |']
        lines+=['','| Order | '+' | '.join(a+'-'+b for a,b in PAIRS)+' |','|---|'+'---:|'*len(PAIRS)]
        for order,subsets in result['target'].items():lines+=['| '+order+' | '+' | '.join(f'{subsets[s]["task_comparisons_pp"][a+"-"+b]:+.6f}' for a,b in PAIRS)+' |']
    lines+=['','All domain/channel paired distributions, signs, worst decile/single and ASSD common-valid cohorts are in public_aggregate.json. No OD/OC macro ASSD. No patient independence, statistical significance, novelty, clinical safety or untouched-test claim.','C/G change the full adaptation scheme relative to A. G-C measures the combined published perturbation and dynamic-LR rule, not either component separately. Published entropy omits the Bernoulli complement term; preserved deliberately.','Old P2 controls bind the exact checkpoint, evaluator, content and order. Old proxy arms require condensed images/source masks at deployment; C/G do not rehearse source samples. No new proxy/source training. C/G do not use fixed source ensembling.','Cost includes 8 C / 9 G model forwards per image, 1 / 2 backwards and one base Adam; G perturb and restore are counted separately. Old P2 peaks are shared-schedule measurements, not isolated per-arm peaks.','CPU reconstruction rereads every scalar record and restores Dice from pixel counts; it cannot recompute discarded pixel ASSD. Source file unchanged and only registered BN affine parameters adapt.','Engineering completion does not establish usefulness. No automatic further run, parameter search or DD continuation.','']
    (out/'B1_EXPERIMENT_REPORT.md').write_text('\n'.join(lines));private_json(out/'verification.json',dict(status=COMPLETE,formal=totals,exit_code=0))
    if torch.cuda.is_initialized():raise ValueError('CPU recompute initialized CUDA')
