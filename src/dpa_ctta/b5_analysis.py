"""CPU-only ordered scalar validation and complete fixed-policy comparison."""
import json,math
from pathlib import Path
import numpy as np
import torch
from .b5_data import BUDGET,SMOKE,stream,old_records,map_rows
from .b5_host import policy,SCORE_HEAD_BN
from .p2_data import SUBSETS
from .p2_analysis import validate_metric
from .m1_analysis import read_lines,paired
from .host_diagnostic_analysis import distribution
from .host_diagnostic_run import private_json

COMPLETE='B5_SCORE_HEAD_PRESERVATION_COMPLETE'
PAIRS=(('H','F'),('H','C'),('F','C'),('H0','C0'),('H','H0'),('H','A'),('H','EA'),('H','N'))


def validate_rows(rows,ordered,arm,order,n):
    if len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('B5 coverage')
    for i,(r,e,ref) in enumerate(zip(rows,ordered,n)):
        want=dict(task='polyp',arm=arm,order=order,visit=i+1,**{k:e[k] for k in ('domain','subset','group_id','sample_id')})
        if any(r[k]!=v for k,v in want.items()):raise ValueError('B5 ordered identity')
        zero=arm=='H0';update=int(not zero)
        if r['counts']!=dict(forwards=1 if zero else 8,backwards=update,base_adam=update,perturb=0,restore=0) or r['adam_step']!=(0 if zero else i+1):raise ValueError('B5 update lifecycle')
        if r['layer_policy']!=policy(arm) or not all(r[k] for k in ('prediction_fixed_before_label','frozen_parameters_checked','head_state_checked')):raise ValueError('B5 layer policy')
        if r['prediction_origin']!=('canonical_stateless' if zero else 'continuous_update') or r['lr']!=(None if zero else 1e-4):raise ValueError('B5 forward origin')
        if zero and (not r['stateless_checked'] or r['optimizer_update_l2']!=0):raise ValueError('H0 state')
        if len(r['metrics'])!=1 or r['metrics'][0]['channel']!='polyp':raise ValueError('single channel')
        m=r['metrics'][0];v=ref['metrics'][0];validate_metric(m)
        if any(m[k]!=v[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')):raise ValueError('GT binding')
        if abs(r['prediction_foreground_fraction']-m['pred_pixels']/m['total_pixels'])>1e-6:raise ValueError('foreground scalar')
        if set(r['score_heads_final_original'])!=set(SCORE_HEAD_BN) or r['optimizer_update_l2']<0:raise ValueError('head telemetry')
        if not zero and (set(r['head_input_gradient_l2'])!=set(SCORE_HEAD_BN) or r['diagnostics']['bn_gradient_l2']<0):raise ValueError('gradient telemetry')
        json.dumps(r,allow_nan=False)
    return True


def summarize(arms):
    out=dict(groups=len(arms['N']),domains={})
    for d in dict.fromkeys(r['domain'] for r in arms['N']):
        rows={a:[r for r in rs if r['domain']==d] for a,rs in arms.items()};v=out['domains'][d]=dict(arms={},paired={})
        for a,rs in rows.items():
            ms=[r['metrics'][0] for r in rs];assd=[m['assd'] for m in ms if m['assd'] is not None]
            q=v['arms'][a]=dict(dice_percent=distribution([100*m['dice'] for m in ms]),assd_conditional_mean_px=float(np.mean(assd)) if assd else None,assd_defined=len(assd),assd_undefined=len(ms)-len(assd),**{k+'_count':sum(m[k] for m in ms) for k in ('pred_empty','pred_full','gt_empty','gt_full')})
            q['pixel_totals']={k:sum(m[k] for m in ms) for k in ('pred_pixels','gt_pixels','intersection','total_pixels')};q['pixel_totals'].update(FP=sum(m['pred_pixels']-m['intersection'] for m in ms),FN=sum(m['gt_pixels']-m['intersection'] for m in ms))
            if a in ('F','H','H0'):
                q['mechanism']=dict(head_outputs={n:{k:distribution([r['score_heads_final_original'][n][k] for r in rs]) for k in ('mean','std')} for n in SCORE_HEAD_BN},foreground_fraction=distribution([r['prediction_foreground_fraction'] for r in rs]),adam_update_l2=distribution([r['optimizer_update_l2'] for r in rs]))
                if a!='H0':q['mechanism'].update(loss=distribution([r['diagnostics']['point_bce'] for r in rs]),affine_gradient_l2=distribution([r['diagnostics']['bn_gradient_l2'] for r in rs]))
        v['paired']={a+'-'+b:paired([r['metrics'][0] for r in rows[a]],[r['metrics'][0] for r in rows[b]],False) for a,b in PAIRS}
    out['task_domain_macro_dice_percent']={a:float(np.mean([d['arms'][a]['dice_percent']['mean'] for d in out['domains'].values()])) for a in arms}
    q=out['task_domain_macro_dice_percent'];out['task_comparisons_pp']={a+'-'+b:q[a]-q[b] for a,b in PAIRS};return out


def recompute(out,reg,receipt):
    if torch.cuda.is_initialized():raise ValueError('CPU closeout only')
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='B5_RUN_COMPLETE':raise ValueError('incomplete run')
    totals={k:0 for k in BUDGET};cost={};target={}
    def account(name,rs):
        totals['records']+=len(rs)
        for k in totals:
            if k!='records':totals[k]+=sum(r['counts'][k] for r in rs)
        cost[name]=dict(records=len(rs),host_seconds=sum(r['host_seconds'] for r in rs),pipeline_seconds=sum(r['pipeline_seconds'] for r in rs),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rs))
    canonical=read_lines(out/'polyp_canonical_H0.jsonl');old0=old_records(reg,0);validate_rows(canonical,stream(reg,0),'H0',None,old0['N']);account('H0',canonical)
    for o in (0,1):
        arms=old0 if o==0 else old_records(reg,o);arms['H0']=map_rows(canonical,stream(reg,o))
        for a in ('F','H'):
            rs=arms[a]=read_lines(out/f'polyp_{o}_{a}.jsonl');validate_rows(rs,stream(reg,o),a,o,arms['N']);account(f'{o}_{a}',rs)
        target[f'order{o}']={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()}) for s in SUBSETS}
    if totals!=BUDGET or done['progress']!=totals:raise ValueError('B5 physical totals')
    comparisons=[target[o]['remaining_dev']['task_comparisons_pp'] for o in target]
    if all(x['H-A']>0 and x['H-EA']>0 for x in comparisons):decision='H exceeds A/EA in both exposed development order means; retain only as a development candidate subject to domain and paired-tail qualifications, then independent evaluation. No automatic next run.'
    elif any(x['H-C']>0 for x in comparisons):decision='At most partial or mixed recovery relative to C; consistent benefit over A/EA is not established. Do not promote H as a new effective main method or tune further automatically.'
    else:decision='No clear recovery over C. End this structural rescue experiment without trying other layers or losses.'
    if any(x['H-H0']<=0 for x in comparisons):decision+=' Additional consistency optimization does not exceed H0 in both orders; retain the inexpensive zero-update control.'
    public=dict(status=COMPLETE,execution_commit=receipt['commit'],target=target,decision=decision,H0_independent_predictions=len(canonical),H0_CPU_reference_positions=2*len(canonical),formal_new_records=totals['records'],descriptive_effect_reference_pp=.5,limitations=['Exposed development content; remaining_dev is not unused validation.','Single seed and repeated orders are not independent cohorts.','B4 remains a valid negative transfer experiment, not a confirmed implementation bug.','Head policies do not guarantee area, calibration or a unique causal decomposition.','No new Fundus run, DD/SA/G/U/S/I, fusion, LR/radius/layer search or automatic next experiment.'])
    smoke=json.loads((out/'smoke.completion.json').read_text())
    if smoke['physical']!=SMOKE:raise ValueError('smoke totals')
    audit=dict(status=COMPLETE,execution_commit=receipt['commit'],formal=totals,total_including_smoke={k:totals[k]+SMOKE[k] for k in SMOKE},smoke=smoke,cpu=json.loads((out/'CPU_validation.json').read_text()),trainable=done['trainable'],cost=cost,gpu_seconds=done['gpu_seconds'],private_bytes_at_recompute=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),scalar_recompute=True,ASSD_pixel_recomputed=False,new_offline_training=0,exit_code=0)
    private_json(out/'public_aggregate.json',public);private_json(out/'execution_audit.json',audit)
    render(out,public,audit);private_json(out/'verification.json',dict(status=COMPLETE,formal=totals,exit_code=0))


def render(out,p,audit):
    lines=['# B5 score-head preservation','',COMPLETE,'','Execution commit: '+p['execution_commit'],'',p['decision']]
    for subset in SUBSETS:
        names=list(p['target']['order0'][subset]['task_domain_macro_dice_percent']);pairs=[a+'-'+b for a,b in PAIRS]
        for field,columns in [('task_domain_macro_dice_percent',names),('task_comparisons_pp',pairs)]:
            lines+=['','## '+subset+' / '+field,'','| Order | '+' | '.join(columns)+' |','|---|'+'---:|'*len(columns)]
            for order,x in p['target'].items():lines+=['| '+order+' | '+' | '.join(f'{x[subset][field][k]:.6f}' for k in columns)+' |']
    for order,x in p['target'].items():
        lines+=['','## '+order+' remaining_dev domains and paired tails']
        for domain,d in x['remaining_dev']['domains'].items():
            lines+=['',domain+': '+', '.join(f'{a}={d["arms"][a]["dice_percent"]["mean"]:.6f}%' for a in ('C','F','H','H0','A','EA'))]
            for pair in ('H-F','H-H0','H-A','H-EA'):lines+=['',pair+': `'+json.dumps(d['paired'][pair],sort_keys=True)+'`']
    lines+=['','## Cost and scope','',json.dumps(audit['formal'])+'; including smoke: '+json.dumps(audit['total_including_smoke']),f"H0 independent predictions: {p['H0_independent_predictions']}; CPU references: {p['H0_CPU_reference_positions']}.",'Active-stage seconds: '+str(audit['gpu_seconds']),json.dumps(audit['cost'],sort_keys=True),'','F/H reuse B4 PolypC.step unchanged; only three exact head affine/statistics policies differ. Loss crosses frozen heads without no_grad. H0 has no optimizer. The scalar mechanism observations are from the existing final-original forward; they do not establish calibrated probability or unique causality.','All domains, four subsets, unrounded paired signs, ceil(10%) tails, common-valid ASSD, undefined/empty/full, FP/FN and pixel totals are in public_aggregate.json. CPU reconstructs Dice from scalar counts and ASSD cohorts from recorded values, not discarded maps.','Host timing excludes evaluator; pipeline timing includes it. Network calls are not FLOPs, old timings are historical, and repeated content/orders are not independent cohorts.','No new Fundus run or source/DD/selector/outer/inner training. DD, SA, G, U/S/I, fusion, radius/LR/layer search remain closed. B4 negative results are preserved. Completion is distinct from scientific utility; no automatic next experiment.','']
    (out/'B5_EXPERIMENT_REPORT.md').write_text('\n'.join(lines))
