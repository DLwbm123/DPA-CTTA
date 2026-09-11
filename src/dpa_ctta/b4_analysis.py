"""Scalar recomputation; canonical C0 references never count as extra inference."""
import json
from pathlib import Path
import numpy as np
import torch
from .b4_data import BUDGET,stream,old_records,map_rows
from .p2_data import SUBSETS
from .p2_analysis import validate_metric
from .m1_analysis import read_lines,paired
from .host_diagnostic_analysis import distribution,channels
from .host_diagnostic_run import private_json

COMPLETE='B4_FROZEN_C_TRANSFER_COMPLETE'
PAIRS={'fundus':(('C','C0'),('A','C0'),('C0','N')),'polyp':(('C','A'),('C','EA'),('C','C0'),('C0','N'),('C0','A'),('C','O2'),('C','D4'))}


def validate_rows(rows,ordered,task,arm,order,n):
    if len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('B4 coverage')
    for i,(r,e,ref) in enumerate(zip(rows,ordered,n)):
        values=dict(task=task,arm=arm,order=order,visit=i+1,**{k:e[k] for k in ('domain','subset','group_id','sample_id')})
        if any(r[k]!=v for k,v in values.items()):raise ValueError('B4 ordered identity')
        c0=arm=='C0';wanted=dict(forwards=1 if c0 else 8,backwards=int(not c0),base_adam=int(not c0),perturb=0,restore=0)
        if r['counts']!=wanted or r['adam_step']!=(0 if c0 else i+1) or not r['prediction_fixed_before_label'] or not r['frozen_parameters_checked']:raise ValueError('B4 state/counts')
        if c0:
            if r['prediction_origin']!='canonical_stateless' or not r['stateless_checked'] or r['lr'] is not None:raise ValueError('C0 contract')
        elif r['lr']!=1e-4 or r['prediction_origin']!='continuous_update':raise ValueError('C contract')
        if [m['channel'] for m in r['metrics']]!=(['OD','OC'] if task=='fundus' else ['polyp']):raise ValueError('metric channels')
        for m,v in zip(r['metrics'],ref['metrics']):
            validate_metric(m)
            if any(m[k]!=v[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')):raise ValueError('GT binding')
        json.dumps(r,allow_nan=False)
    return True


def summarize(arms,task):
    domains=list(dict.fromkeys(r['domain'] for r in arms['N']));names=['OD','OC','macro'] if task=='fundus' else ['polyp'];out=dict(groups=len(arms['N']),domains={})
    for d in domains:
        rows={a:[r for r in rs if r['domain']==d] for a,rs in arms.items()};item=out['domains'][d]=dict(arms={},paired={})
        for a,rs in rows.items():
            item['arms'][a]={}
            for c in names:
                ms=channels(rs,c);v=dict(dice_percent=distribution([100*m['dice'] for m in ms]))
                if c!='macro':
                    values=[m['assd'] for m in ms if m['assd'] is not None]
                    v.update(assd_conditional_mean_px=float(np.mean(values)) if values else None,assd_defined=len(values),assd_undefined=len(ms)-len(values),**{k+'_count':sum(m[k] for m in ms) for k in ('gt_empty','gt_full','pred_empty','pred_full')})
                item['arms'][a][c]=v
        item['paired']={a+'-'+b:{c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in names} for a,b in PAIRS[task]}
    out['task_domain_macro_dice_percent']={a:float(np.mean([out['domains'][d]['arms'][a][names[-1]]['dice_percent']['mean'] for d in domains])) for a in arms}
    v=out['task_domain_macro_dice_percent'];out['task_comparisons_pp']={a+'-'+b:v[a]-v[b] for a,b in PAIRS[task]}
    return out


def recompute(out,reg,receipt):
    if torch.cuda.is_initialized():raise ValueError('CPU closeout required')
    done=json.loads((out/'run.completion.json').read_text());assert done['status']=='B4_RUN_COMPLETE'
    target={};totals={k:0 for k in BUDGET};cost={}
    def account(key,rows):
        totals['records']+=len(rows)
        for k in totals:
            if k!='records':totals[k]+=sum(r['counts'][k] for r in rows)
        cost[key]=dict(records=len(rows),counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},host_seconds=sum(r['host_seconds'] for r in rows),pipeline_seconds=sum(r['pipeline_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
    for task in ('fundus','polyp'):
        target[task]={};canonical=read_lines(out/f'{task}_canonical_C0.jsonl');base=stream(reg,task,0)
        old0=old_records(reg,task,0);validate_rows(canonical,base,task,'C0',None,old0['N']);account(task+'_C0',canonical)
        for o in range(4 if task=='fundus' else 2):
            ordered=stream(reg,task,o);arms=old0 if o==0 else old_records(reg,task,o);arms['C0']=map_rows(canonical,ordered)
            if task=='polyp':
                rows=arms['C']=read_lines(out/f'polyp_{o}_C.jsonl');validate_rows(rows,ordered,task,'C',o,arms['N']);account(f'polyp_{o}_C',rows)
            target[task][f'order{o}']={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()},task) for s in SUBSETS}
    if totals!=BUDGET or done['progress']!=totals:raise ValueError('B4 physical totals')
    p=[target['polyp'][f'order{o}']['remaining_dev']['task_comparisons_pp'] for o in (0,1)]
    f=[target['fundus'][f'order{o}']['remaining_dev']['task_comparisons_pp']['C-C0'] for o in range(4)]
    if all(r[k]>0 for r in p for k in ('C-A','C-EA','C-C0')):decision='C-PraNet has positive task means versus A/EA/C0 in both development orders; channel/domain tails and cost still qualify utility, not novel-method or unseen-generalization evidence.'
    elif all(r['C-A']>0 for r in p) and any(r['C-C0']<=0 for r in p):decision='Improvement over A does not establish value over current-statistics zero-update inference; retain cheap C0 as a strong comparator.'
    else:decision='No consistent positive transfer versus the prescribed strong controls. Limit prior C evidence to the tested Fundus setting; retain all negative Polyp results without retuning.'
    if any(v<=0 for v in f):decision+=' Fundus consistency updates also fail to exceed C0 in every observed order; narrow attribution to online optimization.'
    result=dict(status=COMPLETE,execution_commit=receipt['commit'],target=target,decision=decision,canonical_predictions=3753,C0_CPU_reference_positions=11408,new_logical_positions=15012,formal_new_records=7357,source_provenance=dict(P2='d3ee6901379be293f47abd1687808caaf5b04266',B3='0fac9b2b1e762ed57b624953edf95058cb1f0380'),limitations=['All content is development data; single seed and repeated orders are not independent patients.','C-PraNet is an explicit task transfer of an existing recipe, not an official GraTa Polyp reproduction or a new loss.','C0 uses current BN statistics and is not N or an optimizer with zero learning rate.','No U/S/I/G expansion, DD, radius/LR search or new domain order.'])
    smoke=json.loads((out/'smoke.completion.json').read_text())
    audit=dict(status=COMPLETE,execution_commit=receipt['commit'],formal=totals,total_including_smoke={k:totals[k]+smoke['physical'][k] for k in totals if k!='records'},smoke=smoke,cpu=json.loads((out/'CPU_validation.json').read_text()),trainable=done['trainable'],cost=cost,gpu_seconds=done['gpu_seconds'],private_bytes_at_recompute=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),scalar_recompute=True,ASSD_pixel_recomputed=False,new_offline_training=0,exit_code=0)
    private_json(out/'public_aggregate.json',result);private_json(out/'execution_audit.json',audit)
    lines=['# B4 frozen C transfer and zero-update comparison','',COMPLETE,'','Execution commit: '+receipt['commit'],'',decision]
    for task in ('polyp','fundus'):
        for subset in SUBSETS:
            arms=list(target[task]['order0'][subset]['task_domain_macro_dice_percent'])
            lines+=['','## '+task+' / '+subset,'','| Order | '+' | '.join(arms)+' |','|---|'+'---:|'*len(arms)]
            for order,rs in target[task].items():lines+=['| '+order+' | '+' | '.join(f'{rs[subset]["task_domain_macro_dice_percent"][a]:.6f}' for a in arms)+' |']
            lines+=['','| Order | '+' | '.join(a+'-'+b for a,b in PAIRS[task])+' |','|---|'+'---:|'*len(PAIRS[task])]
            for order,rs in target[task].items():lines+=['| '+order+' | '+' | '.join(f'{rs[subset]["task_comparisons_pp"][a+"-"+b]:+.6f}' for a,b in PAIRS[task])+' |']
    lines+=['','C-PraNet preprocessing: original and geometric views are CPU RGB with one original ImageNet channel normalization. Strong GraTa style works on an independent RGB numpy copy, then its RGB min/max normalization on CPU and exactly one ImageNet normalization. No probability min/max or output fusion. Fundus C is unchanged and reused.','C0 makes one current-statistics original-image forward per unique group. Its cross-order references are not extra predictions or independent repeats. Standard source-eval adapter parity and C0 versus first pre-update original view were checked in the fixed smoke.','All per-domain/channel paired signs, worst ceil(10%) differences, worst individual differences, conditional and common-valid ASSD are in the aggregate. No macro ASSD; CPU does not reconstruct discarded distance maps.','Network calls are not FLOPs. Old timings are historical and do not establish controlled speedups. No new source-clean scoring or source training.','All target data remain development data. No safety, significance, new algorithm or unseen generalization claim. No U/S/I/G expansion, DD training, radius/LR search or new domain order was executed. The B2/B3 interval series remains closed.','']
    (out/'B4_EXPERIMENT_REPORT.md').write_text('\n'.join(lines));private_json(out/'verification.json',dict(status=COMPLETE,formal=totals,exit_code=0))
