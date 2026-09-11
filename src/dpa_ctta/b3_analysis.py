"""CPU-only combination of explicitly sourced old and new scalar trajectories."""
import json
from pathlib import Path
import numpy as np
import torch
from .b3_orders import ORDERS,ARMS,BUDGET,build_stream
from .b2_analysis import old_records as b1_records,validate_rows as validate_interval
from .b1_analysis import validate_rows as validate_C
from .p2_analysis import validate_metric
from .p2_data import SUBSETS
from .p1_analysis import evaluate
from .m1_analysis import read_lines,paired
from .host_diagnostic_analysis import channels,distribution
from .host_diagnostic_run import private_json
PAIRS=(('I','C'),('I','U'),('I','S'),('I','A'),('C','A'),('U','A'),('S','A'))
COMPLETE='B3_FROZEN_ORDER_COMPARISON_COMPLETE'


def old_records(reg,order):
    arms=b1_records(reg,order);stream=build_stream(reg,order)
    for arm in ('U','S','I'):
        rows=arms[arm]=read_lines(Path(reg['b2_directory'])/f'fundus_{order}_{arm}.jsonl')
        validate_interval(rows,stream,order,arm,arms['N'])
    return arms


def reorder_N(rows,stream):
    lookup={r['group_id']:r for r in rows}
    if len(lookup)!=len(rows) or set(lookup)!={r['group_id'] for r in stream}:raise ValueError('N identity coverage')
    return [lookup[r['group_id']] for r in stream]


def validate_rows(rows,stream,order,arm,n):
    if arm=='C':return validate_C(rows,stream,order,arm,n)
    if arm!='A':return validate_interval(rows,stream,order,arm,n)
    if len(rows)!=len(stream) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('A coverage')
    for i,(r,e,ref) in enumerate(zip(rows,stream,n)):
        want=dict(task='fundus',order=order,arm=arm,visit=i+1,adam_step=i+1,domain=e['domain'],subset=e['subset'],sample_id=e['sample_id'],group_id=e['group_id'])
        if any(r[k]!=v for k,v in want.items()):raise ValueError('A identity/lifecycle')
        if r['counts']!=dict(forwards=2,backwards=1,base_adam=1,perturb=0,restore=0):raise ValueError('A counts')
        c=r['native_counts']
        if any(c[k]!=1 for k in ('online_adam','backward_calls','memory_pushes')) or c['model_forwards']!=2 or c['prompt_forwards']!=2+int(i>=16) or c['retrievals']!=int(i>=16) or c['proxy_forwards'] or c['proxy_images']:raise ValueError('A native counters')
        if r['parent_counters']!=[i+1] or not 1<=r['parent_memory_size']<=min(41,i+1) or r['lr']!=.05:raise ValueError('A optimizer/memory')
        if not r['source_unchanged'] or not r['prediction_fixed_before_label'] or r['interval_diagnostics'] is not None:raise ValueError('A source/label')
        if [m['channel'] for m in r['metrics']]!=['OD','OC']:raise ValueError('A channels')
        for m,v in zip(r['metrics'],ref['metrics']):
            validate_metric(m)
            if any(m[k]!=v[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')):raise ValueError('A GT identity')
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


def combine(target):
    result={}
    for subset in SUBSETS:
        rs=[target[f'order{o}'][subset] for o in range(4)]
        means={label:{a:float(np.mean([rs[o]['task_domain_macro_dice_percent'][a] for o in ids])) for a in ARMS} for label,ids in [('new_orders_mean',(2,3)),('four_orders_mean',(0,1,2,3))]}
        ranges={a:dict(minimum=min(v),maximum=max(v),range=max(v)-min(v)) for a in ARMS for v in [[r['task_domain_macro_dice_percent'][a] for r in rs]]}
        comparisons={a+'-'+b:dict(by_order=v,minimum=min(v),maximum=max(v),range=max(v)-min(v),absolute_by_order=[abs(x) for x in v],positive=sum(x>0 for x in v),negative=sum(x<0 for x in v),zero=sum(x==0 for x in v),new_mean=float(np.mean(v[2:])),four_mean=float(np.mean(v))) for a,b in PAIRS for v in [[r['task_comparisons_pp'][a+'-'+b] for r in rs]]}
        positions={d:[dict(order=o,block_position=ORDERS[o].index(d)+1,arms=rs[o]['domains'][d]['arms'],paired=rs[o]['domains'][d]['paired']) for o in range(4)] for d in ORDERS[0]}
        result[subset]=dict(**means,order_sensitivity=ranges,comparisons=comparisons,domains_by_position=positions)
    return result


def recompute(out,reg,receipt):
    if torch.cuda.is_initialized():raise ValueError('CPU reconstruction only')
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='B3_RUN_COMPLETE':raise ValueError('incomplete run')
    target={};totals={k:0 for k in BUDGET};cost={};old={o:old_records(reg,o) for o in (0,1)}
    background={f'order{o}':{s:{a:float(np.mean([np.mean([100*m['dice'] for r in old[o][a] if r['domain']==d and (s=='all_dev' or r['subset']==s) for m in r['metrics']]) for d in ORDERS[0]])) for a in ('G','O2','D4')} for s in SUBSETS} for o in (0,1)}
    for o in range(4):
        stream=build_stream(reg,o)
        if o<2:arms={a:old[o][a] for a in ('N',*ARMS)}
        else:
            arms={'N':reorder_N(old[0]['N'],stream)}
            for arm in ARMS:
                rows=arms[arm]=read_lines(out/f'fundus_{o}_{arm}.jsonl');validate_rows(rows,stream,o,arm,arms['N'])
                totals['records']+=len(rows)
                for k in totals:
                    if k!='records':totals[k]+=sum(r['counts'][k] for r in rows)
                cost[f'{o}_{arm}']=dict(records=len(rows),counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},host_seconds=sum(r['host_seconds'] for r in rows),pipeline_seconds=sum(r['pipeline_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),native_counts={k:sum(r['native_counts'][k] for r in rows) for k in rows[0]['native_counts']} if arm=='A' else None)
        target[f'order{o}']={s:summarize({a:[r for r in rows if s=='all_dev' or r['subset']==s] for a,rows in arms.items()}) for s in SUBSETS}
    if totals!=BUDGET or done['progress']!=totals:raise ValueError('B3 independent totals')
    aggregate=combine(target);main=aggregate['remaining_dev'];ic=main['comparisons']['I-C'];spatial=all(main['comparisons'][key]['four_mean']>0 and all(x>0 for x in main['comparisons'][key]['by_order'][2:]) for key in ('I-U','I-S'))
    if all(x>0 for x in ic['by_order'][2:]) and ic['four_mean']>0 and spatial:
        decision='Positive matched development signal; review all critical-channel costs before considering independent evaluation. No general or clinical claim.'
    elif ic['four_mean']<=0 and main['order_sensitivity']['I']['minimum']>main['order_sensitivity']['C']['minimum']:
        decision='Observed mean versus worst-order tradeoff; retain C as mean baseline. I is not the default.'
    else:decision='No consistent matched support for replacing C. Retain C and end automatic expansion of this fixed radius configuration.'
    if not spatial:decision+=' Spatial radius placement has no consistent advantage over U/S under the prespecified comparison.'
    result=dict(status=COMPLETE,execution_commit=receipt['commit'],target=target,combined=aggregate,decision=decision,old_background_only=background,provenance=reg['execution_provenance'],coverage=reg['tasks']['fundus']['counts'],distinct_groups=1951,old_adapting_records_reused=19510,new_adapting_records=19510,four_order_adapting_visits=39020,N_display_values=7804,N_new_forwards=0,limitations=['All subsets are development data; order design chosen after B2.','Four of 24 permutations; block and directed-neighbor balance is not equal cumulative exposure or causal identification.','Same contents and seed; orders are not independent patient samples.','Changed time positions change augmentation draws as well as history.','B2 endpoint and conclusions remain unchanged.'])
    smoke=json.loads((out/'smoke.completion.json').read_text())
    audit=dict(status=COMPLETE,execution_commit=receipt['commit'],formal=totals,total_including_smoke={k:totals[k]+smoke['physical'][k] for k in totals if k!='records'},smoke=smoke,cpu=json.loads((out/'CPU_validation.json').read_text()),trainable=done['trainable'],cost=cost,FFT_accounting='A pinned Prompt.forward contains one fft2 and one ifft2 per observed prompt call; not segmentation network forwards',gpu_seconds=done['gpu_seconds'],private_bytes_at_recompute=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),scalar_recompute=True,ASSD_pixel_recomputed=False,new_source_DD_selector_outer_inner=0,exit_code=0)
    private_json(out/'public_aggregate.json',result);private_json(out/'execution_audit.json',audit)
    lines=['# B3 frozen balanced order comparison','',COMPLETE,'','Execution commit: '+receipt['commit'],'',decision,'','New evidence: orders 2/3 only; old 0/1 A from P2, C from B1, U/S/I from B2. N is rearranged by identity, without inference.']
    for subset in SUBSETS:
        lines+=['','## '+subset,'','| Order / descriptive mean | '+' | '.join(ARMS)+' |','|---|'+'---:|'*len(ARMS)]
        for o in range(4):lines+=['| '+str(o)+(' (reused)' if o<2 else ' (new)')+' | '+' | '.join(f'{target[f"order{o}"][subset]["task_domain_macro_dice_percent"][a]:.6f}' for a in ARMS)+' |']
        for label in ('new_orders_mean','four_orders_mean'):lines+=['| '+label+' | '+' | '.join(f'{aggregate[subset][label][a]:.6f}' for a in ARMS)+' |']
        lines+=['','| Pair | order0 | order1 | order2 | order3 | New mean | Four mean |','|---|---:|---:|---:|---:|---:|---:|']
        for k,v in aggregate[subset]['comparisons'].items():lines+=['| '+k+' | '+' | '.join(f'{x:+.6f}' for x in [*v['by_order'],v['new_mean'],v['four_mean']])+' |']
    lines+=['','Four-order method min/max/range and every domain at every block position (OD/OC/macro Dice, conditional ASSD, paired signs and lower tails) are in public_aggregate.json. REFUGE_Valid I-C and I-A both remain visible; all other domain costs remain in the same tables.','All means weight domains equally. Drishti_GS remaining_dev has 37 groups but retains one-quarter domain weight. Lower-tail groups are post-hoc paired-error descriptions, not a frozen hard-case cohort. No significance or clinical thresholds.','The 20-update paired smoke checks entry equivalence on two procedural images per path/arm; it is not a new full native-memory-lifetime test. Historical lifecycle validation is reused. Formal calls are not equal FLOPs between A and BN methods.','No target images, masks, maps, optimizer snapshots or model snapshots were persisted. Scalar Dice is reconstructed; ASSD pixels were discarded. Optional B2 q-error bins are not backfilled. No additional model, seed, radius, source training, Polyp, routing or ensemble run is authorized by completion.','']
    (out/'B3_EXPERIMENT_REPORT.md').write_text('\n'.join(lines));private_json(out/'verification.json',dict(status=COMPLETE,formal=totals,exit_code=0))
