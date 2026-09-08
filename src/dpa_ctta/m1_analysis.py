"""Independent CPU-only reconstruction from frozen registration and scalar logs."""
import json
import math
import numpy as np
from .host_diagnostic_analysis import distribution, channels
from .host_diagnostic_run import private_json
from .source_pilot import validate_arm

ARMS=('N','A','R','D','O')
PAIRS=(('O','D'),('O','R'),('O','A'),('O','N'),('D','R'),('R','A'),('A','N'))


def read_lines(path): return [json.loads(line) for line in path.read_text().splitlines()]


def paired(a,b,macro=False):
    values=[100*(x['dice']-y['dice']) for x,y in zip(a,b)]
    out={'dice_delta_pp':distribution(values)}
    if not macro:
        both=[x['assd']-y['assd'] for x,y in zip(a,b) if x['assd'] is not None and y['assd'] is not None]
        tail=max(1,math.ceil(.1*len(both)))
        out.update(assd_common_valid=len(both),assd_not_jointly_defined=len(a)-len(both),
            assd_left_undefined=sum(x['assd'] is None for x in a),assd_right_undefined=sum(x['assd'] is None for x in b),
            assd_delta_mean_px=float(np.mean(both)) if both else None,assd_delta_median_px=float(np.median(both)) if both else None,
            assd_adverse_upper_decile_mean_px=float(np.sort(both)[-tail:].mean()) if both else None)
    return out


def summarize(arms,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp']
    result={'domains':{},'task_domain_macro_dice_percent':{},'task_comparisons_pp':{}}
    domains=list(dict.fromkeys(r['domain'] for r in arms['N']))
    for domain in domains:
        records={a:[r for r in rs if r['domain']==domain] for a,rs in arms.items()}
        entry=result['domains'][domain]={'arms':{},'paired':{}}
        for arm,rs in records.items():
            entry['arms'][arm]={}
            for c in names:
                ms=channels(rs,c);v={'dice_percent':distribution([100*m['dice'] for m in ms])}
                if c!='macro':
                    valid=[m['assd'] for m in ms if m['assd'] is not None]
                    v.update(assd_conditional_mean_px=float(np.mean(valid)) if valid else None,assd_defined=len(valid),assd_undefined=len(ms)-len(valid),
                        **{k+'_count':sum(m[k] for m in ms) for k in ['gt_empty','gt_full','pred_empty','pred_full']})
                entry['arms'][arm][c]=v
            entry['arms'][arm]['cost']={k:sum(r[k] for r in rs) for k in ['pipeline_elapsed_seconds','host_step_elapsed_seconds']}
            entry['arms'][arm]['cost']['peak_allocated_bytes']=max(r['peak_allocated_bytes'] for r in rs)
        for a,b in PAIRS:
            entry['paired'][a+'-'+b]={c:paired(channels(records[a],c),channels(records[b],c),c=='macro') for c in names}
    c='macro' if task=='fundus' else 'polyp'
    for arm in ARMS:
        result['task_domain_macro_dice_percent'][arm]=float(np.mean([result['domains'][d]['arms'][arm][c]['dice_percent']['mean'] for d in domains]))
    for a,b in PAIRS:
        result['task_comparisons_pp'][a+'-'+b]=result['task_domain_macro_dice_percent'][a]-result['task_domain_macro_dice_percent'][b]
    # Task pooled per-content summaries are explicitly separate from equal-domain task means.
    result['pooled_content_paired']={a+'-'+b:{ch:paired(channels(arms[a],ch),channels(arms[b],ch),ch=='macro') for ch in names} for a,b in PAIRS}
    scores=result['task_domain_macro_dice_percent']
    result['scientific_signal']='NET_GAIN' if scores['O']>max(scores[a] for a in ('N','A','R','D')) else 'HARM_REDUCTION_ONLY' if scores['O']>scores['A'] and scores['O']<=scores['N'] else 'MIXED' if scores['O']>min(scores[a] for a in ('N','A','R','D')) else 'NO_OBSERVED_GAIN'
    return result


def recompute(out,registration,receipt):
    completion=json.loads((out/'run.completion.json').read_text())
    if completion['status']!='M1_RUN_COMPLETE': raise ValueError('missing full execution completion')
    public={'source':{},'target':{},'training':{},'limitations':['exploratory previously evaluated target domains','UNKNOWN patient/video links where unavailable','single seed/order','fixed Base history off-policy truncated one-step objective','equal episodes do not imply equal compute','GM is prompt-space adaptation baseline','fixed masks and synthetic images can retain source information']}
    measured=dict(online=0,outer=0,inner=0,records=0);compute={};source_bytes={}
    for task,reg in registration['tasks'].items():
        history=read_lines(out/f'history_{task}.jsonl')
        if len(history)!=32 or any(r['visit']!=i+1 or r['group_id']!=reg['history'][i]['group_id'] or r['counts']['online_adam']!=i+1 or r['counts']['memory_pushes']!=i+1 for i,r in enumerate(history)): raise ValueError('history order/counter mismatch')
        hd=json.loads((out/f'history_{task}.completion.json').read_text())
        if not hd['source_unchanged'] or hd['counts']!=history[-1]['counts']: raise ValueError('history completion mismatch')
        measured['online']+=history[-1]['counts']['online_adam'];compute['history_'+task]=hd['counts']
        public['training'][task]={};source_bytes[task]=reg['checkpoint']['bytes']
        for method in ('D','O'):
            rows=read_lines(out/f'train_{task}_{method}.jsonl');done=json.loads((out/f'train_{task}_{method}.completion.json').read_text())
            if len(rows)!=600 or done['episodes']!=600 or done['source_unchanged'] is not True: raise ValueError('training incomplete')
            for row,wanted in zip(rows,reg['episodes']):
                if any(row[k]!=v for k,v in wanted.items()) or row['image_adam_step']!=row['episode']: raise ValueError('episode identity/order/optimizer mismatch')
                json.dumps(row,allow_nan=False)
                if row['counts']['differentiable_inner']!=(row['episode'] if method=='O' else 0): raise ValueError('inner counter mismatch')
            if rows[-1]['counts']!=done['counts']: raise ValueError('training completion count mismatch')
            frozen=json.loads((out/'final_artifacts.frozen.json').read_text())
            if done['final_sha256']!=frozen[task][method]: raise ValueError('frozen final mismatch')
            measured['outer']+=rows[-1]['image_adam_step'];measured['inner']+=rows[-1]['counts']['differentiable_inner']
            compute['training_'+task+'_'+method]=done['counts']
            public['training'][task][method]=dict(episodes=600,elapsed_seconds=done['elapsed_seconds'],peak_allocated_bytes=done['peak_allocated_bytes'],artifact_bytes=done['artifact_bytes'],
                loss_blocks=[dict(start=i+1,end=min(i+100,600),loss=distribution([r['loss'] for r in rows[i:i+100]]),gradient_norm=distribution([r['gradient_norm'] for r in rows[i:i+100]])) for i in range(0,600,100)])
        for stage in ('source','target'):
            expected=reg['source_query'] if stage=='source' else reg['target']
            if not expected:
                public[stage][task]={'status':'TARGET_NOT_EVALUATED'};continue
            arms={}
            wanted=[dict(visit=i+1,sample_id=r['sample_id'],group_id=r['group_id'],segment='clean') for i,r in enumerate(expected)]
            for arm in ARMS:
                rows=read_lines(out/f'{stage}_{task}_{arm}.jsonl')
                validate_arm(rows,wanted,task,'B' if arm in ('R','D','O') else arm)
                totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
                for row,exp in zip(rows,expected):
                    if (row['domain'],row['task'],row['arm'])!=(exp['domain'],task,arm): raise ValueError('domain/task/arm mismatch')
                    if row['counts']['online_adam']!=int(arm!='N') or row['counts']['memory_pushes']!=int(arm!='N'): raise ValueError('step/push mismatch')
                    json.dumps(row,allow_nan=False)
                done=json.loads((out/f'{stage}_{task}_{arm}.completion.json').read_text())
                if done['records']!=len(rows) or not done['source_unchanged'] or done['counts']!=totals: raise ValueError('score completion mismatch')
                measured['online']+=totals['online_adam'];measured['records']+=len(rows);compute[stage+'_'+task+'_'+arm]=totals;arms[arm]=rows
            public[stage][task]=summarize(arms,task)
    if any(measured[k]!=completion['progress'][k] for k in measured): raise ValueError('independent totals differ from runtime')
    q=sum(len(r['target']) for r in registration['tasks'].values())
    if measured!=dict(online=64+208+4*q,outer=2400,inner=1200,records=260+5*q): raise ValueError('independent budget mismatch')
    sm=json.loads((out/'smoke.completion.json').read_text())
    measured['online']+=sm['updates']['online'];measured['outer']+=sm['updates']['outer'];measured['inner']+=sm['updates']['inner']
    missing={t:[d for d,c in r['target_counts'].items() if not c['selected']] for t,r in registration['tasks'].items()}
    status='M1_PARTIAL' if any(missing.values()) else 'M1_METHOD_VALIDATION_COMPLETE'
    audit=dict(status=status,experiment_execution_commit=receipt['commit'],updates_and_records=measured,actual_compute=compute,smoke=sm['evidence'],
        GPU_UUID=receipt['gpu_uuid'],gpu_seconds=completion['gpu_seconds'],source_model_bytes=source_bytes,
        private_output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),target_selected=q,missing_domains=missing,
        source_registration_identities=len(registration['identities']),CPU_recompute_exit_code=0,background_resume=False)
    private_json(out/'public_aggregate.json',dict(status=status,**public));private_json(out/'execution_audit.json',audit)
    private_json(out/'verification.json',dict(status=status,independent_CPU_recompute=True,counts=measured,exit_code=0))
