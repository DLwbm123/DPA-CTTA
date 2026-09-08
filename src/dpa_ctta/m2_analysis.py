"""CPU-only reconstruction: 552 new records, with separately attributed M1 baselines."""
import json
from pathlib import Path
import numpy as np
from .host_diagnostic_analysis import channels,distribution
from .m1_analysis import paired,read_lines
from .host_diagnostic_run import private_json
from .source_pilot import validate_arm
from .m2_run import NEW_ARMS,validate_new_arm

OLD_MAP={'N':'N','A':'A','R':'R','D1':'D','O1':'O'}
PAIRS=[('O2','D2'),('O2','O1'),('D2','D1'),('O2','A'),('O2','R'),('O2','N'),('D2','A'),('D2','N')]


def validate_new_results(results,expected,task):
    if set(results)!=set(NEW_ARMS):raise ValueError('only complete D2/O2 new pair accepted')
    for arm,rows in results.items():validate_new_arm(rows,expected,task,arm)
    return True


def interaction(arms,channel):
    if not all(k in arms for k in ('O2','D2','O1','D1')):return 'NOT_AVAILABLE'
    groups=[channels(arms[k],channel) for k in ('O2','D2','O1','D1')]
    out={'dice_delta_pp':distribution([100*((a['dice']-b['dice'])-(c['dice']-d['dice'])) for a,b,c,d in zip(*groups)])}
    if channel!='macro':
        joint=[(a['assd']-b['assd'])-(c['assd']-d['assd']) for a,b,c,d in zip(*groups) if all(x['assd'] is not None for x in (a,b,c,d))]
        out.update(assd_four_way_common_valid=len(joint),assd_not_jointly_defined=len(groups[0])-len(joint),
            assd_interaction_mean_px=float(np.mean(joint)) if joint else None)
    return out


def summarize(arms,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp'];primary=names[-1]
    domains=list(dict.fromkeys(row['domain'] for row in arms['D2']))
    result={'domains':{},'task_domain_macro_dice_percent':{},'task_comparisons_pp':{}}
    for domain in domains:
        rows={a:[r for r in records if r['domain']==domain] for a,records in arms.items()}
        item=result['domains'][domain]={'arms':{},'paired':{}}
        for arm,records in rows.items():
            item['arms'][arm]={}
            for c in names:
                ms=channels(records,c);entry={'dice_percent':distribution([100*m['dice'] for m in ms])}
                if c!='macro':
                    valid=[m['assd'] for m in ms if m['assd'] is not None]
                    entry.update(assd_conditional_mean_px=float(np.mean(valid)) if valid else None,assd_defined=len(valid),assd_undefined=len(ms)-len(valid),
                        **{k+'_count':sum(m[k] for m in ms) for k in ('gt_empty','gt_full','pred_empty','pred_full')})
                item['arms'][arm][c]=entry
        for a,b in PAIRS:
            item['paired'][a+'-'+b]={c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in names} if a in rows and b in rows else 'NOT_AVAILABLE'
        item['interaction']={c:interaction(rows,c) for c in names}
    for arm in arms:
        result['task_domain_macro_dice_percent'][arm]=float(np.mean([result['domains'][d]['arms'][arm][primary]['dice_percent']['mean'] for d in domains]))
    values=result['task_domain_macro_dice_percent']
    for a,b in PAIRS:result['task_comparisons_pp'][a+'-'+b]=values[a]-values[b] if a in values and b in values else None
    result['task_interaction_pp']=(values['O2']-values['D2'])-(values['O1']-values['D1']) if all(k in values for k in ('O1','D1')) else None
    result['pooled_content_paired']={a+'-'+b:{c:paired(channels(arms[a],c),channels(arms[b],c),c=='macro') for c in names} if a in arms and b in arms else 'NOT_AVAILABLE' for a,b in PAIRS}
    result['pooled_content_interaction']={c:interaction(arms,c) for c in names}
    return result


def recompute(out,overlay,registration,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='M2_RUN_COMPLETE':raise ValueError('M2 run completion missing')
    old=Path(overlay['old_directory']);public={'source':{},'target':{},'training':{},'coverage':{t:r['coverage'] for t,r in overlay['tasks'].items()},
        'provenance':{'new_arms':['D2','O2'],'reused_M1_arms':list(OLD_MAP),'M1_execution_commit':'d92ed88e603eb68aea97a57493c96a084e98d91c','new_execution_commit':receipt['commit']}}
    measured=dict(online=0,outer=0,inner=0,records=0);reused=0;compute={};scoring={};finals=json.loads((out/'final_artifacts.frozen.json').read_text())
    for task,reg in registration['tasks'].items():
        public['training'][task]={}
        for arm in NEW_ARMS:
            rows=read_lines(out/f'train_{task}_{arm}.jsonl');complete=json.loads((out/f'train_{task}_{arm}.completion.json').read_text())
            if len(rows)!=600 or complete['episodes']!=600 or complete['source_unchanged'] is not True:raise ValueError('training coverage/state mismatch')
            if complete['episodes_sha256']!=overlay['tasks'][task]['episodes_sha256']:raise ValueError('D2/O2 frozen list identity mismatch')
            for row,wanted in zip(rows,reg['episodes']):
                if any(row[k]!=v for k,v in wanted.items()) or row['image_adam_step']!=row['episode']:raise ValueError('training episode order/identity mismatch')
                if row['counts']['differentiable_inner']!=(row['episode'] if arm=='O2' else 0):raise ValueError('training inner count mismatch')
                json.dumps(row,allow_nan=False)
            if complete['counts']!=rows[-1]['counts'] or complete['final_sha256']!=finals[task][arm]:raise ValueError('training completion/final mismatch')
            measured['outer']+=rows[-1]['image_adam_step'];measured['inner']+=rows[-1]['counts']['differentiable_inner'];compute['train_'+task+'_'+arm]=complete['counts']
            public['training'][task][arm]={k:complete[k] for k in ['episodes','elapsed_seconds','peak_allocated_bytes','artifact_bytes']}
            public['training'][task][arm]['loss_blocks']=[dict(start=i+1,end=i+100,loss=distribution([r['loss'] for r in rows[i:i+100]]),image_gradient_norm=distribution([r['gradient_norm'] for r in rows[i:i+100]])) for i in range(0,600,100)]
        for stage in ('source','target'):
            selected=reg['source_query'] if stage=='source' else reg['target'];arms={}
            wanted=[dict(visit=i+1,sample_id=r['sample_id'],group_id=r['group_id'],segment='clean') for i,r in enumerate(selected)]
            for arm in NEW_ARMS:
                rows=read_lines(out/f'{stage}_{task}_{arm}.jsonl');arms[arm]=rows
                validate_new_arm(rows,wanted,task,arm)
                if any(r['domain']!=e['domain'] for r,e in zip(rows,selected)):raise ValueError('target domain/order mismatch')
                complete=json.loads((out/f'{stage}_{task}_{arm}.completion.json').read_text())
                totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
                if complete['records']!=len(rows) or complete['source_unchanged'] is not True or complete['counts']!=totals:raise ValueError('scoring completion mismatch')
                measured['records']+=len(rows);measured['online']+=totals['online_adam'];compute[stage+'_'+task+'_'+arm]=totals
                scoring[stage+'_'+task+'_'+arm]=dict(records=len(rows),pipeline_seconds=sum(r['pipeline_elapsed_seconds'] for r in rows),host_step_seconds=sum(r['host_step_elapsed_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
            validate_new_results(arms,wanted,task)
            absent=[]
            for label,original_arm in OLD_MAP.items():
                path=old/f'{stage}_{task}_{original_arm}.jsonl'
                if not path.exists():absent.append(label);continue
                rows=read_lines(path)
                validate_arm(rows,wanted,task,'B' if original_arm in ('R','D','O') else original_arm)
                if any((r['task'],r['arm'],r['domain'])!=(task,original_arm,e['domain']) for r,e in zip(rows,selected)):raise ValueError('M1 baseline stream provenance mismatch')
                complete=json.loads((old/f'{stage}_{task}_{original_arm}.completion.json').read_text())
                if complete['status']!='COMPLETE' or complete['records']!=len(rows) or not complete['source_unchanged']:raise ValueError('M1 baseline completion mismatch')
                arms[label]=rows;reused+=len(rows)
            public[stage][task]=summarize(arms,task)
            public[stage][task]['missing_old_raw_arms']=absent
            if absent:
                previous=json.loads((old/'public_aggregate.json').read_text())[stage][task]
                public[stage][task]['old_public_means_only']={label:previous['task_domain_macro_dice_percent'][OLD_MAP[label]] for label in absent}
    expected=sum(2*(len(r['source_query'])+len(r['target'])) for r in registration['tasks'].values())
    if measured!=dict(online=expected,outer=2400,inner=1200,records=expected) or any(done['progress'][k]!=v for k,v in measured.items()):raise ValueError('M2 independent budget mismatch')
    smoke=json.loads((out/'smoke.completion.json').read_text())
    for k in ('online','outer','inner'):measured[k]+=smoke['updates'][k]
    env=json.loads((out/'run.environment_comparison.json').read_text())
    status='M2_EPISODE_COVERAGE_COMPARISON_COMPLETE'
    audit=dict(status=status,experiment_execution_commit=receipt['commit'],updates_and_new_records=measured,reused_M1_records=reused,
        combined_display_records=expected+reused,history_rebuilt_online_steps=overlay['history_rebuilt_online_steps'],history_reused=True,
        actual_compute=compute,smoke=smoke,scoring_cost=scoring,gpu_seconds=done['gpu_seconds'],environment_comparison=env,
        private_output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),independent_CPU_recompute=True,exit_code=0)
    # Receipt binding stays private. Public smoke evidence omits the private registration digest.
    audit['smoke']={k:smoke[k] for k in ['status','updates','evidence','paired_comparison_backend','gpu_seconds','exit_code']}
    public.update(status=status,environment_comparison=env,scientific_interpretation='Assess O2-D2 and method-specific changes per task/domain; no automatic next experiment')
    private_json(out/'public_aggregate.json',public);private_json(out/'execution_audit.json',audit)
    private_json(out/'verification.json',dict(status=status,counts=measured,reused_M1_records=reused,exit_code=0))
