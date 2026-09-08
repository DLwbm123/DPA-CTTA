"""CPU-only reconstruction: 1104 new records, with separately attributed M1 baselines."""
import json
from pathlib import Path
import numpy as np
from .host_diagnostic_analysis import channels,distribution
from .m1_analysis import paired,read_lines
from .host_diagnostic_run import private_json
from .source_pilot import validate_arm
from .m3_run import NEW_ARMS,TRAIN_ARMS,validate_new_arm

OLD_MAP={'N':'N','A':'A','R':'R','D1':'D','O1':'O','D2':'D2','O2':'O2'}
PAIRS=[('O3','D3'),('O3','R3'),('O3','O2T'),('O2T','O2'),('R3','R'),('O3','A'),('O3','N'),('D3','D2'),('O3','O2')]


def validate_new_results(results,expected,task):
    if set(results)!=set(NEW_ARMS):raise ValueError('only complete R3/D3/O3/O2T new set accepted')
    for arm,rows in results.items():validate_new_arm(rows,expected,task,arm)
    return True


def summarize(arms,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp'];primary=names[-1]
    domains=list(dict.fromkeys(row['domain'] for row in arms['D3']))
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
        for arm,records in rows.items():
            item['arms'][arm]['cost']={k:sum(r.get(k,0.) for r in records) for k in ['pipeline_elapsed_seconds','host_step_elapsed_seconds']}
            item['arms'][arm]['cost']['peak_allocated_bytes']=max(r.get('peak_allocated_bytes',0) for r in records)
            if arm in NEW_ARMS:item['arms'][arm]['condition_stats']={k:distribution([r['condition_stats'][k] for r in records]) for k in ['clipping_fraction','change_l2']}
        for a,b in PAIRS:
            item['paired'][a+'-'+b]={c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in names} if a in rows and b in rows else 'NOT_AVAILABLE'
    for arm in arms:
        result['task_domain_macro_dice_percent'][arm]=float(np.mean([result['domains'][d]['arms'][arm][primary]['dice_percent']['mean'] for d in domains]))
    values=result['task_domain_macro_dice_percent']
    for a,b in PAIRS:result['task_comparisons_pp'][a+'-'+b]=values[a]-values[b] if a in values and b in values else None
    result['pooled_content_paired']={a+'-'+b:{c:paired(channels(arms[a],c),channels(arms[b],c),c=='macro') for c in names} if a in arms and b in arms else 'NOT_AVAILABLE' for a,b in PAIRS}
    return result


def recompute(out,overlay,registration,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='M3_RUN_COMPLETE':raise ValueError('M3 run completion missing')
    old=Path(overlay['old_directory']);m2=Path(overlay['m2_directory']);public={'source':{},'target':{},'training':{},'coverage':{t:r['coverage'] for t,r in overlay['tasks'].items()},
        'provenance':{'new_arms':list(NEW_ARMS),'reused_M1_M2_arms':list(OLD_MAP),'M1_execution_commit':'d92ed88e603eb68aea97a57493c96a084e98d91c','M2_execution_commit':'fef00f5bb1ea9c557054215ebe00ca5ee932ab81','new_execution_commit':receipt['commit']}}
    measured=dict(online=0,outer=0,inner=0,records=0);reused=0;compute={};scoring={};finals=json.loads((out/'final_artifacts.frozen.json').read_text())
    for task,reg in registration['tasks'].items():
        public['training'][task]={}
        for arm in TRAIN_ARMS:
            rows=read_lines(out/f'train_{task}_{arm}.jsonl');complete=json.loads((out/f'train_{task}_{arm}.completion.json').read_text())
            if len(rows)!=600 or complete['episodes']!=600 or complete['source_unchanged'] is not True:raise ValueError('training coverage/state mismatch')
            if complete['episodes_sha256']!=overlay['tasks'][task]['episodes_sha256']:raise ValueError('D3/O3 frozen list identity mismatch')
            for row,wanted in zip(rows,reg['episodes']):
                if any(row[k]!=v for k,v in wanted.items()) or row['image_adam_step']!=row['episode']:raise ValueError('training episode order/identity mismatch')
                if row['counts']['differentiable_inner']!=(row['episode'] if arm=='O3' else 0):raise ValueError('training inner count mismatch')
                if row['counts']['condition_transforms']!=row['episode']:raise ValueError('offline T count mismatch')
                json.dumps(row,allow_nan=False)
            if complete['counts']!=rows[-1]['counts'] or complete['final_sha256']!=finals[task][arm]:raise ValueError('training completion/final mismatch')
            measured['outer']+=rows[-1]['image_adam_step'];measured['inner']+=rows[-1]['counts']['differentiable_inner'];compute['train_'+task+'_'+arm]=complete['counts']
            public['training'][task][arm]={k:complete[k] for k in ['episodes','elapsed_seconds','peak_allocated_bytes','artifact_bytes']}
            public['training'][task][arm]['loss_blocks']=[dict(start=i+1,end=i+100,loss=distribution([r['loss'] for r in rows[i:i+100]]),image_gradient_norm=distribution([r['gradient_norm'] for r in rows[i:i+100]]),image_delta_norm=distribution([r['image_delta_norm'] for r in rows[i:i+100]]),zero_gradient_fraction=sum(r['gradient_norm']==0 for r in rows[i:i+100])/100,condition_stats={k:distribution([r['condition_stats'][k] for r in rows[i:i+100]]) for k in ['clipping_fraction','change_l2']}) for i in range(0,600,100)]
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
                baseline=m2 if original_arm in ('D2','O2') else old
                path=baseline/f'{stage}_{task}_{original_arm}.jsonl'
                if not path.exists():absent.append(label);continue
                rows=read_lines(path)
                validate_arm(rows,wanted,task,'B' if original_arm in ('R','D','O','D2','O2') else original_arm)
                if any((r['task'],r['arm'],r['domain'])!=(task,original_arm,e['domain']) for r,e in zip(rows,selected)):raise ValueError('M1 baseline stream provenance mismatch')
                complete=json.loads((baseline/f'{stage}_{task}_{original_arm}.completion.json').read_text())
                if complete['status']!='COMPLETE' or complete['records']!=len(rows) or not complete['source_unchanged']:raise ValueError('M1 baseline completion mismatch')
                arms[label]=rows;reused+=len(rows)
            public[stage][task]=summarize(arms,task)
            public[stage][task]['missing_old_raw_arms']=absent
            if absent:raise ValueError('Required old records unavailable; paired comparison cannot be reconstructed')
    expected=sum(4*(len(r['source_query'])+len(r['target'])) for r in registration['tasks'].values())
    if measured!=dict(online=expected,outer=2400,inner=1200,records=expected) or any(done['progress'][k]!=v for k,v in measured.items()):raise ValueError('M3 independent budget mismatch')
    smoke=json.loads((out/'smoke.completion.json').read_text())
    for k in ('online','outer','inner'):measured[k]+=smoke['updates'][k]
    env=json.loads((out/'run.environment_comparison.json').read_text())
    status='M3_CONDITIONED_PROXY_COMPLETE'
    audit=dict(status=status,experiment_execution_commit=receipt['commit'],updates_and_new_records=measured,reused_M1_M2_records=reused,
        combined_display_records=expected+reused,history_rebuilt_online_steps=overlay['history_rebuilt_online_steps'],history_reused=True,
        actual_compute=compute,smoke=smoke,scoring_cost=scoring,gpu_seconds=done['gpu_seconds'],environment_comparison=env,
        private_output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),independent_CPU_recompute=True,exit_code=0)
    # Receipt binding stays private. Public smoke evidence omits the private registration digest.
    audit['smoke']={k:smoke[k] for k in ['status','updates','evidence','paired_comparison_backend','gpu_seconds','exit_code']}
    public.update(status=status,environment_comparison=env,scientific_interpretation='Explore O3-D3/R3/O2T and net benefits per task/domain; no automatic NET_GAIN or M4')
    private_json(out/'public_aggregate.json',public);private_json(out/'execution_audit.json',audit)
    render_report(out,public,audit)
    private_json(out/'verification.json',dict(status=status,counts=measured,reused_M1_M2_records=reused,exit_code=0))


def render_report(out,public,audit):
    arms=['N','A','R','D2','O2','R3','O2T','D3','O3'];lines=['# M3 conditioned proxy experiment report','',public['status'],'',
        'Execution commit: '+audit['experiment_execution_commit'],'',
        'Exploratory single-seed/order comparison. Completion is not evidence of superiority. FDA-style partial mixing is an existing method family, not a new claim of style-transfer novelty.','',
        '## Absolute Dice (%)','', '| Split / task / domain / channel | '+' | '.join(arms)+' |','| --- | '+' | '.join(['---:']*len(arms))+' |']
    for stage in ('source','target'):
        for task,result in public[stage].items():
            v=result['task_domain_macro_dice_percent']
            lines.append('| '+stage+' / '+task+' / domain-equal mean | '+' | '.join(f'{v[a]:.6f}' for a in arms)+' |')
            for domain,d in result['domains'].items():
                for c in ['OD','OC','macro'] if task=='fundus' else ['polyp']:
                    lines.append('| '+stage+' / '+task+' / '+domain+' / '+c+' | '+' | '.join(f"{d['arms'][a][c]['dice_percent']['mean']:.6f}" for a in arms)+' |')
    lines+=['','## Target paired differences (percentage points)','','| Task / domain | '+' | '.join(a+'-'+b for a,b in PAIRS)+' |','| --- | '+' | '.join(['---:']*len(PAIRS))+' |']
    for task,result in public['target'].items():
        lines.append('| '+task+' / domain-equal mean | '+' | '.join(f"{result['task_comparisons_pp'][a+'-'+b]:+.6f}" for a,b in PAIRS)+' |')
        c='macro' if task=='fundus' else 'polyp'
        for domain,d in result['domains'].items():lines.append('| '+task+' / '+domain+' | '+' | '.join(f"{d['paired'][a+'-'+b][c]['dice_delta_pp']['mean']:+.6f}" for a,b in PAIRS)+' |')
    lines+=['','## Historical appendix (Dice %)','','| Split / task / domain | D1 | O1 |','| --- | ---: | ---: |']
    for stage in ('source','target'):
        for task,result in public[stage].items():
            c='macro' if task=='fundus' else 'polyp'
            for domain,d in result['domains'].items():lines.append('| '+stage+' / '+task+' / '+domain+' | '+' | '.join(f"{d['arms'][a][c]['dice_percent']['mean']:.6f}" for a in ['D1','O1'])+' |')
    lines+=['','## Training and online cost','','| Task / method | Episodes | Training seconds | Peak bytes |','| --- | ---: | ---: | ---: |']
    for task,methods in public['training'].items():
        for arm,v in methods.items():lines.append(f"| {task} / {arm} | {v['episodes']} | {v['elapsed_seconds']:.3f} | {v['peak_allocated_bytes']} |")
    lines+=['','| New scoring stream | Records | Host step seconds including T | Pipeline seconds | Peak bytes |','| --- | ---: | ---: | ---: | ---: |']
    for name,v in audit['scoring_cost'].items():lines.append(f"| {name} | {v['records']} | {v['host_step_seconds']:.3f} | {v['pipeline_seconds']:.3f} | {v['peak_allocated_bytes']} |")
    lines+=['','## Evidence and limits','',
        'Counts including smoke: '+json.dumps(audit['updates_and_new_records'])+'.',
        f"Reused old records: {audit['reused_M1_M2_records']}; combined display: {audit['combined_display_records']}. GPU-stage seconds: {audit['gpu_seconds']:.3f}. Private bytes at recompute: {audit['private_output_bytes']}.",
        'The [public aggregate](public_aggregate.json) preserves all paired medians/signs/worst-decile Dice differences, OD/OC, shared ASSD cohorts/missing counts/adverse tails, D1/O1 historical arms, 100-episode losses, zero-gradient fractions, image changes, clipping and conditioning L2. [Execution audit](execution_audit.json) contains forward/backward/transform/retrieval/push and three distinct update ledgers.',
        'D/O losses are different objectives. Fixed 600 steps do not prove convergence. Historical timing is shared-GPU observational evidence, not a controlled hardware benchmark. Source/target labels never enter online conditioning; target labels are accessed after prediction.',
        'O2T controls test-only conditioning; no D2T was run. D3-D2 and O3-O2 mix training and deployment changes. Phase retention does not prove semantic retention or privacy. UNKNOWN patient/video linkage, previous target exposure, fixed off-policy Base history and one-step truncation remain limitations.',
        'No automatic NET_GAIN based solely on a positive mean. Assess domain cancellations and O3-D3/R3/O2T jointly; no automatic next round.',
        '', 'References: [Yang and Soatto, FDA (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_FDA_Fourier_Domain_Adaptation_for_Semantic_Segmentation_CVPR_2020_paper.html); [Kang et al., ICML 2023](https://proceedings.mlr.press/v202/kang23a.html).','']
    with (out/'M3_EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(lines))
