"""Independent M4 CPU reconstruction; identity, order, counters and all paired metrics."""
import copy
import json
from pathlib import Path
import numpy as np
from .host_diagnostic_analysis import channels,distribution
from .m1_analysis import paired,read_lines
from .host_diagnostic_run import private_json
from .source_pilot import validate_arm
from .m4_run import TRAIN_ARMS,validate_new_arm
from .m4_sequences import target_order
PAIRS=[('T4','L4'),('T4','D4'),('L4','O2'),('T4','A'),('T4','D2'),('T4','O2'),('T4','N'),('T4','O3')]

def summarize(arms,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp'];primary=names[-1]
    domains=list(dict.fromkeys(row['domain'] for row in arms['T4']))
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
        for a,b in PAIRS:
            item['paired'][a+'-'+b]={c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in names} if a in rows and b in rows else 'NOT_AVAILABLE'
    for arm in arms:
        result['task_domain_macro_dice_percent'][arm]=float(np.mean([result['domains'][d]['arms'][arm][primary]['dice_percent']['mean'] for d in domains]))
    values=result['task_domain_macro_dice_percent']
    for a,b in PAIRS:result['task_comparisons_pp'][a+'-'+b]=values[a]-values[b] if a in values and b in values else None
    result['pooled_content_paired']={a+'-'+b:{c:paired(channels(arms[a],c),channels(arms[b],c),c=='macro') for c in names} if a in arms and b in arms else 'NOT_AVAILABLE' for a,b in PAIRS}
    return result


def wanted(selected):return [dict(visit=i+1,sample_id=r['sample_id'],group_id=r['group_id'],segment='clean') for i,r in enumerate(selected)]


def reuse(overlay,reg,task,stage,arm,order):
    parent=Path(overlay['m3_directory'] if arm=='O3' else overlay['m2_directory'] if arm in ['D2','O2'] else overlay['old_directory'])
    original=reg['source_query'] if stage=='source' else reg['target'];rows=read_lines(parent/f'{stage}_{task}_{arm}.jsonl')
    validate_arm(rows,wanted(original),task,arm if arm in ['N','A'] else 'B')
    if any((r['task'],r['arm'],r['domain'])!=(task,arm,e['domain']) for r,e in zip(rows,original)):raise ValueError('old provenance mismatch')
    done=json.loads((parent/f'{stage}_{task}_{arm}.completion.json').read_text())
    if done['status']!='COMPLETE' or done['records']!=len(rows) or not done['source_unchanged']:raise ValueError('old completion missing')
    if order:
        if arm!='N':raise ValueError('only stateless N may reuse reverse order')
        by_id={(r['sample_id'],r['group_id']):r for r in rows}
        if len(by_id)!=len(rows):raise ValueError('N identities duplicated')
        rows=[dict(by_id[(r['sample_id'],r['group_id'])],visit=i+1) for i,r in enumerate(target_order(original,1))]
    return rows


def recompute(out,overlay,registration,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='M4_RUN_COMPLETE':raise ValueError('M4 incomplete')
    measured=dict(online=0,outer=0,inner=0,source_visits=0,records=0);compute={};scoring={};reused=0
    public=dict(status='M4_TRAJECTORY_COMPARISON_COMPLETE',source={},target={'order0':{},'order1':{}},training={},provenance=dict(execution_commit=receipt['commit'],baseline_release='538a768363a79a5dca27de186e2b776c858c5e2b',new_arms=list(TRAIN_ARMS),reverse_baselines=['A','D2','O2'],N_reverse='stateless per-content metrics reordered on CPU; no new model run',training_history='own carried state, window truncated; not fixed Base library'))
    frozen=json.loads((out/'final_artifacts.frozen.json').read_text())
    for task,reg in registration['tasks'].items():
        public['training'][task]={}
        for arm in TRAIN_ARMS:
            rows=read_lines(out/f'train_{task}_{arm}.jsonl');completion=json.loads((out/f'train_{task}_{arm}.completion.json').read_text())
            if len(rows)!=150 or completion['source_visits']!=600 or completion['outer_steps']!=150 or not completion['source_unchanged']:raise ValueError('training completion')
            if completion['sequence_sha256']!=overlay['tasks'][task]['sequence']['sha256'] or completion['final_sha256']!=frozen[task][arm]:raise ValueError('training sequence/final identity')
            for i,row in enumerate(rows):
                if row['window']!=i+1 or row['image_adam_step']!=i+1 or row['sequence']!=reg['sequence'][i*4:(i+1)*4] or row['count']!=(i%30+1)*4:raise ValueError('training row identity/order/state')
                if row['stream']!=i//30:raise ValueError('stream reset boundary')
                for k in ['source_visits','differentiable_inner','memory_pushes']:
                    if row['counts'][k]!=(i+1)*4:raise ValueError('training update counts')
                json.dumps(row,allow_nan=False)
            if completion['counts']!=rows[-1]['counts']:raise ValueError('training counter final')
            measured['outer']+=150;measured['inner']+=rows[-1]['counts']['differentiable_inner'];measured['source_visits']+=rows[-1]['counts']['source_visits']
            compute['train_'+task+'_'+arm]=completion['counts']
            public['training'][task][arm]={k:completion[k] for k in ['source_visits','outer_steps','elapsed_seconds','peak_allocated_bytes','artifact_bytes']}
            public['training'][task][arm]['blocks']=[dict(start=i+1,end=i+30,loss=distribution([r['loss'] for r in rows[i:i+30]]),gradient_norm=distribution([r['gradient_norm'] for r in rows[i:i+30]]),image_delta_norm=distribution([r['image_delta_norm'] for r in rows[i:i+30]]),zero_gradient_fraction=sum(r['gradient_norm']==0 for r in rows[i:i+30])/30) for i in range(0,150,30)]
        for stage,order,new in [('source',0,TRAIN_ARMS),('target',0,TRAIN_ARMS),('target',1,(*TRAIN_ARMS,'A','D2','O2'))]:
            selected=reg['source_query'] if stage=='source' else target_order(reg['target'],order);arms={}
            for arm in new:
                name=f'{stage}{order}_{task}_{arm}';rows=read_lines(out/(name+'.jsonl'));validate_new_arm(rows,wanted(selected),task,arm)
                if any(r['domain']!=e['domain'] or r['order']!=order for r,e in zip(rows,selected)):raise ValueError('score domain/order')
                c=json.loads((out/(name+'.completion.json')).read_text());totals={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']}
                if c['records']!=len(rows) or c['counts']!=totals or not c['source_unchanged']:raise ValueError('score completion')
                measured['records']+=len(rows);measured['online']+=totals['online_adam'];compute[name]=totals;arms[arm]=rows
                scoring[name]=dict(records=len(rows),pipeline_seconds=sum(r['pipeline_elapsed_seconds'] for r in rows),host_step_seconds=sum(r['host_step_elapsed_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
            for arm in (['N','A','R','D2','O2','O3'] if order==0 else ['N']):
                arms[arm]=reuse(overlay,reg,task,stage,arm,order);reused+=len(arms[arm])
            result=summarize(arms,task)
            result['old_cost_provenance']='Order0 historical arms retain historical costs. Reverse N costs are historical metadata of reused predictions, not new execution.'
            if stage=='source':public['source'][task]=result
            else:public['target']['order'+str(order)][task]=result
    if measured!=dict(online=2172,outer=900,inner=3600,source_visits=3600,records=2172) or any(done['progress'][k]!=v for k,v in measured.items()):raise ValueError('independent budget mismatch')
    smoke=json.loads((out/'smoke.completion.json').read_text())
    for k in ['online','outer','inner']:measured[k]+=smoke['updates'][k]
    env=json.loads((out/'run.environment_comparison.json').read_text())
    audit=dict(status=public['status'],execution_commit=receipt['commit'],updates_and_new_records=measured,reused_records_display_occurrences=reused,actual_compute=compute,scoring_cost=scoring,
        smoke={k:smoke[k] for k in ['status','updates','evidence','paired_comparison_backend','gpu_seconds','exit_code']},gpu_seconds=done['gpu_seconds'],environment_comparison=env,
        private_output_bytes=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),independent_CPU_recompute=True,exit_code=0)
    public['scientific_interpretation']='Assess T4-L4/D4 across both tasks and orders, strong historical controls and tails jointly; positive means alone do not establish useful or stable gain; no automatic M5.'
    private_json(out/'public_aggregate.json',public);private_json(out/'execution_audit.json',audit);render_report(out,public,audit)
    private_json(out/'verification.json',dict(status=public['status'],counts=measured,reused_records_display_occurrences=reused,exit_code=0))


def render_report(out,public,audit):
    lines=['# M4 trajectory distillation report','',public['status'],'','Execution commit: '+audit['execution_commit'],'',
        'Six source-only fits completed. Each fit used 600 image visits, 150 outer updates and 600 functional inner Adam updates. Four distinct images per window; K=4 proxy batch unchanged. D4/L4 detach each incoming image state; T4 retains within-window Adam and memory-value derivatives, including the host gradient under native AdaBN stop-gradient statistics. All use native identity-proxy online deployment.','']
    for order,results in public['target'].items():
        lines+=['## Target '+order,'','| Task / domain | N | A | D2 | O2 | D4 | L4 | T4 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for task,result in results.items():
            v=result['task_domain_macro_dice_percent'];cols=['N','A','D2','O2','D4','L4','T4']
            lines.append('| '+task+' / domain-equal mean | '+' | '.join(f'{v[a]:.6f}' for a in cols)+' |')
            channel='macro' if task=='fundus' else 'polyp'
            for domain,d in result['domains'].items():lines.append('| '+task+' / '+domain+' | '+' | '.join(f"{d['arms'][a][channel]['dice_percent']['mean']:.6f}" for a in cols)+' |')
        lines+=['','| Task / domain | '+' | '.join(a+'-'+b for a,b in PAIRS)+' |','| --- | '+' | '.join(['---:']*len(PAIRS))+' |']
        for task,result in results.items():
            v=result['task_comparisons_pp'];lines.append('| '+task+' / mean | '+' | '.join(f'{v[a+"-"+b]:+.6f}' if v[a+'-'+b] is not None else 'N/A' for a,b in PAIRS)+' |')
            channel='macro' if task=='fundus' else 'polyp'
            for domain,d in result['domains'].items():
                lines.append('| '+task+' / '+domain+' | '+' | '.join(f"{d['paired'][a+'-'+b][channel]['dice_delta_pp']['mean']:+.6f}" if isinstance(d['paired'][a+'-'+b],dict) else 'N/A' for a,b in PAIRS)+' |')
    lines+=['','## Source-only preservation (fresh host; not post-target forgetting)','','| Task | N | A | D2 | O2 | O3 | D4 | L4 | T4 |','| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for task,result in public['source'].items():
        v=result['task_domain_macro_dice_percent'];lines.append('| '+task+' | '+' | '.join(f'{v[a]:.6f}' for a in ['N','A','D2','O2','O3','D4','L4','T4'])+' |')
    lines+=['','## Training cost','','| Task / arm | Image visits | Outer updates | Seconds | Peak allocated bytes |','| --- | ---: | ---: | ---: | ---: |']
    for task,arms in public['training'].items():
        for arm,v in arms.items():lines.append(f"| {task} / {arm} | {v['source_visits']} | {v['outer_steps']} | {v['elapsed_seconds']:.3f} | {v['peak_allocated_bytes']} |")
    lines+=['','## Incremental scoring cost','','| Stream | Visits | Host seconds/image | Pipeline seconds | Peak allocated bytes |','| --- | ---: | ---: | ---: | ---: |']
    for name,v in audit['scoring_cost'].items():lines.append(f"| {name} | {v['records']} | {v['host_step_seconds']/v['records']:.6f} | {v['pipeline_seconds']:.3f} | {v['peak_allocated_bytes']} |")
    lines+=['','## Evidence and interpretation limits','',
        'Update counts including smoke: '+json.dumps(audit['updates_and_new_records'])+'.',
        f"GPU-stage wall seconds: {audit['gpu_seconds']:.3f}; private bytes measured at recompute: {audit['private_output_bytes']}. Old records reused for display: {audit['reused_records_display_occurrences']} (includes reverse-order stateless N reuse).",
        'The [aggregate](public_aggregate.json) contains OD/OC, all per-domain paired means/medians/signs/worst-decile and worst-single Dice differences, ASSD common-valid/missing counts/adverse tails and empty/full outcomes. ASSD is in pixels; no macro ASSD or zero imputation. The [audit](execution_audit.json) separates native online, outer, functional inner, forwards, memory and incremental runtime.',
        'T4-L4 is the matched derivative comparison. L4-O2 and T4-O2 also change collected history, sequence organization and aggregation frequency; they cannot isolate on-policy effects or cross-step derivatives. Own evolving histories use earlier S versions and four-step truncation, not a fully recomputed on-policy prefix or 120-step BPTT.',
        'Tasks are not pooled. Orders contain the same target contents, not independent patients or independent replicate datasets. Targets were previously exposed; patient/video linkage remains UNKNOWN. Source scores use a fresh source host and do not measure forgetting after target adaptation. Historical timings are not matched hardware/FLOPs comparisons. No target checkpoint, seed, style or domainwise baseline selection.',
        'Completion is separate from scientific merit. Interpret joint comparisons and tails; do not auto-label a positive mean NET_GAIN. No additional M5, style/norm/sampler search or automatic retry is authorized. Raw images/masks, sample identities/logs, synthetic proxies, checkpoints and state histories remain private.',
        'Prior multi-step distillation: [Cazenavette et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPR_2022_paper.html). This experiment does not claim to invent trajectory distillation.','']
    with (out/'M4_EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(lines))
