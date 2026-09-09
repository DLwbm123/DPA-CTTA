"""P2 scalar reconstruction with 16-cell complementarity and matched interactions."""
import json
import numpy as np
import torch
from .p2_data import ARMS,PARENTS,VIEWS,SUBSETS,expected
from .p1_analysis import evaluate
from .p1_host import check_probability
from .m1_analysis import read_lines,paired
from .host_diagnostic_analysis import channels,distribution
from .host_diagnostic_run import private_json
PAIRS=[('EA','A'),('EA','N'),('EO2','O2'),('ED4','D4'),('O2','A'),('D4','A'),('EO2','EA'),('ED4','EA'),('EO2','ED4')]


def pixel_cells(gt,q,b,e):
    for p in [q,b,e]:check_probability(p)
    if not gt.shape==q.shape==b.shape==e.shape or not torch.all((gt==0)|(gt==1)):raise ValueError('cell shapes/binary GT')
    code=8*gt.to(torch.int64).cpu()+4*(q.detach().cpu()>=.5).long()+2*(b.detach().cpu()>=.5).long()+(e.detach().cpu()>=.5).long()
    return [torch.bincount(code[0,c].flatten(),minlength=16).tolist() for c in range(code.shape[1])]


def marginal(cells,axis):
    table=np.asarray(cells,dtype=np.int64).reshape(2,2,2,2);gt=np.indices(table.shape)[0];pred=np.indices(table.shape)[axis]
    return dict(total_pixels=int(table.sum()),gt_pixels=int(table[gt==1].sum()),pred_pixels=int(table[pred==1].sum()),intersection=int(table[(gt==1)&(pred==1)].sum()))


def complement(cells):
    table=np.asarray(cells,dtype=np.int64).reshape(2,2,2,2);result={}
    for g,label in [(0,'background'),(1,'foreground')]:
        c=table[g];n,b,e=np.indices(c.shape)
        result[label]=dict(pixels=int(c.sum()),hard_disagreement=int(c[n!=b].sum()),N_wrong_B_correct=int(c[(n!=g)&(b==g)].sum()),retained_B_correct=int(c[(n!=g)&(b==g)&(e==g)].sum()),B_wrong_N_correct=int(c[(b!=g)&(n==g)].sum()),repaired_B_error=int(c[(b!=g)&(n==g)&(e==g)].sum()),new_FP_or_FN_vs_B=int(c[(b==g)&(e!=g)].sum()),corrected_FP_or_FN_vs_B=int(c[(b!=g)&(e==g)].sum()),agreement_changed=int(c[(n==b)&(e!=b)].sum()))
    return result


def validate_metric(m):
    na,nb,n,k=(m[v] for v in ['pred_pixels','gt_pixels','total_pixels','intersection'])
    if not all(type(v)==int for v in [na,nb,n,k]) or n<=0 or not 0<=k<=min(na,nb)<=max(na,nb)<=n:raise ValueError('pixel counts')
    if m['dice']!=(2*k/(na+nb) if na+nb else 1.):raise ValueError('Dice reconstruction')
    for key,value in dict(gt_empty=nb==0,gt_full=nb==n,pred_empty=na==0,pred_full=na==n).items():
        if type(m[key])!=bool or m[key]!=value:raise ValueError('metric flags')
    if (m['assd'] is None)!=(na==0 or nb==0) or (m['assd'] is not None and m['assd']<0):raise ValueError('ASSD undefined')


def validate_bundle(arms,stream,task,order):
    if set(arms)!=set(ARMS) or any(len(rs)!=len(stream) for rs in arms.values()):raise ValueError('coverage')
    if len({r['group_id'] for r in stream})!=len(stream):raise ValueError('duplicate expected group')
    for a,rows in arms.items():
        parent=VIEWS.get(a,a)
        for i,(r,row) in enumerate(zip(rows,stream)):
            values=dict(task=task,order=order,arm=a,visit=i+1,sample_id=row['sample_id'],group_id=row['group_id'],domain=row['domain'],subset=row['subset'],parent=parent,prediction_id=f'{task}:{order}:{i+1}:{a}')
            if any(r[k]!=v for k,v in values.items()):raise ValueError('row identity/order')
            up=int(a in PARENTS)
            if any(r['counts'][k]!=up for k in ['online_adam','backward_calls','memory_pushes']):raise ValueError('physical update count')
            if r['counts']['teacher_forwards']!=int(a=='N'):raise ValueError('teacher count')
            if parent!='N' and (r['parent_adam_step']!=i+1 or r['parent_counters']!=[i+1] or not 1<=r['parent_memory_size']<=min(41,i+1)):raise ValueError('parent lifecycle')
            if parent=='N' and (r['parent_adam_step']!=0 or r['parent_counters'] or r['parent_memory_size']!=0):raise ValueError('N lifecycle')
            if not r['source_unchanged'] or not r['prediction_fixed_before_label']:raise ValueError('source/label invariant')
            names=['OD','OC'] if task=='fundus' else ['polyp']
            if [m['channel'] for m in r['metrics']]!=names:raise ValueError('channels')
            for m in r['metrics']:validate_metric(m)
            if a in VIEWS:
                n,b=arms['N'][i],arms[parent][i]
                if r['parent_prediction_ids']!=[n['prediction_id'],b['prediction_id']] or r['formula']!='.5*q0+.5*p_parent':raise ValueError('view parent/formula')
                for k in ['parent_adam_step','parent_memory_size','parent_counters','parent_state_tag']:
                    if r[k]!=b[k]:raise ValueError('view state binding')
                if len(r['pixel_cells'])!=len(names):raise ValueError('cell channel coverage')
                for ch,cells in enumerate(r['pixel_cells']):
                    if len(cells)!=16 or any(type(x)!=int or x<0 for x in cells):raise ValueError('16-cell table')
                    for axis,other in [(1,n),(2,b),(3,r)]:
                        if any(other['metrics'][ch][k]!=v for k,v in marginal(cells,axis).items()):raise ValueError('cell marginal mismatch')
                if abs(r['probability_mean']-.5*n['probability_mean']-.5*b['probability_mean'])>1e-6:raise ValueError('probability mean formula')
            elif r['parent_prediction_ids'] or r['formula'] is not None or r['pixel_cells'] is not None:raise ValueError('raw provenance')
            json.dumps(r,allow_nan=False)
    return True


def summarize(arms,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp'];domains=list(dict.fromkeys(r['domain'] for r in arms['N']));result=dict(domains={},groups=len(arms['N']),task_domain_macro_dice_percent={},task_comparisons_pp={},task_interactions_pp={})
    for domain in domains:
        rows={a:[r for r in rs if r['domain']==domain] for a,rs in arms.items()};entry=result['domains'][domain]=dict(arms={},paired={},interactions={},complementarity={})
        for a,rs in rows.items():
            entry['arms'][a]={}
            for c in names:
                ms=channels(rs,c);v=dict(dice_percent=distribution([100*m['dice'] for m in ms]))
                if c!='macro':
                    valid=[m['assd'] for m in ms if m['assd'] is not None];v.update(assd_conditional_mean_px=float(np.mean(valid)) if valid else None,assd_defined=len(valid),assd_undefined=len(ms)-len(valid),**{k+'_count':sum(m[k] for m in ms) for k in ['gt_empty','gt_full','pred_empty','pred_full']})
                entry['arms'][a][c]=v
        for a,b in PAIRS:entry['paired'][a+'-'+b]={c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in names}
        for e,b in [('EO2','O2'),('ED4','D4')]:
            key=f'({e}-{b})-(EA-A)';entry['interactions'][key]={}
            for c in names:
                tuples=zip(*(channels(rows[a],c) for a in [e,b,'EA','A']))
                entry['interactions'][key][c]=dict(dice_delta_pp=distribution([100*((x['dice']-y['dice'])-(z['dice']-w['dice'])) for x,y,z,w in tuples]))
        for a in VIEWS:
            entry['complementarity'][a]={c:dict(cells=np.asarray([r['pixel_cells'][i] for r in rows[a]],dtype=np.int64).sum(0).tolist(),**complement(np.asarray([r['pixel_cells'][i] for r in rows[a]],dtype=np.int64).sum(0))) for i,c in enumerate(names if task!='fundus' else names[:2])}
    for a in ARMS:result['task_domain_macro_dice_percent'][a]=float(np.mean([result['domains'][d]['arms'][a][names[-1]]['dice_percent']['mean'] for d in domains])) if domains else None
    scores=result['task_domain_macro_dice_percent'];result['task_comparisons_pp']={a+'-'+b:scores[a]-scores[b] if domains else None for a,b in PAIRS}
    for e,b in [('EO2','O2'),('ED4','D4')]:result['task_interactions_pp'][f'({e}-{b})-(EA-A)']=(scores[e]-scores[b])-(scores['EA']-scores['A']) if domains else None
    result['pooled_content_paired']={a+'-'+b:{c:paired(channels(arms[a],c),channels(arms[b],c),c=='macro') for c in names} for a,b in PAIRS} if domains else {}
    return result


def recompute(out,reg,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='P2_RUN_COMPLETE':raise ValueError('incomplete run')
    public=dict(status='P2_FROZEN_FULL_STREAM_COMPLETE',execution_commit=receipt['commit'],coverage={t:r['counts'] for t,r in reg['tasks'].items()},target={},limitations=reg['limitations'])
    totals=dict(records=0,online=0,teacher_forwards=0,outer=0,inner=0);cost={}
    for task in reg['tasks']:
        public['target'][task]={}
        for order in [0,1]:
            stream=expected(reg,task,order);arms={a:read_lines(out/f'{task}_{order}_{a}.jsonl') for a in ARMS};validate_bundle(arms,stream,task,order)
            for a,rs in arms.items():
                totals['records']+=len(rs);totals['online']+=sum(r['counts']['online_adam'] for r in rs);totals['teacher_forwards']+=sum(r['counts']['teacher_forwards'] for r in rs)
                cost[f'{task}_{order}_{a}']=dict(records=len(rs),counts={k:sum(r['counts'][k] for r in rs) for k in rs[0]['counts']} if rs else {},**{k:sum(r[k] for r in rs) for k in ['host_step_elapsed_seconds','teacher_seconds','deployment_step_seconds','pipeline_elapsed_seconds']},shared_peak_allocated_bytes=max((r['peak_allocated_bytes'] for r in rs),default=0),resident_models=1 if a in ['N','A'] else 2 if a in ['O2','D4','EA'] else 3)
            public['target'][task]['order'+str(order)]={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()},task) for s in SUBSETS}
    b=reg['budget'];wanted={k:b[k] for k in totals}
    if totals!=wanted or done['progress']!=wanted:raise ValueError('independent counts')
    smoke=json.loads((out/'smoke.completion.json').read_text())
    audit=dict(status=public['status'],execution_commit=receipt['commit'],formal=totals,online_including_smoke=totals['online']+12,smoke={k:smoke[k] for k in ['status','online','evidence','gpu_seconds','exit_code']},cost=cost,gpu_seconds=done['gpu_seconds'],scalar_recompute=True,pixel_inference_repeated=False,cell_marginals_verified=True,private_bytes_at_recompute=sum(p.stat().st_size for p in out.iterdir() if p.is_file()),teacher_accounting='q0 shared once per visit; each independent ensemble charged teacher forward and residency',memory_accounting='measured peak is shared schedule; resident_models excludes prompt and counts proxy clone when present',exit_code=0)
    private_json(out/'public_aggregate.json',public);private_json(out/'execution_audit.json',audit);render(out,public,audit);private_json(out/'verification.json',dict(status=public['status'],formal=totals,exit_code=0))
    if torch.cuda.is_initialized():raise ValueError('CPU process initialized CUDA')


def render(out,public,audit):
    lines=['# P2 frozen factorial full-stream comparison','',public['status'],'','Execution commit: '+public['execution_commit'],'']
    for subset in SUBSETS:
        lines+=['## '+subset,'','| Task/order | Groups | '+' | '.join(ARMS)+' |','| --- | ---: | '+' | '.join(['---:']*len(ARMS))+' |']
        for task,orders in public['target'].items():
            for order,subsets in orders.items():
                r=subsets[subset];v=r['task_domain_macro_dice_percent'];lines+=['| '+task+'/'+order+' | '+str(r['groups'])+' | '+' | '.join(f'{v[a]:.6f}' if v[a] is not None else 'N/A' for a in ARMS)+' |']
        lines+=['','| Task/order | '+' | '.join(a+'-'+b for a,b in PAIRS)+' |','| --- | '+' | '.join(['---:']*len(PAIRS))+' |']
        for task,orders in public['target'].items():
            for order,subsets in orders.items():
                v=subsets[subset]['task_comparisons_pp'];lines+=['| '+task+'/'+order+' | '+' | '.join(f'{v[a+"-"+b]:+.6f}' if v[a+'-'+b] is not None else 'N/A' for a,b in PAIRS)+' |']
    lines+=['','## Audit and interpretation boundary','',json.dumps(audit['formal'])+'; additional smoke online=12.',
        'All paired distributions, domain/channel results, two matched interaction contrasts and foreground/background 16-cell counts are in public_aggregate.json. Cells reconstruct N/parent/view Dice and explain retained/repaired/new FP/FN; background counts are not presented as a replacement for foreground segmentation.',
        'Frozen A/O2/D4 are independent full streams. EA/EO2/ED4 use their own parent plus the same source q0, probabilities averaged at 0.5 then thresholded >=0.5. No GT gating, SA, new proxy training or new source evaluation.',
        'Independent CPU reconstruction verifies coverage, identities, parent/view binding, counters, every Dice and each 16-cell marginal; it does not reconstruct discarded pixel probabilities or independently recalculate ASSD. Undefined ASSD stays undefined; no macro ASSD.',
        'Shared schedule executes one teacher per visit; standalone ensemble costs include a source forward and extra model. Shared allocated peak is not an isolated per-method peak. See execution_audit.json for measured costs and forwards.',
        'remaining_dev is primary, not a globally untouched test. Known role exclusions preserved; UNKNOWN linkage disclosed. Single seed and two orders on the same contents do not establish patient-independent significance. P1 comparisons change both contents and histories.',
        'Completion is separate from scientific value. Apply predeclared interpretation after examining matched controls and tails; no automatic P3 or parameter search. Source data, masks, model/proxy files, private paths, identities and per-asset digests are not published.']
    with (out/'P2_EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
