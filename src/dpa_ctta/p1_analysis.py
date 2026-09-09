"""Pixel evaluation at runtime; independent CPU reconstruction from scalar records."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from .p1_data import ARMS,PARENTS,expected
from .p1_host import check_probability
from .m1_analysis import paired,read_lines
from .host_diagnostic_analysis import channels,distribution
from .host_diagnostic_run import private_json

PAIRS=[('SA','A'),('ENS_SA','ENS_A'),('ENS_A','A'),('ENS_A','N'),('ENS_SA','SA'),('A','N')]+[(a,b) for a in ['SA','ENS_SA'] for b in ['N','O2','D4','L4']]


def evaluate(probability,mask,task):
    from scipy.ndimage import binary_erosion,distance_transform_edt
    p=probability.detach().cpu();check_probability(p)
    if p.shape!=mask.shape or p.shape[:2]!=(1,2 if task=='fundus' else 1) or not torch.all((mask==0)|(mask==1)):raise ValueError('metric shape/binary mask')
    hard=(p>=.5).numpy();gt=mask.bool().numpy();out=[]
    for c,name in enumerate(['OD','OC'] if task=='fundus' else ['polyp']):
        a,b=hard[0,c],gt[0,c];na,nb=int(a.sum()),int(b.sum());intersection=int((a&b).sum());assd=None
        if na and nb:
            sa=a^binary_erosion(a,border_value=0);sb=b^binary_erosion(b,border_value=0)
            assd=float(np.concatenate((distance_transform_edt(~sb)[sa],distance_transform_edt(~sa)[sb])).mean())
        out.append(dict(channel=name,dice=2*intersection/(na+nb) if na+nb else 1.,assd=assd,
            gt_empty=nb==0,gt_full=nb==b.size,pred_empty=na==0,pred_full=na==a.size,
            pred_pixels=na,gt_pixels=nb,intersection=intersection,total_pixels=a.size))
    return out


def validate_rows(rows,stream,task,order,arm):
    if len(rows)!=len(stream):raise ValueError('record coverage')
    for i,(r,e) in enumerate(zip(rows,stream)):
        if any(r[k]!=v for k,v in dict(visit=i+1,task=task,order=order,arm=arm,domain=e['domain'],sample_id=e['sample_id'],group_id=e['group_id'],subset=e['subset'],parent=PARENTS[arm]).items()):raise ValueError('identity/order/parent mismatch')
        parent=PARENTS[arm];up=int(arm in ['A','SA','O2','D4','L4'])
        if r['counts']['online_adam']!=up or r['counts']['memory_pushes']!=up or r['counts']['backward_calls']!=up:raise ValueError('update count')
        if parent!='N' and (r['parent_adam_step']!=i+1 or r['parent_counters']!=[i+1] or not 1<=r['parent_memory_size']<=min(i+1,41)):raise ValueError('parent lifecycle')
        if r['source_unchanged'] is not True or r['prediction_fixed_before_label'] is not True:raise ValueError('invariants')
        if r['prediction_id']!=f'{task}:{order}:{i+1}:{arm}':raise ValueError('prediction identity')
        wanted=[f'{task}:{order}:{i+1}:N',f'{task}:{order}:{i+1}:{parent}'] if arm.startswith('ENS_') else []
        if r['parent_prediction_ids']!=wanted or r['formula']!=('.5*q0+.5*p_parent' if wanted else None):raise ValueError('ensemble provenance')
        names=['OD','OC'] if task=='fundus' else ['polyp']
        if [m['channel'] for m in r['metrics']]!=names:raise ValueError('metric channels')
        for m in r['metrics']:
            na,nb,n,k=(m[v] for v in ['pred_pixels','gt_pixels','total_pixels','intersection'])
            if not all(type(v)==int for v in [na,nb,n,k]) or n<=0 or not 0<=k<=min(na,nb)<=max(na,nb)<=n:raise ValueError('pixel counts')
            if m['dice']!=(2*k/(na+nb) if na+nb else 1.):raise ValueError('Dice scalar reconstruction')
            if any(m[v]!=b for v,b in dict(gt_empty=nb==0,gt_full=nb==n,pred_empty=na==0,pred_full=na==n).items()):raise ValueError('empty/full')
            if (m['assd'] is None)!=(na==0 or nb==0) or (m['assd'] is not None and m['assd']<0):raise ValueError('ASSD validity')
        json.dumps(r,allow_nan=False)
    return True


def summarize(arms,task):
    names=['OD','OC','macro'] if task=='fundus' else ['polyp'];primary=names[-1]
    domains=list(dict.fromkeys(r['domain'] for r in arms['N']));out=dict(domains={},task_domain_macro_dice_percent={},task_comparisons_pp={})
    for domain in domains:
        rows={a:[r for r in rs if r['domain']==domain] for a,rs in arms.items()};item=out['domains'][domain]=dict(arms={},paired={})
        for a,rs in rows.items():
            item['arms'][a]={}
            for c in names:
                ms=channels(rs,c);v=dict(dice_percent=distribution([100*m['dice'] for m in ms]))
                if c!='macro':
                    valid=[m['assd'] for m in ms if m['assd'] is not None]
                    v.update(assd_conditional_mean_px=float(np.mean(valid)) if valid else None,assd_defined=len(valid),assd_undefined=len(ms)-len(valid),**{k+'_count':sum(m[k] for m in ms) for k in ['gt_empty','gt_full','pred_empty','pred_full']})
                item['arms'][a][c]=v
        for a,b in PAIRS:item['paired'][a+'-'+b]={c:paired(channels(rows[a],c),channels(rows[b],c),c=='macro') for c in names}
    for a in arms:out['task_domain_macro_dice_percent'][a]=float(np.mean([out['domains'][d]['arms'][a][primary]['dice_percent']['mean'] for d in domains])) if domains else None
    v=out['task_domain_macro_dice_percent']
    out['task_comparisons_pp']={a+'-'+b:v[a]-v[b] if domains else None for a,b in PAIRS}
    out['coverage_domains']=domains;out['groups']=len(arms['N'])
    return out


def recompute(out,reg,receipt):
    done=json.loads((out/'run.completion.json').read_text())
    if done['status']!='P1_RUN_COMPLETE':raise ValueError('incomplete execution')
    public=dict(status='P1_NO_DD_COMPARISON_COMPLETE',execution_commit=receipt['commit'],counts={t:r['counts'] for t,r in reg['tasks'].items()},target={},limitations=reg['limitations']);totals=dict(records=0,online=0,outer=0,inner=0,teacher_forwards=0);cost={}
    for task in reg['tasks']:
        public['target'][task]={}
        for order in [0,1]:
            stream=expected(reg,task,order);arms={a:read_lines(out/f'{task}_{order}_{a}.jsonl') for a in ARMS}
            for arm,rows in arms.items():
                validate_rows(rows,stream,task,order,arm);totals['records']+=len(rows)
                totals['online']+=sum(r['counts']['online_adam'] for r in rows);totals['teacher_forwards']+=sum(r['counts']['teacher_forwards'] for r in rows)
                cost[f'{task}_{order}_{arm}']={k:sum(r[k] for r in rows) for k in ['teacher_seconds','host_step_elapsed_seconds','deployment_step_seconds','pipeline_elapsed_seconds']}
                cost[f'{task}_{order}_{arm}'].update(records=len(rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),counts={k:sum(r['counts'][k] for r in rows) for k in rows[0]['counts']})
                for i,r in enumerate(rows):
                    if arm.startswith('ENS_'):
                        parent=arms[PARENTS[arm]][i]
                        for k in ['parent_adam_step','parent_counters','parent_memory_size','anchor','parent_state_tag']:
                            if r[k]!=parent[k]:raise ValueError('ensemble parent mismatch')
                        if r['probability_mean']!=.5*arms['N'][i]['probability_mean']+.5*parent['probability_mean']:
                            # Reduction of the pixel mean can differ by float32 rounding.
                            if abs(r['probability_mean']-(.5*arms['N'][i]['probability_mean']+.5*parent['probability_mean']))>1e-6:raise ValueError('ensemble scalar mean')
            public['target'][task]['order'+str(order)]={subset:summarize({a:[r for r in rows if subset=='combined' or r['subset']==subset] for a,rows in arms.items()},task) for subset in ['extension_dev','legacy_dev','combined']}
    budget=reg['budget']
    if totals!=dict(records=budget['records'],online=budget['online'],outer=0,inner=0,teacher_forwards=2*budget['groups']) or totals!=done['progress']:raise ValueError('independent count mismatch')
    smoke=json.loads((out/'smoke.completion.json').read_text())
    audit=dict(status=public['status'],execution_commit=receipt['commit'],formal=totals,smoke={k:smoke[k] for k in ['status','online','evidence','gpu_seconds','exit_code'] if k in smoke},online_including_smoke=totals['online']+12,cost=cost,gpu_seconds=done['gpu_seconds'],scalar_recompute=True,pixel_inference_repeated=False,ensemble_formula_checked_live=True,teacher_cost='Shared q0 once per visit. N/SA/ENS views each charged one source forward for standalone deployment; physical count only on N rows.',memory_cost='A/SA/N/ensemble peak is shared-bundle peak, not isolated per-arm allocation.',exit_code=0)
    private_json(out/'public_aggregate.json',public);private_json(out/'execution_audit.json',audit)
    render(out,public,audit);private_json(out/'verification.json',dict(status=public['status'],formal=totals,exit_code=0))
    if torch.cuda.is_initialized():raise ValueError('CPU recompute initialized CUDA')


def render(out,public,audit):
    lines=['# P1 no-DD current-image source anchor','',public['status'],'','Execution commit: '+public['execution_commit'],'','No offline training; lambda=0.1, probability averaging=0.5 fixed. Extension development is primary; it is not an untouched or patient-independent test.','']
    for subset in ['extension_dev','legacy_dev','combined']:
        lines += ['## '+subset,'','| Task/order | Groups | '+' | '.join(ARMS)+' |','| --- | ---: | '+' | '.join(['---:']*len(ARMS))+' |']
        for task,orders in public['target'].items():
            for order,result in orders.items():
                r=result[subset];v=r['task_domain_macro_dice_percent'];lines+=['| '+task+'/'+order+' | '+str(r['groups'])+' | '+' | '.join(f'{v[a]:.6f}' if v[a] is not None else 'N/A' for a in ARMS)+' |']
        lines+=['','| Task/order | SA-A | ENS_SA-ENS_A | ENS_A-A | ENS_A-N |','| --- | ---: | ---: | ---: | ---: |']
        for task,orders in public['target'].items():
            for order,result in orders.items():
                v=result[subset]['task_comparisons_pp'];lines+=['| '+task+'/'+order+' | '+' | '.join(f'{v[k]:+.6f}' if v[k] is not None else 'N/A' for k in ['SA-A','ENS_SA-ENS_A','ENS_A-A','ENS_A-N'])+' |']
    lines+=['','## Audit and limits','',json.dumps(audit['formal'])+'; smoke online=12; outer/inner/teacher training=0.',
        'Task means are equal-domain (Fundus OD/OC macro). All domain/channel/subset paired distributions, signs, worst decile/single, ASSD common-valid and adverse tails are in public_aggregate.json; undefined ASSD is never zero-imputed.',
        'Eight predictions use five adapting trajectories. Ensemble rows carry zero physical updates and bind to parent prediction/state identifiers. Source predictions are shared within a visit; standalone costs charge teacher computation to each applicable method.',
        'GT is read only after N/A/SA/ENS_A/ENS_SA are fixed; historical proxy controls also predict first. No target probability/logit/image/mask files are stored. Independent CPU process verifies coverage, parents, counts, Dice from pixel counts and scalar aggregates; it cannot reconstruct discarded pixel maps or independently recompute pixel ASSD.',
        'Legacy groups occur within new expanded histories. Different orders reuse the same contents. Single seed, previously exposed development and UNKNOWN patient/video links prohibit unsupported statistical/clinical claims. No new source-clean or forgetting measurement.',
        'Completion is distinct from usefulness. Apply the frozen interpretation rules to extension results and all controls/tails; no performance gates, coefficient search or automatic P2. Source anchoring and ensembling are established ideas; no novelty/SOTA claim.',
        'Only source/config/tests and deidentified aggregate evidence are public. Data, checkpoint/proxy files, paths, group identifiers and per-asset digests remain private.']
    with (out/'P1_EXPERIMENT_REPORT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
