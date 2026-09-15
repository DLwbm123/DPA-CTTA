"""Independent scalar validation and separate primary/recurrence aggregates.

No image, mask, checkpoint, covariance or Jacobian reconstruction is performed.
"""
import json,math,re
from pathlib import Path
from statistics import mean
from ..r1.plan import binding,bound
from ..r1.evidence import invalidate,publish,output_bytes
from ..p2_analysis import validate_metric
from ..host_diagnostic_analysis import channels,distribution
from ..m1_analysis import paired
from ..r1.plan import science as primary_science
from .plan import science,matrix,allocation,stream,stream_summary,registration_digest,SCIENCE_SHA

ARMS=('C','RP','MT','MT_RP','FT','FT_RP','KDG','K_ALL','K_MAG','K_FREE','G_BOUND','G_CONST','G_SHUFFLE','G_GLOBAL')
PAIRS=list(dict.fromkeys([(a,'C') for a in ARMS if a!='C']+
    [('MT_RP','MT'),('FT_RP','FT'),('MT_RP','RP'),('FT_RP','RP'),('MT','FT'),('MT_RP','FT_RP')]+
    [('KDG',b) for b in ('K_ALL','K_MAG','K_FREE')]+[('G_BOUND',b) for b in ('G_CONST','G_SHUFFLE','G_GLOBAL')]))
SUBSETS=('remaining_dev','legacy_dev','p1_extension_dev','all_dev')
COUNT_KEYS=('network_forwards','loss_backward_calls','jacobian_vjp_calls','adam_calls','actual_parameter_replacements')


def read(path):return json.loads(Path(path).read_text())


def validate_auxiliary(value,metrics,arm,pca):
    def quality(rows):
        if [r['channel'] for r in rows]!=['OD','OC']:raise ValueError('teacher channels')
        for r,main in zip(rows,metrics):
            n=r['total_pixels'];fg=r['gt_foreground'];bg=r['gt_background'];pred=r['pred_foreground'];k=r['intersection']
            if any(type(v) is not int for v in (n,fg,bg,pred,k)) or n<=0 or fg+bg!=n or not 0<=k<=min(fg,pred)<=max(fg,pred)<=n:raise ValueError('teacher counts')
            if n!=main['total_pixels'] or fg!=main['gt_pixels']:raise ValueError('teacher GT binding')
            if r['dice']!=(2*k/(fg+pred) if fg+pred else 1.):raise ValueError('teacher Dice replay')
            for key,count in [('foreground',fg),('background',bg)]:
                total=r[key+'_brier_sum'];average=r[key+'_brier']
                if not 0<=total<=count or (average is None)!=(count==0) or (count and abs(average-total/count)>1e-12):raise ValueError('conditional Brier')
            total=r['foreground_brier_sum']+r['background_brier_sum']
            if abs(r['brier_sum']-total)>1e-9 or abs(r['brier']-total/n)>1e-12:raise ValueError('Brier sum')
    quality(value['q'])
    memory=value['memory']
    if pca is None:
        if memory!={'status':'NOT_APPLICABLE'}:raise ValueError('fabricated memory')
    else:
        if memory['status']!='ACTUAL_COMMITTED_STUDENT_PRE_TOKENS' or len(memory['regions'])!=4:raise ValueError('token provenance')
        for i,r in enumerate(memory['regions']):
            if r['region']!=i or r['selected_tokens']!=pca['input']['selected_region_counts'][i] or r['correct_tokens']+r['incorrect_tokens']!=r['selected_tokens'] or min(r['correct_tokens'],r['incorrect_tokens'])<0:raise ValueError('actual token quality counts')
    if arm.startswith('G_'):
        quality(value['qstar']);g=value['graph'];rows=g['transitions']
        if len(rows)!=6 or {(r['channel'],r['scope']) for r in rows}!={(c,s) for c in ('OD','OC') for s in ('all','allowed','hard_flip')}:raise ValueError('transition coverage')
        if g['outside_allowed_max_change']!=0 or g['reliable_max_change']!=0:raise ValueError('fixed support')
        for c,channel in enumerate(('OD','OC')):
            n=metrics[c]['total_pixels'];r={v['scope']:v for v in rows if v['channel']==channel}
            if r['all']['denominator']!=n or r['allowed']['denominator']!=g['allowed_counts'][c]:raise ValueError('transition denominators')
            for x in r.values():
                parts=[x[k] for k in ('wrong_to_correct','correct_to_wrong','correct_unchanged','wrong_unchanged')]
                if any(type(v) is not int or v<0 for v in parts) or sum(parts)!=x['denominator'] or not 0<=x['denominator']<=n:raise ValueError('transition partition')
            for k in ('wrong_to_correct','correct_to_wrong'):
                if not r['all'][k]==r['allowed'][k]==r['hard_flip'][k]:raise ValueError('flip support')
            if r['hard_flip']['correct_unchanged'] or r['hard_flip']['wrong_unchanged']:raise ValueError('hard flip contains unchanged')
            old=value['q'][c];new=value['qstar'][c]
            correct=lambda x:n-x['gt_foreground']-x['pred_foreground']+2*x['intersection']
            if correct(new)-correct(old)!=r['all']['wrong_to_correct']-r['all']['correct_to_wrong']:raise ValueError('transition correctness replay')
            if not 0<=g['reliable_correct'][c]<=g['reliable_counts'][c]<=n or g['allowed_counts'][c]+g['reliable_counts'][c]>n or not 0<=g['changed_counts'][c]<=g['allowed_counts'][c]:raise ValueError('anchor and support counts')
    elif 'qstar' in value or 'graph' in value:raise ValueError('unexpected graph evaluation')


def validate(rows,ordered,job,identity):
    if len(rows)!=job['records'] or len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('complete unique trajectory required')
    past=[dict(n=0,contributing_images=0,version=0,eigh_calls=0,rank=0,ready=False) for _ in range(4)]
    for visit,(row,expected) in enumerate(zip(rows,ordered),1):
        bound(row,identity);json.dumps(row,allow_nan=False)
        if any(row[k]!=expected[k] for k in ('group_id','sample_id','domain','subset')) or row['arm']!=job['arm'] or row['order']!=job['order']:raise ValueError('ordered identity')
        a=row['r4t'];arm=job['arm'];c=a['counts']
        want=dict(network_forwards=9 if arm in ('MT_RP','FT_RP') else 8,loss_backward_calls=1,adam_calls=1,jacobian_vjp_calls=0,actual_parameter_replacements=0)
        if c!=want or any(type(v) is not int for v in c.values()):raise ValueError('physical counters')
        if a['global_visit']!=visit or a['arm']!=arm or row['prediction_fixed_before_label'] is not True or row['auxiliary_after_state_commit'] is not True or a['source_unchanged'] is not True:raise ValueError('causal commit')
        if a['teacher_update']!=arm.startswith('MT'):raise ValueError('EMA policy')
        if [v['channel'] for v in row['metrics']]!=['OD','OC']:raise ValueError('metric channels')
        for metric in row['metrics']:validate_metric(metric)
        if set(row['asset_io'])!={'image','mask'} or any(type(v['bytes']) is not int or v['bytes']<=0 or not math.isfinite(v['read_verify_decode_seconds']) or v['read_verify_decode_seconds']<0 for v in row['asset_io'].values()):raise ValueError('verified asset IO')
        p=a['pca'];validate_auxiliary(row['auxiliary'],row['metrics'],arm,p)
        if (p is not None)!=(arm in ('RP','MT_RP','FT_RP')):raise ValueError('RP scope')
        if p is not None:
            if len(p['banks'])!=4 or p['basis_versions_used']!=[b['version'] if b['ready'] else None for b in past]:raise ValueError('past PCA snapshots')
            inputs=p['input'];regions=inputs['region_token_counts'];selected=inputs['selected_region_counts']
            if len(regions)!=4 or sum(regions[:2])!=1024 or sum(regions[2:])!=1024 or inputs['assigned_bank_counts']!=selected:raise ValueError('regional quotas')
            for b,prior,k,sampled,zero in zip(p['banks'],past,selected,inputs['sampled_region_counts'],inputs['zero_vectors']):
                if any(type(v) is not int or v<0 for v in (k,sampled,zero)) or k+zero!=sampled or sampled>32:raise ValueError('token counts')
                n=prior['n']+k;images=prior['contributing_images']+int(k>0);refresh=bool(k and images>=16 and n>=128 and images%16==0)
                want=dict(n=n,contributing_images=images,merges=images,last_visit=visit,version=visit if refresh else prior['version'],eigh_calls=prior['eigh_calls']+int(refresh))
                if any(b[key]!=value for key,value in want.items()):raise ValueError('PCA scalar replay')
                if b['ready']!=(b['rank']>0) or not 0<=b['rank']<=8:raise ValueError('PCA rank')
                if not refresh and (b['rank'],b['ready'])!=(prior['rank'],prior['ready']):raise ValueError('PCA refresh cadence')
            past=p['banks']
        if arm.startswith('G_'):
            g=a['graph']
            if g['steps']!=64 or g['grid']!=128 or g['potential_undirected_edges']!=32512 or not all(g[k] is True for k in ('exact_outside_allowed','exact_reliable_full','exact_fixed_nodes')):raise ValueError('graph algorithm contract')
        if arm.startswith('K'):
            kernels=a['kernels']
            if set(kernels)!={'res.conv1','res.layer1.2.conv2'} or a['trainable_scalars']!=dict(KDG=21283,K_ALL=21283,K_MAG=19264,K_FREE=23424)[arm]:raise ValueError('kernel coordinates')
    return {key:sum(r['r4t']['counts'][key] for r in rows) for key in COUNT_KEYS}

def summarize(arms):
    result={}
    for domain in primary_science()['orders'][0]:
        rs={a:[r for r in rows if r['domain']==domain] for a,rows in arms.items()}
        values={a:{c:dict(dice_percent=distribution([100*m['dice'] for m in channels(rows,c)]),**({} if c=='macro' else dict(assd_conditional=distribution([m['assd'] for m in channels(rows,c) if m['assd'] is not None]),assd_defined=sum(m['assd'] is not None for m in channels(rows,c)),assd_undefined=sum(m['assd'] is None for m in channels(rows,c))))) for c in ('OD','OC','macro')} for a,rows in rs.items()}
        result[domain]=dict(arms=values,paired={a+'-'+b:{c:paired(channels(rs[a],c),channels(rs[b],c),c=='macro') for c in ('OD','OC','macro')} for a,b in PAIRS})
    means={a:{c:mean(v['arms'][a][c]['dice_percent']['mean'] for v in result.values()) for c in ('OD','OC','macro')} for a in arms}
    return dict(domains=result,domain_equal_dice_percent=means,comparisons_pp={a+'-'+b:means[a]['macro']-means[b]['macro'] for a,b in PAIRS},
        pooled_content_paired={a+'-'+b:{c:paired(channels(arms[a],c),channels(arms[b],c),c=='macro') for c in ('OD','OC','macro')} for a,b in PAIRS})


def scalar_mechanisms(rows):
    collected={}
    def visit(v,path):
        if isinstance(v,dict):
            for k,x in v.items():visit(x,path+'.'+k)
        elif isinstance(v,list):
            for i,x in enumerate(v):visit(x,path+'.'+str(i))
        elif isinstance(v,(int,float)):
            if not math.isfinite(v):raise ValueError('nonfinite trace')
            collected.setdefault(path,[]).append(float(v))
    for row in rows:
        visit(row['r4t'],'r4t');visit(row['auxiliary'],'auxiliary')
    return {k:distribution(v) for k,v in collected.items()}


def recompute(out,reg):
    out=Path(out);invalidate(out)
    try:
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU-only closeout')
        receipt=read(out/'receipt.json');identity=receipt['binding'];jobs=matrix()['jobs'];devices=receipt['devices'];slots=len(devices)
        expected=dict(science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'])
        if any(identity.get(k)!=v for k,v in expected.items()) or not re.fullmatch('[0-9a-f]{40}',identity['code_sha']) or not re.fullmatch('[0-9a-f]{32}',identity['run_id']):raise ValueError('execution bindings')
        if not 1<=slots<=3 or len({d['uuid'] for d in devices})!=slots or receipt['jobs']!=jobs or receipt['schedule']!=allocation(jobs,slots):raise ValueError('complete registered matrix')
        if (out/'dispatch.stopped.json').exists() or list(out.rglob('*failure.json')):raise ValueError('failed run cannot be complete')
        processes=read(out/'matrix.processes.json');bound(processes,identity)
        from .execution import caps
        limits=caps()
        if processes['status']!='COMPUTE_COMPLETE' or any(c!=0 for c in processes['exit_codes']) or processes.get('unstarted_jobs'):raise ValueError('process completion')
        if not 0<=processes['wall_seconds']<=limits['wall_seconds'] or not 0<=processes['active_seconds']<=limits['active_seconds']:raise ValueError('run time caps')
        assigned={a['job_id']:a['worker'] for a in receipt['schedule']['assignments']}
        expected_process={('smoke','device'+str(i)):binding(receipt,i) for i in range(slots)}
        expected_process.update({('formal',j['job_id']):binding(receipt,assigned[j['job_id']],j) for j in jobs})
        entries=processes['processes']
        if len(entries)!=len(expected_process) or len({(e['phase'],e['key']) for e in entries})!=len(expected_process) or processes['exit_codes']!=[e['exit_code'] for e in entries]:raise ValueError('process coverage')
        for e in entries:
            bound(e,expected_process[e['phase'],e['key']])
            if e['status']!='EXITED' or e['exit_code']!=0 or e['pid']!=e['pgid'] or e['pid']<=0:raise ValueError('owned process exit')
        started=read(out/'processes.started.json');bound(started,identity)
        if started['processes']!=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]:raise ValueError('started process evidence')
        smoke=[]
        for i in range(slots):
            s=read(out/('device'+str(i))/'smoke.completion.json');bound(s,binding(receipt,i))
            if s['status']!='MECHANICAL_SMOKE_COMPLETE' or any(s['physical'][k]!=v for k,v in dict(network_forwards=260,loss_backward_calls=32,adam_calls=32).items()) or not s['physical']['jacobian_vjp_calls']==0:raise ValueError('smoke physical counts')
            if s['backend']['seed']!=20260907 or s['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('smoke asset/runtime evidence')
            smoke.append(s)
        all_rows={};mechanism={};physical={k:0 for k in COUNT_KEYS};physical['records']=0
        for j in jobs:
            folder=out/j['job_id'];done=read(folder/'completion.json');ident=binding(receipt,assigned[j['job_id']],j);bound(done,ident)
            rows=[json.loads(s) for s in (folder/'records.jsonl').read_text().splitlines()]
            counts=validate(rows,stream(reg,j['order']),j,ident)
            if done['status']!='TRAJECTORY_COMPLETE' or done['records']!=len(rows) or done['physical']!=counts:raise ValueError('completion record mismatch')
            if done['backend']['seed']!=20260907 or done['checkpoint_io']['bytes']!=reg['checkpoint']['bytes'] or not 0<=done['seconds']<=limits['trajectory_seconds']:raise ValueError('runtime evidence')
            all_rows[j['order'],j['arm']]=rows;mechanism[j['job_id']]=scalar_mechanisms(rows);physical['records']+=len(rows)
            for key in COUNT_KEYS:physical[key]+=counts[key]
        budget=science()['formal_budget']
        if physical['records']!=budget['scoring_records'] or any(physical[k]!=budget[k] for k in ('network_forwards','loss_backward_calls','adam_calls')) or physical['jacobian_vjp_calls']>budget['jacobian_vjp_calls']:raise ValueError('complete physical budget')
        arms=[a['name'] for a in science()['arms']]
        for order in range(5):
            reference=all_rows[order,'C']
            for arm in arms:
                if any(any(x[k]!=y[k] for x,y in zip(r['metrics'],c['metrics']) for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) for r,c in zip(all_rows[order,arm],reference)):raise ValueError('cross-arm GT binding')
        target={str(o):{subset:summarize({a:[r for r in all_rows[o,a] if subset=='all_dev' or r['subset']==subset] for a in arms}) for subset in SUBSETS} for o in range(5)}
        primary={s:dict(arms={a:{c:mean(target[str(o)][s]['domain_equal_dice_percent'][a][c] for o in range(4)) for c in ('OD','OC','macro')} for a in arms},comparisons_pp={a+'-'+b:mean(target[str(o)][s]['comparisons_pp'][a+'-'+b] for o in range(4)) for a,b in PAIRS}) for s in SUBSETS}
        result=dict(status='R4T_EXPERIMENT_COMPLETE',binding=identity,physical=physical,smoke_physical={k:sum(s['physical'][k] for s in smoke) for k in COUNT_KEYS},target=target,
            primary_four_order_equal=primary,secondary_recurrence=target['4'],mechanism=mechanism,next_execution_authorized=False,
            limitations=['Exposed shared contents; four primary orders are not independent patient samples.',
            'Secondary stream remains separate; no five-stream average.', 'Drishti_GS primary subset has 37 contents and one-quarter domain weight.',
            'Only scalar-supported quantities replayed; covariance, Jacobian, graph geometry and ASSD geometry not reconstructed.'])
        report='# R4T experiment results\n\n'+result['status']+'\n\n'+json.dumps(dict(primary=primary['remaining_dev'],secondary=target['4']['remaining_dev']['domain_equal_dice_percent'],physical=physical),indent=2)+'\n'
        if output_bytes(out)+len(json.dumps(result).encode())+len(report.encode())>limits['bytes']:raise ValueError('publication cap')
        alias=out/'R4T_EXPERIMENT_REPORT.md'
        if not alias.is_symlink():alias.symlink_to('current/R1_EXPERIMENT_REPORT.md')
        publish(out,result,report);return result
    except BaseException as exc:
        invalidate(out,'INCOMPLETE',type(exc).__name__+': '+str(exc));raise
