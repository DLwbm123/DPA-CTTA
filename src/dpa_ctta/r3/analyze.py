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
from .plan import science,matrix,allocation,stream,stream_summary,registration_digest,SCIENCE_SHA

FAMILIES={a:(b,c) for a,b,c in [('T_LR','T_ISO','T_DIAG'),('U_PCA','U_RAND','U_SCALE'),('S_JOINT','S_SHARED','S_NOPCA'),('M_TRANSPORT','M_IDPOST','M_SHUFFLE'),('G_PCA','G_ISO','G_ORDER')]}
PAIRS=[(a,b) for a,controls in FAMILIES.items() for b in ('C','RP',*controls)]
SUBSETS=('remaining_dev','legacy_dev','p1_extension_dev','all_dev')
COUNT_KEYS=('network_forwards','loss_backward_calls','jacobian_vjp_calls','adam_calls','actual_parameter_replacements')


def read(path):return json.loads(Path(path).read_text())


def validate(rows,ordered,job,identity):
    if len(rows)!=job['records'] or len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('complete unique trajectory required')
    past={};slot_counts=[]
    for visit,(row,expected) in enumerate(zip(rows,ordered),1):
        bound(row,identity)
        if any(row[k]!=expected[k] for k in ('group_id','sample_id','domain','subset')) or row['arm']!=job['arm'] or row['order']!=job['order']:raise ValueError('ordered identity')
        json.dumps(row,allow_nan=False)
        a=row['r3'];c=a['counts'];arm=job['arm']
        if a['global_visit']!=visit or a['arm']!=arm or row['prediction_fixed_before_label'] is not True:raise ValueError('causal visit')
        if set(c)!=set(COUNT_KEYS) or any(type(x) is not int or x<0 for x in c.values()):raise ValueError('physical scalar types')
        if c['network_forwards']!=(9 if arm.startswith(('T_','S_')) else 8) or c['loss_backward_calls']!=1 or c['adam_calls']!=1:raise ValueError('per-visit base calls')
        if c['jacobian_vjp_calls']>(8 if arm.startswith('U_') else 0) or c['actual_parameter_replacements']>(1 if arm.startswith(('U_','S_')) else 0):raise ValueError('extra calls')
        if [v['channel'] for v in row['metrics']]!=['OD','OC']:raise ValueError('channels')
        for metric in row['metrics']:validate_metric(metric)
        if set(row['asset_io'])!={'image','mask'} or any(type(v['bytes']) is not int or v['bytes']<=0 or not math.isfinite(v['read_verify_decode_seconds']) or v['read_verify_decode_seconds']<0 for v in row['asset_io'].values()):raise ValueError('verified asset IO')
        if arm=='C':continue
        slot=0
        if arm.startswith('S_'):
            ctx=a['context'];slot=ctx['slot']
            if ctx['created']:
                if slot!=len(slot_counts) or len(slot_counts)>=3:raise ValueError('context creation/capacity')
                slot_counts.append(0)
            if slot not in range(len(slot_counts)) or ctx['active_slots']!=1:raise ValueError('one active slot')
            slot_counts[slot]+=1
            if ctx['slot_assigned_counts']!=slot_counts or ctx['slot_updates']!=slot_counts or sum(slot_counts)!=visit:raise ValueError('slot/global counters')
            if ctx['optimizer_steps']!=([visit] if arm=='S_SHARED' else slot_counts):raise ValueError('Adam state isolation')
        p=row['pca'] if arm=='RP' else dict(input=a['memory_input'],banks=a['banks'],basis_versions_used=a['basis_versions_used'])
        old=past.get(slot,[dict(n=0,contributing_images=0,version=0,eigh_calls=0,rank=0,ready=False) for _ in range(4)])
        if len(p['banks'])!=4 or p['basis_versions_used']!=[b['version'] if b['ready'] else None for b in old]:raise ValueError('old PCA snapshots')
        if arm!='RP' and a['density_versions_used']!=[b.get('density_stat_version') for b in old]:raise ValueError('old density snapshots')
        inputs=p['input'];regions=inputs['region_token_counts'];selected=inputs['selected_region_counts']
        if len(regions)!=4 or sum(regions[:2])!=1024 or sum(regions[2:])!=1024 or inputs['assigned_bank_counts']!=selected:raise ValueError('regional quotas')
        for b,prior,k,sampled,zero in zip(p['banks'],old,selected,inputs['sampled_region_counts'],inputs['zero_vectors']):
            if any(type(v) is not int or v<0 for v in (k,sampled,zero)) or k+zero!=sampled or sampled>32:raise ValueError('selected actual token counts')
            n=prior['n']+k;images=prior['contributing_images']+int(k>0);refresh=bool(k and images>=16 and n>=128 and images%16==0)
            want=dict(n=n,contributing_images=images,merges=images,last_visit=visit,version=visit if refresh else prior['version'],eigh_calls=prior['eigh_calls']+int(refresh))
            if any(b[key]!=value for key,value in want.items()):raise ValueError('sufficient-statistic scalar replay')
            if b['ready']!=(b['rank']>0) or not 0<=b['rank']<=8:raise ValueError('basis rank')
            if not refresh and (b['rank'],b['ready'])!=(prior['rank'],prior['ready']):raise ValueError('basis refresh cadence')
            if arm.startswith('M_') and b['frame_version']!=visit:raise ValueError('post-update coordinate frame')
        past[slot]=p['banks'];json.dumps(row,allow_nan=False)
    return {key:sum(r['r3']['counts'][key] for r in rows) for key in COUNT_KEYS}


def summarize(arms):
    result={}
    for domain in science()['orders'][0]:
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
    for row in rows:visit(row['r3'],'r3')
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
            if s['status']!='MECHANICAL_SMOKE_COMPLETE' or any(s['physical'][k]!=v for k,v in dict(network_forwards=316,loss_backward_calls=38,adam_calls=38).items()) or not 0<=s['physical']['jacobian_vjp_calls']<=24:raise ValueError('smoke physical counts')
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
        if physical['records']!=budget['scoring_records'] or any(physical[k]!=budget[k] for k in ('network_forwards','loss_backward_calls','adam_calls')) or physical['jacobian_vjp_calls']>budget['jacobian_vjp_upper']:raise ValueError('complete physical budget')
        arms=[a['name'] for a in science()['arms']]
        for order in range(5):
            reference=all_rows[order,'C']
            for arm in arms:
                if any(any(x[k]!=y[k] for x,y in zip(r['metrics'],c['metrics']) for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) for r,c in zip(all_rows[order,arm],reference)):raise ValueError('cross-arm GT binding')
        target={str(o):{subset:summarize({a:[r for r in all_rows[o,a] if subset=='all_dev' or r['subset']==subset] for a in arms}) for subset in SUBSETS} for o in range(5)}
        primary={s:dict(arms={a:{c:mean(target[str(o)][s]['domain_equal_dice_percent'][a][c] for o in range(4)) for c in ('OD','OC','macro')} for a in arms},comparisons_pp={a+'-'+b:mean(target[str(o)][s]['comparisons_pp'][a+'-'+b] for o in range(4)) for a,b in PAIRS}) for s in SUBSETS}
        result=dict(status='R3_EXPERIMENT_COMPLETE',binding=identity,physical=physical,smoke_physical={k:sum(s['physical'][k] for s in smoke) for k in COUNT_KEYS},target=target,
            primary_four_order_equal=primary,secondary_recurrence=target['4'],mechanism=mechanism,next_execution_authorized=False,
            limitations=['Exposed shared contents; four primary orders are not independent patient samples.',
            'Secondary stream remains separate; no five-stream average.', 'Drishti_GS primary subset has 37 contents and one-quarter domain weight.',
            'Only scalar-supported quantities replayed; covariance, Jacobian, graph geometry and ASSD geometry not reconstructed.'])
        report='# R3 experiment results\n\n'+result['status']+'\n\n'+json.dumps(dict(primary=primary['remaining_dev'],secondary=target['4']['remaining_dev']['domain_equal_dice_percent'],physical=physical),indent=2)+'\n'
        if output_bytes(out)+len(json.dumps(result).encode())+len(report.encode())>limits['bytes']:raise ValueError('publication cap')
        alias=out/'R3_EXPERIMENT_REPORT.md'
        if not alias.is_symlink():alias.symlink_to('current/R1_EXPERIMENT_REPORT.md')
        publish(out,result,report);return result
    except BaseException as exc:
        invalidate(out,'INCOMPLETE',type(exc).__name__+': '+str(exc));raise
