"""R2 scalar validation, bound R1 controls and independent CPU closeout."""
import json, math, re
from pathlib import Path
import numpy as np
from ..r1 import analyze as r1
from ..r1.evidence import invalidate, publish, output_bytes
from ..p2_analysis import validate_metric
from ..host_diagnostic_analysis import channels, distribution
from ..m1_analysis import paired
from .plan import science, matrix, allocation, binding, bound, digest, SCIENCE, registration_digest, stream, R1_CODE, R1_SCIENCE
from .memory import SPECS
from .weighted_pca import RHO

SUBSETS = r1.SUBSETS
PAIRS = tuple(dict.fromkeys([(a,b) for a in SPECS for b in ('C','C_PCA_REGION')]+[('R2_DE',b) for b in ('R2_D','R2_E','R2_F','R2_DE_S')]))


def historical_metadata(assets):
    """Only metadata and path presence; never open old scalar contents here."""
    reg=assets['registration'];out=Path(assets['r1_result_directory'])
    receipt=json.loads((out/'receipt.json').read_text());pointer=json.loads((out/'current_result.json').read_text())
    expected=dict(code_sha=R1_CODE,science_sha256=R1_SCIENCE,registration_digest=registration_digest(reg))
    if any(receipt['binding'].get(k)!=v for k,v in expected.items()) or not pointer.get('valid') or pointer['status']!='R1_EXPERIMENT_COMPLETE' or pointer['binding']!=receipt['binding']:
        raise ValueError('primary historical execution binding')
    controls={}
    for o in range(4):
        for arm,index in [('C',0),('C_PCA_REGION',5)]:
            p=out/f'o{o}a{index}';done=json.loads((p/'completion.json').read_text());identity=done['binding']
            if any(identity.get(k)!=v for k,v in receipt['binding'].items()) or identity.get('arm')!=arm or identity.get('order')!=o or done['status']!='TRAJECTORY_COMPLETE' or done['backend']['seed']!=20260907:
                raise ValueError('historical arm/order/seed binding')
            job=next(j for j in receipt['jobs'] if j['arm']==arm and j['order']==o)
            slot=next(a['worker'] for a in receipt['schedule']['assignments'] if a['job_id']==job['job_id'])
            bound(done,binding(receipt,slot,job))
            if done['records']!=job['records'] or done['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('historical completion/checkpoint metadata')
            if not (p/'records.jsonl').is_file(): raise FileNotFoundError('required primary historical scalars')
            controls[o,arm]=p
    return receipt,controls


def pca_scalars(p,previous,arm,visit):
    if not isinstance(p,dict) or len(p.get('banks',[]))!=4 or len(p.get('basis_versions_used',[]))!=4: raise ValueError('four banks required')
    previous=previous or [dict(n=0,contributing_images=0,eigh_calls=0,version=0,ready=False,rank=0,W=0.,Q=0.) for _ in range(4)]
    if p['basis_versions_used']!=[b['version'] if b['ready'] else None for b in previous]: raise ValueError('prior snapshot binding')
    if any(v is not None and v>=visit for v in p['basis_versions_used']): raise ValueError('current sample in basis')
    inputs=p['input']
    for key in ('region_token_counts','sampled_region_counts','selected_region_counts','zero_vectors','assigned_bank_counts'):
        if len(inputs[key])!=4 or any(type(v) is not int or v<0 for v in inputs[key]): raise ValueError('nonnegative region quota')
    regions,sampled,selected,zero=[inputs[k] for k in ('region_token_counts','sampled_region_counts','selected_region_counts','zero_vectors')]
    if sum(regions[:2])!=1024 or sum(regions[2:])!=1024 or any(s>min(r,32) or n+z!=s for r,s,n,z in zip(regions,sampled,selected,zero)) or selected!=inputs['assigned_bank_counts']:
        raise ValueError('selection and assigned quota')
    if inputs['missing_foreground']!=[regions[i]==0 for i in (1,3)]: raise ValueError('foreground flag')
    rho=1. if arm=='R2_D' else RHO
    for b,old,k in zip(p['banks'],previous,selected):
        n=old['n']+k;images=old['contributing_images']+int(k>0)
        refresh=bool(k and images>=16 and n>=128 and images%16==0)
        want=dict(n=n,raw_n=n,images=images,contributing_images=images,merges=images,last_visit=visit,eigh_calls=old['eigh_calls']+int(refresh),version=visit if refresh else old['version'])
        if any(type(b.get(key)) is not int or b[key]!=value for key,value in want.items()): raise ValueError('raw counts/visit/refresh')
        for key,value in dict(W=old['W']*rho+k,Q=old['Q']*rho**2+k,rho=rho).items():
            if not math.isfinite(b[key]) or not math.isclose(b[key],value,rel_tol=1e-12,abs_tol=1e-12): raise ValueError('weighted scalar replay')
        if type(b['rank']) is not int or not 0<=b['rank']<=8 or type(b['ready']) is not bool or b['ready']!=(b['rank']>0): raise ValueError('rank/ready')
        if not refresh and any(b[key]!=old[key] for key in ('ready','rank')): raise ValueError('basis changed without refresh')
        expected_bytes=8448+(8*(32+32*b['rank']+b['rank']) if b['ready'] else 0)
        if b['state_bytes']!=expected_bytes: raise ValueError('PCA tensor storage')
        change=b.get('projector_change')
        if change is not None and (not refresh or not math.isfinite(change) or change<0): raise ValueError('projector change audit')
    if not math.isfinite(p['subloss']) or p['subloss']<0 or not math.isfinite(p['base_loss']): raise ValueError('finite extra loss')
    return p['banks']


def validate(rows,ordered,arm,order,identity):
    if arm not in SPECS or len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows): raise ValueError('R2 coverage/arm')
    previous=None
    for i,(r,e) in enumerate(zip(rows,ordered),1):
        bound(r,identity)
        if any(r[k]!=e[k] for k in ('group_id','sample_id','domain','subset')) or r['arm']!=arm or r['order']!=order or r['global_visit']!=i: raise ValueError('ordered identity')
        if any(r[k]!=i for k in ('segment_age','optimizer_steps_since_reset','total_adam_calls')) or r['reset_count']!=0 or r['reset_before_current'] is not False or r['controller'] is not None: raise ValueError('one Adam, no recovery')
        if r['counts']!=dict(forwards=8,backwards=1,base_adam=1,perturb=0,restore=0) or r['prediction_fixed_before_label'] is not True: raise ValueError('counts/label isolation')
        if [v['channel'] for v in r['metrics']]!=['OD','OC']: raise ValueError('channel order')
        for metric in r['metrics']:validate_metric(metric)
        if set(r['asset_io'])!={'image','mask'}:raise ValueError('asset IO fields')
        for v in r['asset_io'].values():
            if type(v['bytes']) is not int or v['bytes']<=0 or not math.isfinite(v['read_verify_decode_seconds']) or v['read_verify_decode_seconds']<0:raise ValueError('asset IO')
        previous=pca_scalars(r['pca'],previous,arm,i)
        a=r['r2'];p=r['pca']
        if a['base_loss']!=p['base_loss'] or a['extra_loss']!=p['subloss'] or not math.isclose(a['weighted_extra_loss'],.05*a['extra_loss'],rel_tol=1e-6,abs_tol=1e-9):raise ValueError('loss accounting')
        if a['shadow_readiness_only']!=(arm=='R2_F') or a['active_regions']!=len(a['region_energies']):raise ValueError('shadow/region audit')
        expected=[j for j,(n,v) in enumerate(zip(p['input']['region_token_counts'],p['basis_versions_used'])) if n and v is not None]
        if [v['bank'] for v in a['region_energies']]!=expected:raise ValueError('active region membership')
        for d in a['region_energies']:
            if d['tokens']!=p['input']['region_token_counts'][d['bank']] or any(not math.isfinite(v) or v<0 for k,v in d.items() if k not in ('bank','tokens')):raise ValueError('region scalar energies')
        if not expected and a['extra_loss']!=0:raise ValueError('inactive extra loss')
        json.dumps(r,allow_nan=False)


def execution_evidence(out,assets):
    reg=assets['registration'];receipt=json.loads((out/'receipt.json').read_text());identity=receipt['binding'];jobs=matrix()['jobs'];devices=receipt['devices']
    if set(identity)!={'run_id','code_sha','science_sha256','registration_digest'} or not re.fullmatch('[0-9a-f]{32}',identity['run_id']) or not re.fullmatch('[0-9a-f]{40}',identity['code_sha']):raise ValueError('execution identity')
    if identity['science_sha256']!=digest(SCIENCE) or identity['registration_digest']!=registration_digest(reg):raise ValueError('science/registration binding')
    packet=json.loads((out/'packet.private.json').read_text());bound(packet,identity)
    from .plan import authorize
    ids=authorize(packet['authorization'],reg,current_sha=identity['code_sha'])
    if [d['index'] for d in devices]!=ids or len({d['uuid'] for d in devices})!=len(devices):raise ValueError('authorized devices')
    if packet['assets']!=assets or packet['devices']!=devices or packet['jobs']!=jobs or packet['schedule']!=allocation(jobs,len(ids)):raise ValueError('packet consistency')
    if receipt['jobs']!=jobs or receipt['schedule']!=packet['schedule'] or receipt['formal_budget']!=science()['formal_budget'] or receipt['smoke_budget']!=science()['per_gpu_smoke_budget']:raise ValueError('matrix/schedule/budget')
    completed=[]
    if 'continuation' in packet:
        from .continuation import inspect_source,read
        continuation=inspect_source(packet['continuation']['source_directory'],assets,ids,packet['authorization'])
        if packet['continuation']!=continuation or receipt.get('continuation')!=continuation or packet['runtime_caps']!=receipt.get('runtime_caps'):raise ValueError('continuation evidence binding')
        original=read(Path(continuation['source_directory'])/'receipt.json')
        if [(d['index'],d['uuid']) for d in devices]!=[(d['index'],d['uuid']) for d in original['devices']]:raise ValueError('continuation device identity')
        completed=continuation['completed_jobs']
        if any((out/k).exists() for k in completed):raise ValueError('carried trajectory must not be rewritten')
        from .plan import caps
        maximum=caps();maximum['active_seconds']-=continuation['prior_active_seconds'];maximum['wall_seconds']-=continuation['prior_wall_seconds'];maximum['bytes']-=continuation['source_bytes']
        if set(packet['runtime_caps'])!=set(maximum) or any(not 0<v<=maximum[k] for k,v in packet['runtime_caps'].items()):raise ValueError('cumulative continuation caps')
    elif 'continuation' in receipt:raise ValueError('receipt-only continuation')
    if (out/'dispatch.stopped.json').exists() or list(out.glob('*.failure.json')):raise ValueError('failure contradicts completion')
    process=json.loads((out/'matrix.processes.json').read_text());bound(process,identity)
    if process['status']!='COMPUTE_COMPLETE' or not 0<=process['active_seconds']<=86400 or not 0<=process['wall_seconds']<=86400:raise ValueError('process completion/budget')
    if completed and any(process[k]>packet['runtime_caps'][k] for k in ('active_seconds','wall_seconds')):raise ValueError('continuation runtime cap')
    slots={v['job_id']:v['worker'] for v in receipt['schedule']['assignments']}
    expected={('smoke','device'+str(i)):binding(receipt,i) for i in range(len(ids))}
    expected.update({('formal',j['job_id']):binding(receipt,slots[j['job_id']],j) for j in jobs if j['job_id'] not in completed})
    entries=process['processes']
    if len(entries)!=len(expected) or len({(e['phase'],e['key']) for e in entries})!=len(expected) or process['exit_codes']!=[e['exit_code'] for e in entries]:raise ValueError('process coverage')
    for e in entries:
        bound(e,expected[e['phase'],e['key']])
        if e['exit_code']!=0 or e['status']!='EXITED' or e['pid']!=e['pgid'] or e['pid']<=0:raise ValueError('owned process exit')
    started=json.loads((out/'processes.started.json').read_text());bound(started,identity)
    if started['processes']!=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]:raise ValueError('created process inventory')
    smokes=[]
    for i in range(len(ids)):
        p=out/('device'+str(i));s=json.loads((p/'smoke.completion.json').read_text());bound(s,binding(receipt,i))
        if list(p.glob('*failure.json')) or s['status']!='MECHANICAL_SMOKE_COMPLETE' or s['physical']!=dict(forwards=112,backwards=14,base_adam=14,perturb=0,restore=0) or s['backend']['seed']!=20260907 or s['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('smoke evidence')
        smokes.append(s)
    return receipt,slots,smokes


def historical_rows(assets):
    reg=assets['registration'];old,controls=historical_metadata(assets)
    r1.execution_evidence(Path(assets['r1_result_directory']),reg)
    result={}
    for (o,arm),path in controls.items():
        done=json.loads((path/'completion.json').read_text());rows=[json.loads(s) for s in (path/'records.jsonl').read_text().splitlines()]
        r1.validate(rows,stream(reg,o),arm,o,done['binding'])
        counts={k:sum(r['counts'][k] for r in rows) for k in ('forwards','backwards','base_adam','perturb','restore')}
        if done['records']!=len(rows) or done['physical']!=counts or done['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('old completion/scalar counts')
        result[o,arm]=rows
    for o in range(4):
        if any(any(x[k]!=y[k] for x,y in zip(a['metrics'],b['metrics']) for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) for a,b in zip(result[o,'C'],result[o,'C_PCA_REGION'])):
            raise ValueError('historical GT metadata')
    return result


def summarize(arms):
    result={}
    for domain in science()['orders'][0]:
        rs={a:[r for r in rows if r['domain']==domain] for a,rows in arms.items()}
        values={a:{c:dict(dice_percent=distribution([100*m['dice'] for m in channels(rows,c)]),**({} if c=='macro' else dict(assd_conditional=distribution([m['assd'] for m in channels(rows,c) if m['assd'] is not None]),assd_defined=sum(m['assd'] is not None for m in channels(rows,c)),assd_undefined=sum(m['assd'] is None for m in channels(rows,c)),**{k:sum(m[k] for m in channels(rows,c)) for k in ('pred_empty','pred_full','gt_empty','gt_full')}))) for c in ('OD','OC','macro')} for a,rows in rs.items()}
        result[domain]=dict(arms=values,paired={a+'-'+b:{c:paired(channels(rs[a],c),channels(rs[b],c),c=='macro') for c in ('OD','OC','macro')} for a,b in PAIRS})
    means={a:{c:float(np.mean([v['arms'][a][c]['dice_percent']['mean'] for v in result.values()])) for c in ('OD','OC','macro')} for a in arms}
    comparisons={a+'-'+b:means[a]['macro']-means[b]['macro'] for a,b in PAIRS}
    return dict(domains=result,domain_equal_dice_percent=means,comparisons_pp=comparisons,
                pooled_content_paired={a+'-'+b:{c:paired(channels(arms[a],c),channels(arms[b],c),c=='macro') for c in ('OD','OC','macro')} for a,b in PAIRS},
                factorial_interaction_pp=comparisons['R2_DE-R2_D']-comparisons['R2_E-C_PCA_REGION'])


def assess(target):
    scores={a:[target[str(o)]['remaining_dev']['domain_equal_dice_percent'][a]['macro'] for o in range(4)] for a in ('C','C_PCA_REGION',*SPECS)}
    items={}
    for arm in SPECS:
        delta=np.array(scores[arm])-scores['C']
        domain={d:float(np.mean([target[str(o)]['remaining_dev']['domains'][d]['arms'][arm]['macro']['dice_percent']['mean']-target[str(o)]['remaining_dev']['domains'][d]['arms']['C']['macro']['dice_percent']['mean'] for o in range(4)])) for d in science()['orders'][0]}
        items[arm]=dict(mean_gain_C_pp=float(delta.mean()),positive_orders=int((delta>0).sum()),practical_signal=bool(delta.mean()>=.5 and (delta>0).sum()>=3),risk_warning=any(v< -2 for v in domain.values()) or min(scores[arm])-min(scores['C'])<-.5,domain_mean_deltas=domain)
    controls=('R2_D','R2_E','R2_F','R2_DE_S')
    needed=all(np.mean(scores['R2_DE'])>np.mean(scores[c]) for c in controls) and all(sum(a>b for a,b in zip(scores['R2_DE'],scores[c]))>=3 for c in ('R2_F','R2_DE_S'))
    return dict(arms=items,DE_matched_necessity_supported=needed,status='DESCRIPTIVE_ONLY',next_execution_authorized=False,note='Shared exposed contents; thresholds are resource references, not significance or execution gates.')


def recompute(out,assets):
    out=Path(out);invalidate(out)
    try:
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU closeout required')
        receipt,slots,smokes=execution_evidence(out,assets);reg=assets['registration'];all_rows=historical_rows(assets)
        physical=dict(new_records=0,forwards=0,backwards=0,adam=0);mechanism={}
        for job in matrix()['jobs']:
            from .continuation import source_for
            p,origin=source_for(receipt,out,job);done=json.loads((p/'completion.json').read_text());identity=binding(origin,slots[job['job_id']],job);bound(done,identity)
            if done['status']!='TRAJECTORY_COMPLETE' or list(p.glob('*failure.json')):raise ValueError('trajectory incomplete')
            rows=[json.loads(s) for s in (p/'records.jsonl').read_text().splitlines()];validate(rows,stream(reg,job['order']),job['arm'],job['order'],identity)
            counts={k:sum(r['counts'][k] for r in rows) for k in ('forwards','backwards','base_adam','perturb','restore')}
            if len(rows)!=job['records'] or done['records']!=len(rows) or done['physical']!=counts or counts!=dict(forwards=job['forwards'],backwards=job['backwards'],base_adam=job['adam'],perturb=0,restore=0):raise ValueError('completion/JSONL/job counts')
            if done['backend']['seed']!=20260907 or done['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('backend/checkpoint binding')
            reference=all_rows[job['order'],'C']
            if any(any(x[k]!=y[k] for x,y in zip(r['metrics'],c['metrics']) for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) for r,c in zip(rows,reference)):raise ValueError('old/new GT metadata')
            all_rows[job['order'],job['arm']]=rows;physical['new_records']+=len(rows)
            for key,source in [('forwards','forwards'),('backwards','backwards'),('adam','base_adam')]:physical[key]+=counts[source]
            mechanism[job['job_id']]=dict(ready_visits=sum(r['r2']['active_regions']>0 for r in rows),extra_loss=distribution([r['r2']['extra_loss'] for r in rows]),last_banks=rows[-1]['pca']['banks'],host_seconds=sum(r['host_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
        if physical!={k:science()['formal_budget'][k] for k in physical}:raise ValueError('matrix budget')
        target={str(o):{subset:summarize({a:[r for r in all_rows[o,a] if subset=='all_dev' or r['subset']==subset] for a in ('C','C_PCA_REGION',*SPECS)}) for subset in SUBSETS} for o in range(4)}
        result=dict(binding=receipt['binding'],status='R2_EXPERIMENT_COMPLETE',physical=physical,smoke_physical={k:sum(s['physical'][k] for s in smokes) for k in smokes[0]['physical']},historical_primary_records=sum(len(v) for (o,a),v in all_rows.items() if a in ('C','C_PCA_REGION')),target=target,mechanism=mechanism,assessment=assess(target),secondary_A_C0='NOT_RECOMPUTED: optional historical reference; primary C/R validated',limitations=['Exposed development data; four orders share contents.','Scalar replay checks W/Q/counts/refresh, not feature covariance or ASSD geometry.','F includes shadow PCA cost; no direction in its loss.','Same rule does not ensure identical F/DE tokens after adaptation.','No automatic next run.'])
        result['four_order_equal']={subset:dict(arms={a:{c:float(np.mean([target[str(o)][subset]['domain_equal_dice_percent'][a][c] for o in range(4)])) for c in ('OD','OC','macro')} for a in ('C','C_PCA_REGION',*SPECS)},comparisons_pp={a+'-'+b:float(np.mean([target[str(o)][subset]['comparisons_pp'][a+'-'+b] for o in range(4)])) for a,b in PAIRS},factorial_interaction_pp=float(np.mean([target[str(o)][subset]['factorial_interaction_pp'] for o in range(4)]))) for subset in SUBSETS}
        result['focus_domains']={str(o):{d:target[str(o)]['remaining_dev']['domains'][d] for d in ('REFUGE_Valid','ORIGA','Drishti_GS')} for o in range(4)}
        result['limitations'].append('Drishti_GS remaining_dev has 37 contents and one-quarter domain weight; four orders are not independent samples.')
        if 'continuation' in receipt:
            from .continuation import public_accounting
            result['continuation']=public_accounting(receipt['continuation'],physical,result['smoke_physical'])
            result['limitations'].append(result['continuation']['note'])
        report=render_report(result)
        if output_bytes(out)+receipt.get('continuation',{}).get('source_bytes',0)+len(json.dumps(result).encode())+len(report.encode())>2*1024**3:raise ValueError('output budget')
        # Preserve R1's atomic publisher unchanged; the R2 alias follows the same current pointer.
        alias=out/'R2_EXPERIMENT_REPORT.md'
        if not alias.is_symlink():alias.symlink_to('current/R1_EXPERIMENT_REPORT.md')
        publish(out,result,report);return result
    except BaseException as e:
        invalidate(out,'INCOMPLETE',type(e).__name__+': '+str(e));raise


def render_report(result):
    lines=['# R2 experiment report','','R2_EXPERIMENT_COMPLETE','','Primary endpoint: remaining_dev, OD/OC mean, four domains equally weighted, then four orders equally weighted.','','| Arm | Order 0 | Order 1 | Order 2 | Order 3 | Mean | Mean gain C (pp) |','|---|---:|---:|---:|---:|---:|---:|']
    means=result['four_order_equal']['remaining_dev']['arms']
    for arm in ('C','C_PCA_REGION',*SPECS):
        values=[result['target'][str(o)]['remaining_dev']['domain_equal_dice_percent'][arm]['macro'] for o in range(4)]
        lines.append('| '+arm+' | '+' | '.join(f'{v:.4f}' for v in [*values,means[arm]['macro'],means[arm]['macro']-means['C']['macro']])+' |')
    lines+=['','## Matched comparisons','','```json',json.dumps(result['four_order_equal']['remaining_dev']['comparisons_pp'],indent=2),'```','','Factorial interaction (descriptive pp): '+str(result['four_order_equal']['remaining_dev']['factorial_interaction_pp']),'','## Focus domains','','REFUGE_Valid OC and ORIGA / Drishti_GS gains and costs are detailed by order below. ASSD means are conditional on defined values; paired ASSD uses jointly defined contents only.','','| Order | Domain | Arm | OC Dice % | OC ASSD px | OC Dice decline vs C | Macro gain C (pp) |','|---|---|---|---:|---:|---:|---:|']
    for o,domains in result['focus_domains'].items():
        for d,entry in domains.items():
            for arm in SPECS:
                oc=entry['arms'][arm]['OC'];paired_oc=entry['paired'][arm+'-C']['OC']['dice_delta_pp']
                assd=oc['assd_conditional']['mean'];decline=paired_oc['negative']/paired_oc['n'] if paired_oc['n'] else None
                delta=entry['arms'][arm]['macro']['dice_percent']['mean']-entry['arms']['C']['macro']['dice_percent']['mean']
                lines.append(f"| {o} | {d} | {arm} | {oc['dice_percent']['mean']:.4f} | {assd} | {decline} | {delta:.4f} |")
    lines+=['','## Resource-reference assessment','','```json',json.dumps(result['assessment'],indent=2),'```','','## Counts and limits','','```json',json.dumps(dict(formal=result['physical'],smoke=result['smoke_physical'],historical_primary=result['historical_primary_records']),indent=2),'```','',*['- '+v for v in result['limitations']],'','Full per-subset, domain, channel, paired-tail and mechanism scalars are in public_aggregate.json. No automatic next experiment.','']
    if 'continuation' in result:lines+=['## IO continuation accounting','','```json',json.dumps(result['continuation'],indent=2),'```','']
    return '\n'.join(lines)
