"""Independent CPU scalar closeout; no real image or source reader import."""
import json,math,re
from pathlib import Path
import numpy as np
from .plan import science,stream,matrix,allocation,binding,bound,registration_digest,digest,SCIENCE
from .evidence import invalidate,publish
from ..p2_analysis import validate_metric
from ..m1_analysis import paired
from ..host_diagnostic_analysis import channels,distribution

PAIRS=(('C_SENS','C'),('C_PER256','C'),('C_SENS','C_PER256'),('C_PCA_GLOBAL','C'),('C_PCA_REGION','C'),('C_PCA_REGION','C_PCA_GLOBAL'),('C_PCA_REGION','C_PCA_SHUFFLED'))
SUBSETS=('remaining_dev','legacy_dev','p1_extension_dev','all_dev')


def pca_scalars(p,previous,arm,visit):
    count=1 if arm=='C_PCA_GLOBAL' else 4
    if not isinstance(p,dict) or len(p.get('banks',[]))!=count or len(p.get('basis_versions_used',[]))!=count:raise ValueError('PCA bank/version cardinality')
    previous=previous or [dict(n=0,contributing_images=0,merges=0,eigh_calls=0,version=0,ready=False,rank=0) for _ in range(count)]
    if p['basis_versions_used']!=[b['version'] if b['ready'] else None for b in previous]:raise ValueError('PCA prior snapshot binding')
    if any(v is not None and v>=visit for v in p['basis_versions_used']):raise ValueError('PCA current item in own basis')
    inputs=p['input']
    for key in ('region_token_counts','sampled_region_counts','selected_region_counts','zero_vectors','assigned_bank_counts'):
        values=inputs[key];n=count if key=='assigned_bank_counts' else 4
        if len(values)!=n or any(type(v) is not int or v<0 for v in values):raise ValueError('PCA nonnegative quota cardinality')
    regions,sampled,selected,zero=[inputs[k] for k in ('region_token_counts','sampled_region_counts','selected_region_counts','zero_vectors')]
    if sum(regions[:2])!=1024 or sum(regions[2:])!=1024 or any(s>min(r,32) or n+z!=s for r,s,n,z in zip(regions,sampled,selected,zero)):raise ValueError('PCA token/selection quota')
    if inputs['missing_foreground']!=[regions[i]==0 for i in (1,3)]:raise ValueError('PCA foreground flags')
    assigned=[sum(selected)] if count==1 else selected
    if inputs['assigned_bank_counts']!=assigned:raise ValueError('PCA matched bank quota')
    for b,old,k in zip(p['banks'],previous,assigned):
        n=old['n']+k;images=old['contributing_images']+int(k>0)
        refresh=k>0 and images>=16 and images%16==0 and n>=128
        expected=dict(n=n,contributing_images=images,merges=images,eigh_calls=old['eigh_calls']+int(refresh),version=visit if refresh else old['version'],last_visit=visit)
        if any(type(b.get(key)) is not int or b[key]!=value for key,value in expected.items()):raise ValueError('PCA cumulative counts/refresh/version')
        if type(b['rank']) is not int or not 0<=b['rank']<=8 or type(b['ready']) is not bool or b['ready']!=(b['rank']>0):raise ValueError('PCA rank/readiness')
        if not refresh and any(b[key]!=old[key] for key in ('ready','rank')):raise ValueError('PCA state changed without refresh')
        want_bytes=(32+32*32)*8+((32+32*b['rank']+b['rank'])*8 if b['ready'] else 0)
        if b['state_bytes']!=want_bytes:raise ValueError('PCA scalar storage inventory')
    if sum(b['state_bytes'] for b in p['banks'])>=1024**2 or not math.isfinite(p['subloss']) or p['subloss']<0:raise ValueError('PCA bounded finite loss/state')
    if not any(v is not None for v in p['basis_versions_used']) and p['subloss']!=0:raise ValueError('PCA inactive loss')
    return p['banks']


def validate(rows,ordered,arm,order,expected=None):
    if len(rows)!=len(ordered) or len({r['group_id'] for r in rows})!=len(rows):raise ValueError('coverage/duplicate')
    resets=0;age=0;controller_age=0;ema=best=None;previous=None
    for i,(r,e) in enumerate(zip(rows,ordered),1):
        if expected is not None:
            bound(r,expected)
            for entry in r['asset_io'].values():
                if type(entry['bytes']) is not int or entry['bytes']<=0 or not math.isfinite(entry['read_verify_decode_seconds']) or entry['read_verify_decode_seconds']<0:raise ValueError('asset IO accounting')
            if set(r['asset_io'])!={'image','mask'}:raise ValueError('current image/mask IO evidence')
        if any(r[k]!=e[k] for k in ('group_id','sample_id','domain','subset')) or r['global_visit']!=i or r['order']!=order or r['arm']!=arm:raise ValueError('ordered identity')
        if type(r['reset_before_current']) is not bool:raise ValueError('reset flag type')
        if arm=='C_SENS':
            c=r['controller'];s=c['sensitivity'];trigger=False
            if s is None:
                if c['trend'] is not None:raise ValueError('empty sensitivity advanced controller')
            else:
                if not math.isfinite(s) or not 0<=s<=1:raise ValueError('sensitivity range')
                if ema is None:ema=s;best=max(s,1e-6);controller_age=1
                else:ema=.9*ema+(1-.9)*s;best=min(best,max(ema,1e-6));controller_age+=1
                trigger=controller_age>=50 and ema>7*best
                want=dict(trigger=trigger,age=controller_age,ema=ema,best=best,resets=resets+int(trigger));trend=c['trend']
                if not isinstance(trend,dict) or set(trend)!=set(want):raise ValueError('controller trend fields')
                if any(not math.isclose(trend[k],want[k],rel_tol=1e-12,abs_tol=1e-15) if k in ('ema','best') else type(trend[k]) is not type(want[k]) or trend[k]!=want[k] for k in want):raise ValueError('controller trend replay mismatch')
                if trigger:controller_age=0;ema=best=None
            if r['reset_before_current']!=trigger:raise ValueError('sensitivity reset causality')
        elif r.get('controller') is not None:raise ValueError('unexpected controller')
        if r['reset_before_current']:resets+=1;age=0
        age+=1
        if r['reset_count']!=resets or r['total_adam_calls']!=i or r['optimizer_steps_since_reset']!=age or r['segment_age']!=age:raise ValueError('reset/Adam lifecycle')
        if arm=='C_PER256' and r['reset_before_current']!=(i>1 and (i-1)%256==0):raise ValueError('period boundary')
        if arm not in ('C_PER256','C_SENS') and resets:raise ValueError('unexpected recovery')
        if r['counts']!=dict(forwards=11 if arm=='C_SENS' else 8,backwards=1,base_adam=1,perturb=0,restore=0):raise ValueError('physical calls')
        if r['prediction_fixed_before_label'] is not True:raise ValueError('evaluator isolation')
        if [m['channel'] for m in r['metrics']]!=['OD','OC']:raise ValueError('channels')
        for m in r['metrics']:validate_metric(m)
        p=r['pca']
        if arm.startswith('C_PCA_'):
            previous=pca_scalars(p,previous,arm,i)
        elif p is not None:raise ValueError('unexpected PCA')
        json.dumps(r,allow_nan=False)


def summarize(arms):
    result={}
    for d in dict.fromkeys(r['domain'] for r in arms['C']):
        selected={a:[r for r in rs if r['domain']==d] for a,rs in arms.items()}
        result[d]=dict(arms={a:{c:dict(dice_percent=distribution([100*m['dice'] for m in channels(rs,c)]),**({} if c=='macro' else dict(assd_defined=sum(m['assd'] is not None for m in channels(rs,c)),assd_undefined=sum(m['assd'] is None for m in channels(rs,c)),**{k:sum(m[k] for m in channels(rs,c)) for k in ('pred_empty','pred_full','gt_empty','gt_full')}))) for c in ('OD','OC','macro')} for a,rs in selected.items()},paired={a+'-'+b:{c:paired(channels(selected[a],c),channels(selected[b],c),c=='macro') for c in ('OD','OC','macro')} for a,b in PAIRS})
    means={a:float(np.mean([v['arms'][a]['macro']['dice_percent']['mean'] for v in result.values()])) for a in arms}
    return dict(domains=result,task_domain_macro_dice_percent=means,comparisons_pp={a+'-'+b:means[a]-means[b] for a,b in PAIRS})


def recompute(out,reg):
    out=Path(out);invalidate(out)
    try:return complete(out,reg)
    except BaseException as error:
        invalidate(out,'INCOMPLETE',type(error).__name__+': '+str(error))
        raise


def execution_evidence(out,reg):
    receipt=json.loads((out/'receipt.json').read_text());identity=receipt['binding'];jobs=matrix()['jobs'];devices=receipt['devices']
    if set(identity)!={'run_id','code_sha','science_sha256','registration_digest'} or not re.fullmatch('[0-9a-f]{32}',identity['run_id']) or not re.fullmatch('[0-9a-f]{40}',identity['code_sha']):raise ValueError('full execution identity')
    if identity['science_sha256']!=digest(SCIENCE) or identity['registration_digest']!=registration_digest(reg):raise ValueError('execution science/registration binding')
    if not 1<=len(devices)<=3 or len({d['uuid'] for d in devices})!=len(devices) or len({d['index'] for d in devices})!=len(devices):raise ValueError('unique device slots')
    if receipt['jobs']!=jobs or receipt['schedule']!=allocation(jobs,len(devices)) or receipt['formal_budget']!=science()['formal_budget'] or receipt['smoke_budget']!=science()['per_gpu_smoke_budget']:raise ValueError('frozen matrix/schedule/budget receipt')
    if (out/'dispatch.stopped.json').exists() or list(out.glob('*.failure.json')):raise ValueError('failure flag contradicts completion')
    processes=json.loads((out/'matrix.processes.json').read_text());bound(processes,identity)
    if processes['status']!='COMPUTE_COMPLETE' or any(c!=0 for c in processes['exit_codes']):raise ValueError('worker exit state')
    if not 0<=processes['active_seconds']<=86400 or not 0<=processes['wall_seconds']<=86400:raise ValueError('completed matrix exceeded time cap')
    expected={('smoke','device'+str(i)):binding(receipt,i) for i in range(len(devices))}
    job_worker={a['job_id']:a['worker'] for a in receipt['schedule']['assignments']}
    expected.update({('formal',j['job_id']):binding(receipt,job_worker[j['job_id']],j) for j in jobs})
    entries=processes['processes']
    if len(entries)!=len(expected) or len({(e['phase'],e['key']) for e in entries})!=len(expected) or processes['exit_codes']!=[e['exit_code'] for e in entries]:raise ValueError('process coverage')
    for entry in entries:
        bound(entry,expected[entry['phase'],entry['key']])
        if entry['exit_code']!=0 or entry['status']!='EXITED' or entry['pid']!=entry['pgid'] or entry['pid']<=0:raise ValueError('owned worker completion')
    started=json.loads((out/'processes.started.json').read_text());bound(started,identity)
    if started['processes']!=[{k:e[k] for k in ('pid','pgid','binding','phase','key')} for e in entries]:raise ValueError('created process evidence')
    smoke=[]
    for i in range(len(devices)):
        p=out/('device'+str(i));proof=json.loads((p/'smoke.completion.json').read_text());bound(proof,binding(receipt,i))
        if list(p.glob('*failure.json')) or proof['status']!='MECHANICAL_SMOKE_COMPLETE' or proof['physical']!=dict(forwards=118,backwards=14,base_adam=14,perturb=0,restore=0):raise ValueError('bound device smoke budget/failure')
        if not proof['backend'] or proof['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('device backend/checkpoint IO evidence')
        smoke.append(proof)
    return receipt,job_worker,smoke


def complete(out,reg):
    import torch
    if torch.cuda.is_initialized():raise ValueError('CPU closeout must not initialize GPU')
    receipt,job_worker,smokes=execution_evidence(out,reg)
    all_rows={};physical=dict(records=0,forwards=0,backwards=0,adam=0);mechanism={};target={}
    for job in matrix()['jobs']:
        p=out/job['job_id'];done=json.loads((p/'completion.json').read_text())
        identity=binding(receipt,job_worker[job['job_id']],job);bound(done,identity)
        if done['status']!='TRAJECTORY_COMPLETE' or list(p.glob('*failure.json')):raise ValueError('incomplete/contradictory trajectory')
        rows=[json.loads(line) for line in (p/'records.jsonl').read_text().splitlines()];validate(rows,stream(reg,job['order']),job['arm'],job['order'],identity);all_rows[(job['order'],job['arm'])]=rows
        counts={k:sum(r['counts'][k] for r in rows) for k in ('forwards','backwards','base_adam','perturb','restore')}
        budget=dict(forwards=job['forwards'],backwards=job['backwards'],base_adam=job['adam'],perturb=0,restore=0)
        if done['records']!=len(rows) or len(rows)!=job['records'] or done['physical']!=counts or counts!=budget:raise ValueError('completion/JSONL/job counts')
        if not done['backend'] or done['checkpoint_io']['bytes']!=reg['checkpoint']['bytes']:raise ValueError('formal backend/checkpoint IO evidence')
        physical['records']+=len(rows)
        for k,v in [('forwards','forwards'),('backwards','backwards'),('adam','base_adam')]:physical[k]+=sum(r['counts'][v] for r in rows)
        mechanism[job['job_id']]=dict(resets=rows[-1]['reset_count'],reset_visits=[r['global_visit'] for r in rows if r['reset_before_current']],host_seconds=sum(r['host_seconds'] for r in rows),peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows),pca_ready_visits=sum(any(v is not None for v in r['pca']['basis_versions_used']) for r in rows if r['pca']),pca_last=None if not rows[-1]['pca'] else rows[-1]['pca']['banks'],subloss_mean=None if not rows[-1]['pca'] else float(np.mean([r['pca']['subloss'] for r in rows])))
    if physical!={k:science()['formal_budget'][k] for k in physical}:raise ValueError('matrix counts')
    for o in range(4):
        arms={a:all_rows[o,a] for a in science()['arms']}
        reference=arms['C']
        for arm,rows in arms.items():
            if any(any(m[k]!=n[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) for r,c in zip(rows,reference) for m,n in zip(r['metrics'],c['metrics'])):raise ValueError('cross-arm GT binding')
        target[str(o)]={s:summarize({a:[r for r in rs if s=='all_dev' or r['subset']==s] for a,rs in arms.items()}) for s in SUBSETS}
    pairs=[a+'-'+b for a,b in PAIRS];desc={p:float(np.mean([target[str(o)]['remaining_dev']['comparisons_pp'][p] for o in range(4)])) for p in pairs}
    secondary=secondary_controls(all_rows,reg)
    decision=assess(target,mechanism)
    result=dict(binding=receipt['binding'],status='R1_EXPERIMENT_COMPLETE',physical=physical,smoke_physical={k:sum(s['physical'][k] for s in smokes) for k in smokes[0]['physical']},device_models=[d['model'] for d in receipt['devices']],mixed_device_models=len({d['model'] for d in receipt['devices']})>1,target=target,descriptive_four_order_comparisons_pp=desc,mechanism=mechanism,secondary_A_C0=secondary,scientific_assessment=decision,scientific_status=decision['status'],limitations=['Exposed development data; orders share contents.','Only scalar-supported PCA properties and sensitivity-controller replay are checked; no PCA features or ASSD geometry reconstructed.','Device contention and mixed models do not support controlled speedup claims.','Descriptive assessment is not external research selection or execution permission.','No automatic next run or combination.'])
    synopsis=dict(physical=physical,descriptive_matched_comparisons=desc,assessment_including_risk_and_inactive=decision,secondary_controls={o:{a:{k:v for k,v in data.items() if k!='comparisons'} for a,data in entries.items()} for o,entries in secondary.items()},mixed_device_models=result['mixed_device_models'])
    report='# R1 results\n\nR1_EXPERIMENT_COMPLETE\n\n'+json.dumps(synopsis,indent=2)+'\n\nAll domain/channel/subset pairs and mechanism coverage are in public_aggregate.json. This descriptive assessment is not final external research selection. No automatic next experiment.\n'
    if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())+len(json.dumps(result).encode())+len(report.encode())>2*1024**3:raise ValueError('result publication exceeds private output cap')
    publish(out,result,report)
    return result


def historical_binding(path,entry,name,order):
    from .assets import verified_bytes
    if not entry or not entry.get('receipt_sha256'):raise ValueError('historical receipt binding unavailable')
    receipt=json.loads(verified_bytes(entry['receipt_path'],entry['receipt_sha256']))
    if {k:receipt.get(k) for k in entry['receipt_binding']}!=entry['receipt_binding']:raise ValueError('historical receipt identity mismatch')
    if Path(receipt['output_directory']).resolve()!=path.parent.resolve() or Path(entry['receipt_path']).parent.resolve()!=path.parent.resolve():raise ValueError('historical output/receipt directory mismatch')
    expected=f'fundus_{order}_A.jsonl' if name=='A' else 'fundus_canonical_C0.jsonl'
    if path.name!=expected:raise ValueError('historical method/order file mismatch')
    old_reg=json.loads(verified_bytes(entry['registration_path'],receipt['registration_sha256']))
    if Path(entry['registration_path']).parent.resolve()!=path.parent.resolve():raise ValueError('historical registration directory mismatch')
    done=json.loads((path.parent/'run.completion.json').read_text())
    expected_done='P2_RUN_COMPLETE' if name=='A' and order<2 else 'B3_RUN_COMPLETE' if name=='A' else 'B4_RUN_COMPLETE'
    if done['status']!=expected_done or done.get('exit_code')!=0 or (path.parent/'run.failure.json').exists():raise ValueError('historical run not complete')
    target=old_reg['tasks']['fundus']['target']
    return [r for d in science()['orders'][order if name=='A' else 0] for r in target if r['domain']==d]


def secondary_controls(all_rows,reg):
    result={}
    for o in range(4):
        result[str(o)]={}
        for name,path in reg.get('historical_scalars',{}).get(str(o),{}).items():
            p=Path(path)
            if not p.is_file():result[str(o)][name]=dict(status='UNAVAILABLE');continue
            try:
                if name not in ('A','C0'):raise ValueError('unregistered historical method')
                old=[json.loads(line) for line in p.read_text().splitlines()];lookup={r['group_id']:r for r in old};reference=all_rows[o,'C']
                ordered=historical_binding(p,reg.get('historical_bindings',{}).get(str(o),{}).get(name),name,o)
                if len(old)!=len(ordered) or len(lookup)!=len(old) or set(lookup)!={r['group_id'] for r in reference}:raise ValueError('historical identity coverage')
                for i,(v,e) in enumerate(zip(old,ordered),1):
                    if v.get('task')!='fundus' or v.get('arm')!=name or v.get('order')!=(o if name=='A' else None) or v.get('visit')!=i:raise ValueError('historical method/order/visit mismatch at position '+str(i))
                    if any(v[k]!=e[k] for k in ('group_id','sample_id','domain','subset')):raise ValueError('historical registered full-stream order mismatch at position '+str(i))
                    if name=='C0' and (v.get('prediction_origin')!='canonical_stateless' or v.get('stateless_checked') is not True or v.get('counts')!=dict(forwards=1,backwards=0,base_adam=0,perturb=0,restore=0)):raise ValueError('historical C0 stateless canonical contract')
                    if [m['channel'] for m in v['metrics']]!=['OD','OC']:raise ValueError('historical metric channel set')
                mapped=[lookup[r['group_id']] for r in reference]
                for r,v in zip(reference,mapped):
                    if any(r[k]!=v[k] for k in ('sample_id','domain','subset')):raise ValueError('historical identity')
                    for a,b in zip(r['metrics'],v['metrics']):
                        validate_metric(b)
                        if any(a[k]!=b[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')):raise ValueError('historical GT counts')
            except FileNotFoundError:
                result[str(o)][name]=dict(status='UNAVAILABLE',error='historical receipt/registration/completion file missing');continue
            except (ValueError,KeyError,TypeError) as error:
                result[str(o)][name]=dict(status='UNVERIFIED',error=str(error));continue
            result[str(o)][name]=dict(status='AVAILABLE',comparisons={arm:{subset:{d:{c:paired(channels([r for r in rows if r['domain']==d and (subset=='all_dev' or r['subset']==subset)],c),channels([r for r in mapped if r['domain']==d and (subset=='all_dev' or r['subset']==subset)],c),c=='macro') for c in ('OD','OC','macro')} for d in science()['orders'][o]} for subset in SUBSETS} for (order,arm),rows in all_rows.items() if order==o})
    return result


def assess(target,mechanism):
    cfg=science();s={a:[target[str(o)]['remaining_dev']['task_domain_macro_dice_percent'][a] for o in range(4)] for a in cfg['arms']};items={}
    for arm in cfg['arms'][1:]:
        delta=[a-b for a,b in zip(s[arm],s['C'])];domain={d:float(np.mean([target[str(o)]['remaining_dev']['domains'][d]['arms'][arm]['macro']['dice_percent']['mean']-target[str(o)]['remaining_dev']['domains'][d]['arms']['C']['macro']['dice_percent']['mean'] for o in range(4)])) for d in cfg['orders'][0]}
        job_ids=[j['job_id'] for j in matrix()['jobs'] if j['arm']==arm];active=any(mechanism[k]['resets']>0 for k in job_ids) if arm=='C_SENS' else any(mechanism[k]['pca_ready_visits']>0 for k in job_ids) if 'PCA' in arm else True
        practical=float(np.mean(delta))>=.5 and sum(v>0 for v in delta)>=3;risk=any(v < -2 for v in domain.values()) or min(s[arm])-min(s['C']) < -.5
        items[arm]=dict(mean_gain_pp=float(np.mean(delta)),positive_orders=sum(v>0 for v in delta),domain_mean_deltas=domain,worst_order_score_delta_pp=min(s[arm])-min(s['C']),active=active,practical_signal=practical,risk_warning=risk)
    mean=lambda a:float(np.mean(s[a]));routes=[]
    if items['C_SENS']['active'] and items['C_SENS']['practical_signal'] and mean('C_SENS')>mean('C_PER256'):routes.append('C_SENS')
    elif items['C_PER256']['practical_signal']:routes.append('C_PER256')
    if items['C_PCA_REGION']['active'] and items['C_PCA_REGION']['practical_signal'] and mean('C_PCA_REGION')>max(mean('C_PCA_GLOBAL'),mean('C_PCA_SHUFFLED')):routes.append('C_PCA_REGION')
    elif items['C_PCA_GLOBAL']['active'] and items['C_PCA_GLOBAL']['practical_signal']:routes.append('C_PCA_GLOBAL')
    status='SUPPORTED' if routes and not any(items[a]['risk_warning'] for a in routes) else 'MIXED' if routes or any(v['mean_gain_pp']>0 for v in items.values()) else 'NO_GAIN'
    return dict(status=status,arms=items,candidate_routes=routes,recommendation='Retain C; no sufficient matched evidence' if not routes else 'Independent candidates only; tail review and independent data remain necessary',combination_run_authorized=False,note='Frozen descriptive resource-allocation thresholds, not significance, clinical criteria, or execution gates. INACTIVE mechanisms are identified by active=false; no retuning.')
