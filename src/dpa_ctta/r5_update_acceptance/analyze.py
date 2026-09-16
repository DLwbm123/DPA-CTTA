"""Independent scalar replay and stage gates; never opens raw assets or a model."""
import hashlib,json,math,random
from collections import deque
from statistics import mean
from pathlib import Path
from ..p2_analysis import validate_metric as inherited_validate_metric
from ..host_diagnostic_analysis import channels,distribution
from ..m1_analysis import paired
from ..r1.plan import bound,binding
from ..r1.evidence import invalidate,publish,write
from .plan import science,SCIENCE_SHA,matrix,stream,stream_summary,registration_digest,fingerprint,allocation
from .rule import PHYSICAL


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]


def validate_metric(m):
    inherited_validate_metric(m)
    pred,gt,n,tp=(m[k] for k in ('pred_pixels','gt_pixels','total_pixels','intersection'))
    cells=(tp,pred-tp,gt-tp,n-pred-gt+tp)
    if any(type(v) is not int or v<0 for v in cells):raise ValueError('R5 infeasible TP/FP/FN/TN')


def replay(traces,ordered,job,identity,p_accept=None):
    if len(traces)!=len(ordered) or len(traces)!=job['records'] or len({r['group_id'] for r in traces})!=len(traces):raise ValueError('trace coverage/duplicates')
    history=deque(maxlen=128);rng=random.Random(20260908);committed=forced=eligible_count=restores=0
    for t,(row,entry) in enumerate(zip(traces,ordered),1):
        bound(row,identity)
        if any(row[k]!=entry[k] for k in ('group_id','sample_id','domain','subset')):raise ValueError('ordered trace identity')
        z=row['trace'];json.dumps(z,allow_nan=False)
        if z['visit']!=t or z['arm']!=job['arm'] or z['counts']!=PHYSICAL or z['source_unchanged'] is not True or z['transaction_complete'] is not True:raise ValueError('visit/physical/causality')
        for name in ('e_pre','e_trial'):
            x=z[name];n=x['count'];s=x['sse']
            if type(n) is not int or n!=524288 or not math.isfinite(s) or not 0<=s<=n or x['mean']!=s/n:raise ValueError('error SSE/count')
        if [r['region'] for r in z['regions']]!=['OD_fg','OD_bg','OC_fg','OC_bg']:raise ValueError('four regions')
        valid=[]
        for r in z['regions']:
            n,s=r['count'],r['sse']
            if type(n) is not int or not 0<=n<=262144 or not math.isfinite(s) or not 0<=s<=n or r['mean']!=(s/n if n else None):raise ValueError('regional SSE/count/null')
            if n:valid.append(s/n)
        if any(sum(r['count'] for r in z['regions'][i:i+2])>262144 for i in (0,2)):raise ValueError('region overlap')
        risk=max(valid) if valid else None
        values=sorted(history);h=(len(values)-1)*.90
        if not values:q90=None
        else:
            low=int(h);q90=values[low]+(h-low)*(values[min(low+1,len(values)-1)]-values[low])
        # Independent implementation, with the specified float64 arithmetic order.
        if z['r']!=risk or z['past_count']!=len(history) or z['q90_past']!=q90:raise ValueError('past window/quantile')
        reason='warmup' if t<=32 else 'empty_reliable_regions' if risk is None else 'insufficient_history' if len(history)<32 else 'eligible'
        eligible=reason=='eligible';shadow=not eligible or (z['e_trial']['mean']<=z['e_pre']['mean']+1e-12 and risk<=q90+1e-12)
        arm=job['arm'];draw=rng.random() if arm=='C_RANDOM' else None
        if arm=='C_RANDOM' and (type(p_accept) not in (int,float) or not 0<=p_accept<=1):raise ValueError('calibration missing')
        accept=True if arm in ('C','C_HALF') else shadow if arm=='C_VERIFY' else (not eligible or draw<p_accept)
        expected=dict(eligible=eligible,forced=not eligible,reason=reason,shadow_accept=shadow,accept=accept,random_draw=draw,p_accept=p_accept if arm=='C_RANDOM' else None)
        if any(z[k]!=v for k,v in expected.items()):raise ValueError('decision replay')
        if any(type(z[k]) is not bool for k in ('eligible','forced','shadow_accept','accept')):raise ValueError('boolean decision types')
        committed+=int(accept);forced+=int(not eligible);eligible_count+=int(eligible);restores+=int(not accept)
        totals=dict(n_visits=t,n_candidate_adam=t,n_committed=committed,n_rejected=t-committed,n_forced=forced,n_eligible=eligible_count,parameter_restorations=restores)
        if z['totals']!=totals or z['adam_committed_step']!=committed:raise ValueError('transaction counters')
        if risk is not None:history.append(risk)
        if z['buffer_count_after']!=len(history):raise ValueError('append timing')
    return totals


def join(traces,evaluations,ordered,job,identity,p_accept=None):
    replay(traces,ordered,job,identity,p_accept)
    if len(evaluations)!=len(traces):raise ValueError('evaluation coverage')
    result=[]
    for t,e in zip(traces,evaluations):
        bound(e,identity)
        if any(e[k]!=t[k] for k in ('sample_id','group_id','domain','subset','visit')) or e['visit']!=t['trace']['visit'] or e['evaluation']['transaction_before_GT'] is not True:raise ValueError('evaluation join/causality')
        metrics=e['evaluation']['metrics']
        if set(metrics)!={'pre','q','trial','emit'}:raise ValueError('prediction coverage')
        for name,ms in metrics.items():
            if [v['channel'] for v in ms]!=['OD','OC']:raise ValueError('channels')
            for m in ms:validate_metric(m)
            for m,p in zip(ms,metrics['pre']):
                if any(m[k]!=p[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) or m['total_pixels']!=262144:raise ValueError('GT identity')
        if metrics['emit']!=metrics['trial' if t['trace']['accept'] else 'pre']:raise ValueError('emitted branch metrics')
        json.dumps(e,allow_nan=False)
        result.append(dict(**t,metrics=metrics,L=[100*(q['dice']-p['dice']) for p,q in zip(metrics['pre'],metrics['trial'])]))
    return result


def calibration(trace_sets,identity):
    # This function has no evaluation input. Include complete unlabeled rows in the digest.
    numerator=denominator=0;inputs={}
    for order,rows in sorted(trace_sets.items()):
        if int(order) not in (0,1,4) or len(rows)!=1951:raise ValueError('calibration complete A streams')
        numerator+=sum(int(r['trace']['eligible'] and r['trace']['shadow_accept']) for r in rows)
        denominator+=sum(int(r['trace']['eligible']) for r in rows)
        inputs[str(order)]=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if set(map(int,trace_sets))!={0,1,4}:raise ValueError('three A calibration inputs')
    return dict(binding=identity,status='FROZEN_LABEL_FREE' if denominator else 'INVALID_ZERO_ELIGIBLE',numerator=numerator,denominator=denominator,p_accept=numerator/denominator if denominator else None,input_digests=inputs,uses_labels=False,offline_development_calibration=True)


def domain_mean(rows,fn):
    domains=sorted({r['domain'] for r in rows})
    if len(domains)!=4:raise ValueError('four-domain endpoint required')
    return mean(mean(fn(r) for r in rows if r['domain']==d) for d in domains)


def diagnostic(rows):
    eligible=sum(r['trace']['eligible'] for r in rows);kept=sum(r['trace']['eligible'] and r['trace']['shadow_accept'] for r in rows)
    rate=kept/eligible if eligible else None;primary=[r for r in rows if r['subset']=='remaining_dev']
    gain=domain_mean(primary,lambda r:(1-r['trace']['shadow_accept'])*(-mean(r['L'])))
    random_gain=None if rate is None else domain_mean(primary,lambda r:r['trace']['eligible']*(1-rate)*(-mean(r['L'])))
    return dict(eligible=eligible,eligible_accepted=kept,eligible_rejected=eligible-kept,eligible_acceptance_rate=rate,G=gain,G_random=random_gain,GT_instantaneous_oracle_opportunity=domain_mean(primary,lambda r:max(0,-mean(r['L']))))


def gate_A(streams,mechanically_valid=True):
    if not mechanically_valid or set(streams)!={'0','1','4'}:raise ValueError('mechanical incompleteness is not scientific failure')
    observable=all(s['eligible']>0 and s['eligible_rejected']>0 and s['eligible_acceptance_rate']>=.5 for s in streams.values())
    checks=dict(observable=observable,primary_nonnegative=all(streams[o]['G']>=0 for o in ('0','1')),primary_mean=mean(streams[o]['G'] for o in ('0','1'))>=.20,recurrence=streams['4']['G']>=0,better_than_random=all(streams[o]['G_random'] is not None and streams[o]['G']>streams[o]['G_random'] for o in ('0','1')))
    passed=all(checks.values())
    return dict(status='R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW' if passed else 'R5A_DIAGNOSTIC_COMPLETE_NO_ADVANCE',eligible_for_review=passed,checks=checks,next_execution_authorized=False)


def summarize(rows):
    """All subsets, domains, fixed windows; content-weighted diagnostics explicitly labeled."""
    result={}
    for subset in ('remaining_dev','legacy_dev','p1_extension_dev','all_dev'):
        selected=[r for r in rows if subset=='all_dev' or r['subset']==subset]
        result[subset]={}
        selections=[('all_contents',selected)]+[('domain:'+d,[r for r in selected if r['domain']==d]) for d in sorted({r['domain'] for r in selected})]+[(f'window:{lo}-{hi}',[r for r in selected if lo<=r['visit']<=hi]) for lo,hi in science()['windows']]
        for key,rs in selections:
            block=dict(n=len(rs),eligible=sum(r['trace']['eligible'] for r in rs),forced=sum(r['trace']['forced'] for r in rs),accepted=sum(r['trace']['accept'] for r in rs),shadow_rejected=sum(not r['trace']['shadow_accept'] for r in rs),channels={})
            for i,ch in enumerate(('OD','OC','macro')):
                L=lambda r:mean(r['L']) if ch=='macro' else r['L'][i]
                block['channels'][ch]=dict(L=distribution([L(r) for r in rs]),shadow_rejected_L=distribution([L(r) for r in rs if not r['trace']['shadow_accept']]),shadow_kept_L=distribution([L(r) for r in rs if r['trace']['shadow_accept']]),actually_rejected_L=distribution([L(r) for r in rs if not r['trace']['accept']]),actually_kept_L=distribution([L(r) for r in rs if r['trace']['accept']]))
                predictions={p:channels([dict(metrics=r['metrics'][p]) for r in rs],ch) for p in ('pre','q','trial','emit')}
                block['channels'][ch]['dice_percent']={p:distribution([100*m['dice'] for m in v]) for p,v in predictions.items()}
                if ch!='macro':block['channels'][ch]['assd']={p:dict(conditional=distribution([m['assd'] for m in v if m['assd'] is not None]),defined=sum(m['assd'] is not None for m in v),undefined=sum(m['assd'] is None for m in v)) for p,v in predictions.items()}
                block['channels'][ch]['paired']={p+'-'+q:paired(predictions[p],predictions[q],ch=='macro') for p,q in (('trial','pre'),('emit','pre'),('emit','trial'))}
            block['unlabeled']={k:distribution([r['trace'][k]['mean'] for r in rs]) for k in ('e_pre','e_trial')}
            block['unlabeled']['r']=distribution([r['trace']['r'] for r in rs if r['trace']['r'] is not None]);block['unlabeled']['r_null']=sum(r['trace']['r'] is None for r in rs)
            block['unlabeled']['regions']={name:dict(count=distribution([r['trace']['regions'][i]['count'] for r in rs]),risk=distribution([r['trace']['regions'][i]['mean'] for r in rs if r['trace']['regions'][i]['mean'] is not None])) for i,name in enumerate(('OD_fg','OD_bg','OC_fg','OC_bg'))}
            result[subset][key]=block
        result[subset]['domain_equal_dice_percent']={p:{ch:mean(v['channels'][ch]['dice_percent'][p]['mean'] for k,v in result[subset].items() if k.startswith('domain:')) for ch in ('OD','OC','macro')} for p in ('pre','q','trial','emit')}
    return result


def gate_B(values,domain_deltas,recurrence_delta):
    checks={}
    for control,threshold in (('C',.5),('C_HALF',.2),('C_RANDOM',.2)):
        v=values[control]
        if len(v)!=4:raise ValueError('four primary B orders')
        checks['VERIFY-'+control]=mean(v)>=threshold and sum(x>0 for x in v)>=3
    if len(domain_deltas)!=4:raise ValueError('four domains')
    checks.update(worst_order=min(values['C'])>=-.5,domains=min(domain_deltas)>=-2.,recurrence=recurrence_delta>=-.1)
    return dict(passed=all(checks.values()),checks=checks,next_execution_authorized=False)


def completed(out,reg,scope):
    """Read and validate this finite run, including the original C path smoke."""
    if scope not in ('A','B_NEW'):raise ValueError('R5 finite stage scope')
    out=Path(out);receipt=read(out/'receipt.json');ident=receipt['binding'];jobs=matrix(scope)
    if read(out/'R5_SCOPE.json')!=dict(schema='R5_UPDATE_ACCEPTANCE_V1',scope=scope,run_id=ident['run_id']):raise ValueError('R5 output ownership')
    wanted=dict(science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'],c_fingerprint=fingerprint(),scope=scope)
    if any(ident.get(k)!=v for k,v in wanted.items()) or receipt['jobs']!=jobs:raise ValueError('run science/scope')
    if not __import__('re').fullmatch('[0-9a-f]{40}',ident['code_sha']):raise ValueError('execution SHA')
    m=read(out/'matrix.processes.json');bound(m,ident);slots=len(receipt['devices'])
    if not 1<=slots<=3 or receipt['schedule']!=allocation(jobs,slots):raise ValueError('schedule')
    if m['status']!='COMPUTE_COMPLETE' or m['exit_codes']!=[0]*(slots+len(jobs)) or m['unstarted_jobs'] or list(out.rglob('*failure.json')) or (out/'dispatch.stopped.json').exists():raise ValueError('incomplete compute')
    processes=m['processes'];started=read(out/'processes.started.json');bound(started,ident)
    if started['processes']!=[{k:p[k] for k in ('pid','pgid','binding','phase','key')} for p in processes]:raise ValueError('process ownership history')
    assigned={a['job_id']:a['worker'] for a in receipt['schedule']['assignments']}
    expected={('smoke','device'+str(i)):binding(receipt,i) for i in range(slots)}
    expected.update({('formal',j['job_id']):binding(receipt,assigned[j['job_id']],j) for j in jobs})
    if len(processes)!=len(expected) or len({(p['phase'],p['key']) for p in processes})!=len(expected):raise ValueError('process coverage')
    for p in processes:
        bound(p,expected[p['phase'],p['key']])
        if p['exit_code']!=0 or p['status']!='EXITED' or p['pid']!=p['pgid']:raise ValueError('process failure')
    caps=receipt['caps']
    if not 0<=m['wall_seconds']<=caps['wall_seconds'] or not 0<=m['active_seconds']<=caps['active_seconds']:raise ValueError('time caps')
    recipe=receipt['smoke_recipe']
    for i in range(slots):
        smoke=read(out/f'device{i}/smoke.completion.json');bound(smoke,binding(receipt,i))
        if smoke['status']!='MECHANICAL_SMOKE_COMPLETE' or smoke['C_parity_valid'] is not True or smoke['physical']!={k:recipe[k] for k in PHYSICAL} or smoke['recipe']!=recipe:raise ValueError('C parity smoke')
    rows={};traces={}
    p_accept=None if scope=='A' else receipt['calibration']['p_accept']
    for j in jobs:
        p=out/j['job_id'];done=read(p/'completion.json');identity=binding(receipt,assigned[j['job_id']],j);bound(done,identity)
        t=lines(p/'unlabeled.jsonl');e=lines(p/'evaluation.jsonl');ordered=stream(reg,j['order'])
        rs=join(t,e,ordered,j,identity,p_accept if j['arm']=='C_RANDOM' else None)
        if done['status']!='TRAJECTORY_COMPLETE' or done['records']!=1951 or done['totals']!=t[-1]['trace']['totals'] or done['physical']!={k:v*1951 for k,v in PHYSICAL.items()} or not 0<=done['seconds']<=caps['trajectory_seconds']:raise ValueError('trajectory completion')
        rows[j['order'],j['arm']]=rs;traces[j['order'],j['arm']]=t
    return receipt,rows,traces


def reuse_A(out,reg,expected_calibration):
    receipt,rows,traces=completed(out,reg,'A');current=read(Path(out)/'current_result.json')
    if not current.get('valid') or current['status']!='R5A_DIAGNOSTIC_COMPLETE_ELIGIBLE_FOR_REVIEW':raise PermissionError('qualified complete A required')
    actual=calibration({o:t for (o,a),t in traces.items()},receipt['binding'])
    if actual!=expected_calibration or read(Path(out)/'label_free_calibration.json')!=actual:raise ValueError('frozen calibration binding')
    if not gate_A({str(o):diagnostic(rs) for (o,a),rs in rows.items()})['eligible_for_review']:raise PermissionError('A gate replay failed')
    return receipt,rows


def recompute(out,reg):
    out=Path(out)
    # Never invalidate or rename another phase's historical result aliases.
    if read(out/'R5_SCOPE.json').get('schema')!='R5_UPDATE_ACCEPTANCE_V1':raise ValueError('not an owned R5 output')
    invalidate(out)
    try:
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU-only analyzer')
        raw=read(out/'receipt.json');scope=raw['binding']['scope'];receipt,rows,traces=completed(out,reg,scope)
        result=dict(binding=receipt['binding'],scope=scope,execution_started=True,next_execution_authorized=False,H_t=None,H_reason='No verified matching per-content C0 supplied; no C0 run authorized',physical={k:v*1951*len(rows) for k,v in PHYSICAL.items()},limitations=['Scalar relationship replay only; probabilities, gradients and ASSD geometry are not reconstructed.','Exposed development contents; orders are not independent patients.','G is a C-path shadow diagnostic, not a rollback trajectory or long-term bound.'])
        if scope=='A':
            diagnostics={str(o):diagnostic(rs) for (o,a),rs in rows.items()};result.update(gate_A(diagnostics));result['diagnostics']=diagnostics
            result['primary_two_order_G']=mean(diagnostics[o]['G'] for o in ('0','1'))
            cal=calibration({o:t for (o,a),t in traces.items()},receipt['binding']);result['calibration']=cal
            path=out/'label_free_calibration.json'
            if path.exists():
                if read(path)!=cal:raise ValueError('refuse changed label-free calibration')
            else:write(path,cal)
        else:
            old,oldrows=reuse_A(receipt['reuse_A'],reg,receipt['calibration']);rows.update(oldrows)
            if set(rows)!={(j['order'],j['arm']) for j in matrix('AB')}:raise ValueError('20-job coverage')
            result.update(status='R5B_EXPERIMENT_COMPLETE',reuse_A_binding=old['binding'],physical_total={k:v*1951*20 for k,v in PHYSICAL.items()},calibration=receipt['calibration'])
        ground_truth={}
        for rs in rows.values():
            for r in rs:
                key=r['group_id'];value=[(m['gt_pixels'],m['total_pixels'],m['gt_empty'],m['gt_full']) for m in r['metrics']['pre']]
                if ground_truth.setdefault(key,value)!=value:raise ValueError('cross-trajectory GT binding')
        result['trajectories']={f'{o}:{a}':summarize(rs) for (o,a),rs in rows.items()}
        if scope=='B_NEW':
            result['primary_four_order_equal']={s:{arm:{ch:mean(result['trajectories'][f'{o}:{arm}'][s]['domain_equal_dice_percent']['emit'][ch] for o in range(4)) for ch in ('OD','OC','macro')} for arm in ('C','C_HALF','C_RANDOM','C_VERIFY')} for s in ('remaining_dev','legacy_dev','p1_extension_dev','all_dev')}
            result['secondary_recurrence']={arm:result['trajectories'][f'4:{arm}'] for arm in ('C','C_HALF','C_RANDOM','C_VERIFY')}
            value=lambda o,a:result['trajectories'][f'{o}:{a}']['remaining_dev']['domain_equal_dice_percent']['emit']['macro']
            differences={c:[value(o,'C_VERIFY')-value(o,c) for o in range(4)] for c in ('C','C_HALF','C_RANDOM')}
            domains=sorted({r['domain'] for r in rows[0,'C']});ds=[]
            for d in domains:ds.append(mean(result['trajectories'][f'{o}:C_VERIFY']['remaining_dev']['domain:'+d]['channels']['macro']['dice_percent']['emit']['mean']-result['trajectories'][f'{o}:C']['remaining_dev']['domain:'+d]['channels']['macro']['dice_percent']['emit']['mean'] for o in range(4)))
            result['selection']=gate_B(differences,ds,value(4,'C_VERIFY')-value(4,'C'));result['primary_comparisons_pp']=differences
            result['matched_controls']={str(o):{c:{ch:paired(channels([dict(metrics=r['metrics']['emit']) for r in rows[o,'C_VERIFY'] if r['subset']=='remaining_dev'],ch),channels([dict(metrics=r['metrics']['emit']) for r in rows[o,c] if r['subset']=='remaining_dev'],ch),ch=='macro') for ch in ('OD','OC','macro')} for c in ('C','C_HALF','C_RANDOM')} for o in range(5)}
        publish(out,result,'# R5 scalar results\n\n'+result['status']+'\n\n'+json.dumps({k:v for k,v in result.items() if k not in ('trajectories','matched_controls')},indent=2)+'\n')
        return result
    except BaseException as exc:
        invalidate(out,'INCOMPLETE',type(exc).__name__+': '+str(exc));raise
