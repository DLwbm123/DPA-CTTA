"""R6-owned scalar replay. Never invokes historical R5 result mutation paths."""
import hashlib,json,math,re,struct
from pathlib import Path
from statistics import mean
from ..r5_update_acceptance.analyze import validate_metric
from ..host_diagnostic_analysis import channels,distribution
from ..m1_analysis import paired
from ..r1.plan import bound,binding
from ..r1.evidence import invalidate,publish,output_bytes
from .plan import SCIENCE_SHA,SMOKE,matrix,stream,stream_summary,registration_digest,fingerprint,allocation
from .loss import ARMS,PHYSICAL

CONTROLS=('C','R_SCALE','R_SHUFFLE')
SUBSETS=('remaining_dev','legacy_dev','p1_extension_dev','all_dev')
def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(s) for s in Path(path).read_text().splitlines()]
def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
def near(a,b,rtol=1e-10,atol=1e-14):
    if not math.isfinite(a) or not math.isfinite(b) or not math.isclose(a,b,rel_tol=rtol,abs_tol=atol):raise ValueError('scalar identity mismatch')


def audit_channel(r,visit,c,arm):
    """Independent scalar algebra; cannot reconstruct spatial weights or ASSD."""
    n=r['n_fg'];m=r['n_bg'];N=262144
    if type(n) is not int or type(m) is not int or min(n,m)<0 or n+m!=N or r['channel']!=('OD','OC')[c]:raise ValueError('partition/channel')
    rho=min(8.,max(.125,m/n)) if n and m else 1.
    bg=N/(rho*n+m) if n and m else 1.;fg=rho*bg
    wanted=dict(rho_clipped=(rho!=m/n) if n and m else False,fallback=None if n and m else 'ONE_PARTITION_EMPTY')
    if any(r[k]!=v for k,v in wanted.items()):raise ValueError('cap/fallback')
    for k,v in dict(rho=rho,w_fg=fg,w_bg=bg,applied_w_fg=f32(fg),applied_w_bg=f32(bg),mean_weight=(n*f32(fg)+m*f32(bg))/N).items():near(r[k],v)
    if not .125<=fg<=8 or not .125<=bg<=8:raise ValueError('weight bound')
    energy=[r[k] for k in ('S0','Sw','Sperm','residual_fg_sse','residual_bg_sse','weighted_fg_sse','weighted_bg_sse')]
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in energy):raise ValueError('energy nonnegative finite')
    s0,sw,sp=energy[:3]
    near(s0,r['residual_fg_sse']+r['residual_bg_sse']);near(sw,r['weighted_fg_sse']+r['weighted_bg_sse'])
    near(r['weighted_fg_sse'],r['residual_fg_sse']*f32(fg)**2);near(r['weighted_bg_sse'],r['residual_bg_sse']*f32(bg)**2)
    if s0>N or r['residual_fg_sse']>n or r['residual_bg_sse']>m:raise ValueError('residual probability bound')
    if s0==0:
        if sw!=0 or sp!=0:raise ValueError('zero-energy branch')
        a=b=1.
    else:
        if min(sw,sp)<=0:raise ValueError('positive weighted energy')
        a=math.sqrt(sw/s0);b=math.sqrt(sw/sp)
        if not min(f32(fg),f32(bg))**2*s0-1e-12<=sp<=max(f32(fg),f32(bg))**2*s0+1e-12:raise ValueError('permuted energy bound')
    near(r['a'],a);near(r['b'],b)
    for k in ('bce_sum','bce_fg_sum','bce_bg_sum','weighted_bce_sum','shuffled_bce_sum'):
        if type(r[k]) not in (int,float) or not math.isfinite(r[k]) or r[k]<0:raise ValueError('BCE sum')
    near(r['bce_sum'],r['bce_fg_sum']+r['bce_bg_sum'])
    near(r['weighted_bce_sum'],f32(fg)*r['bce_fg_sum']+f32(bg)*r['bce_bg_sum'])
    if (not n and (r['bce_fg_sum'] or r['residual_fg_sse'])) or (not m and (r['bce_bg_sum'] or r['residual_bg_sse'])):raise ValueError('empty partition sums')
    near(r['residual_weighted_dot'],f32(fg)*r['residual_fg_sse']+f32(bg)*r['residual_bg_sse'])
    near(r['weight_squared_sum'],n*f32(fg)**2+m*f32(bg)**2)
    if r['weight_residual_dot']**2>r['weight_squared_sum']*s0+1e-8 or r['residual_shuffled_dot']**2>s0*sp+1e-8:raise ValueError('Cauchy-Schwarz')
    seed=int.from_bytes(hashlib.sha256(f'R6_WEIGHT_PERM_V1|20260907|{visit}|{c}'.encode('ascii')).digest()[:8],'big')&((1<<63)-1)
    if r['seed']!=seed or type(r['seed']) is not int or r['permutation_histogram_preserved'] is not True:raise ValueError('permutation seed/histogram')
    changed=r['permutation_changed_positions']
    if type(changed) is not int or not 0<=changed<=2*min(n,m) or changed%2:raise ValueError('permutation position count')
    for k in ('base_weight_sha256','shuffled_weight_sha256'):
        if not re.fullmatch('[0-9a-f]{64}',r[k]):raise ValueError('runtime weight checksum')
    if fg==bg and (changed or r['base_weight_sha256']!=r['shuffled_weight_sha256']):raise ValueError('uniform permutation')
    expected=math.sqrt(s0 if arm=='C' else sw if arm=='R_BAL' else s0*f32(a)**2 if arm=='R_SCALE' else sp*f32(b)**2)/(2*N)
    near(r['expected_logit_gradient_l2'],expected,1e-5,1e-12)
    near(r['actual_logit_gradient_l2'],expected,1e-5,1e-12)


def replay(traces,ordered,job,identity):
    if len(traces)!=len(ordered) or len(traces)!=job['records'] or len({r['group_id'] for r in traces})!=len(traces) or len({r['sample_id'] for r in traces})!=len(traces):raise ValueError('trace coverage/duplicates')
    for t,(row,entry) in enumerate(zip(traces,ordered),1):
        bound(row,identity)
        if row['visit']!=t or any(row[k]!=entry[k] for k in ('group_id','sample_id','domain','subset')):raise ValueError('ordered identity')
        z=row['trace'];json.dumps(z,allow_nan=False)
        if z['visit']!=t or z['arm']!=job['arm'] or z['counts']!=PHYSICAL or z['cumulative']!={k:v*t for k,v in PHYSICAL.items()} or z['adam_step']!=t or z['source_unchanged'] is not True or z['transaction_complete'] is not True:raise ValueError('physical/causality/Adam')
        if len(z['channels'])!=2:raise ValueError('two channels')
        for c,r in enumerate(z['channels']):audit_channel(r,t,c,job['arm'])
        key={'C':'bce_sum','R_BAL':'weighted_bce_sum','R_SCALE':'bce_sum','R_SHUFFLE':'shuffled_bce_sum'}[job['arm']]
        expected=sum(r[key]*(f32(r['a']) if job['arm']=='R_SCALE' else f32(r['b']) if job['arm']=='R_SHUFFLE' else 1) for r in z['channels'])/524288
        near(z['actual_loss'],expected,1e-5,1e-12)
        for key in ('bn_gradient_l2','adam_affine_displacement_l2','host_seconds'):
            if type(z[key]) not in (int,float) or not math.isfinite(z[key]) or z[key]<0:raise ValueError('invalid measured cost/norm')


def join(traces,evaluations,ordered,job,identity):
    replay(traces,ordered,job,identity)
    if len(evaluations)!=len(traces):raise ValueError('evaluation coverage')
    result=[]
    for t,e in zip(traces,evaluations):
        bound(e,identity);json.dumps(e,allow_nan=False)
        if any(e[k]!=t[k] for k in ('sample_id','group_id','domain','subset','visit')) or e['evaluation']['transaction_before_GT'] is not True:raise ValueError('evaluation identity/causality')
        metrics=e['evaluation']['metrics']
        if set(metrics)!={'pre','q','post'}:raise ValueError('prediction coverage')
        for ms in metrics.values():
            if [v['channel'] for v in ms]!=['OD','OC']:raise ValueError('channels')
            for m,p in zip(ms,metrics['pre']):
                validate_metric(m)
                if any(m[k]!=p[k] for k in ('gt_pixels','total_pixels','gt_empty','gt_full')) or m['total_pixels']!=262144:raise ValueError('GT identity')
        if any(m['pred_pixels']!=c['n_fg'] for m,c in zip(metrics['q'],t['trace']['channels'])):raise ValueError('actual q partition versus evaluator foreground')
        for key in ('pipeline_seconds','evaluator_seconds'):
            if type(e['evaluation'][key]) not in (int,float) or e['evaluation'][key]<0:raise ValueError('evaluation time')
        result.append(dict(**t,metrics=metrics,evaluator_seconds=e['evaluation']['evaluator_seconds'],pipeline_seconds=e['evaluation']['pipeline_seconds']))
    return result


def gate(values,domain_deltas,recurrence,stage):
    n=2 if stage=='A' else 4 if stage=='B_NEW' else 0
    if not n or set(values)!=set(CONTROLS) or set(recurrence)!=set(CONTROLS) or any(len(v)!=n for v in values.values()) or len(domain_deltas)!=4:raise ValueError('complete gate inputs')
    nums=[x for v in values.values() for x in v]+domain_deltas+list(recurrence.values())
    if any(type(x) not in (int,float) or not math.isfinite(x) for x in nums):raise ValueError('finite gate inputs')
    checks={}
    for c,threshold in zip(CONTROLS,(.5,.2,.2)):
        checks[c+'_mean']=mean(values[c])>=threshold
        checks[c+'_orders']=all(v>=0 for v in values[c]) if stage=='A' else sum(v>0 for v in values[c])>=3
        checks[c+'_recurrence']=recurrence[c]>=-.1
    checks['domains']=min(domain_deltas)>=-2.
    if stage=='B_NEW':checks['worst_C_order']=min(values['C'])>=-.5
    passed=all(checks.values())
    return dict(passed=passed,checks=checks,next_execution_authorized=False)


def summarize(rows):
    result={}
    for subset in SUBSETS:
        selected=[r for r in rows if subset=='all_dev' or r['subset']==subset];result[subset]={}
        domains=sorted({r['domain'] for r in selected})
        if len(domains)!=4:raise ValueError('four domains in each registered subset')
        for key,rs in [('all_contents',selected)]+[('domain:'+d,[r for r in selected if r['domain']==d]) for d in domains]:
            block=dict(n=len(rs),channels={},cost=dict(host_seconds=sum(r['trace']['host_seconds'] for r in rs),evaluator_seconds=sum(r['evaluator_seconds'] for r in rs),pipeline_seconds=sum(r['pipeline_seconds'] for r in rs)))
            for ch in ('OD','OC','macro'):
                preds={p:channels([dict(metrics=r['metrics'][p]) for r in rs],ch) for p in ('pre','q','post')}
                v=dict(dice_percent={p:distribution([100*m['dice'] for m in ms]) for p,ms in preds.items()},post_minus_pre=paired(preds['post'],preds['pre'],ch=='macro'))
                if ch!='macro':
                    v['assd']={p:dict(conditional=distribution([m['assd'] for m in ms if m['assd'] is not None]),defined=sum(m['assd'] is not None for m in ms),undefined=sum(m['assd'] is None for m in ms)) for p,ms in preds.items()}
                    v['pixel_counts']={p:{k:sum(m[k] for m in ms) for k in ('pred_pixels','gt_pixels','intersection','pred_empty','pred_full','gt_empty','gt_full')} for p,ms in preds.items()}
                block['channels'][ch]=v
            block['unlabeled_channels']={ch:{k:distribution([r['trace']['channels'][i][k] for r in rs]) for k in ('n_fg','n_bg','rho','S0','Sw','Sperm','a','b','residual_fg_sse','residual_bg_sse','weighted_fg_sse','weighted_bg_sse','actual_logit_gradient_l2','bce_fg_sum','bce_bg_sum','weighted_bce_sum')} for i,ch in enumerate(('OD','OC'))}
            block['bn_gradient_l2']=distribution([r['trace']['bn_gradient_l2'] for r in rs]);block['adam_affine_displacement_l2']=distribution([r['trace']['adam_affine_displacement_l2'] for r in rs])
            result[subset][key]=block
        result[subset]['domain_equal_dice_percent']={p:{ch:mean(result[subset]['domain:'+d]['channels'][ch]['dice_percent'][p]['mean'] for d in domains) for ch in ('OD','OC','macro')} for p in ('pre','q','post')}
    return result


def gate_inputs(rows):
    orders=sorted(o for o,a in rows if a=='C' and o!=4)
    value=lambda o,a,d=None:mean(mean(mean(m['dice'] for m in r['metrics']['post'])*100 for r in rows[o,a] if r['subset']=='remaining_dev' and r['domain']==domain) for domain in ([d] if d else sorted({r['domain'] for r in rows[o,a]})))
    domains=sorted({r['domain'] for r in rows[0,'C']})
    return ({c:[value(o,'R_BAL')-value(o,c) for o in orders] for c in CONTROLS},[mean(value(o,'R_BAL',d)-value(o,'C',d) for o in orders) for d in domains],{c:value(4,'R_BAL')-value(4,c) for c in CONTROLS})


def completed(out,reg,scope):
    """Read and validate this finite run, including the original C path smoke."""
    if scope not in ('A','B_NEW'):raise ValueError('R6 finite stage scope')
    out=Path(out);receipt=read(out/'receipt.json');ident=receipt['binding'];jobs=matrix(scope)
    if read(out/'R6_SCOPE.json')!=dict(schema='R6_REGIONAL_CONSISTENCY_V1',scope=scope,run_id=ident['run_id']):raise ValueError('R6 output ownership')
    wanted=dict(science_sha256=SCIENCE_SHA,registration_digest=registration_digest(reg),stream_digest=stream_summary(reg)['stream_digest'],production_fingerprint=fingerprint(),scope=scope)
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
    if recipe!=SMOKE or output_bytes(out)>caps['bytes']:raise ValueError('recipe/output cap')
    for i in range(slots):
        smoke=read(out/f'device{i}/smoke.completion.json');bound(smoke,binding(receipt,i))
        if smoke['status']!='MECHANICAL_SMOKE_COMPLETE' or smoke['C_parity_valid'] is not True or smoke['physical']!={k:recipe[k] for k in PHYSICAL} or smoke['recipe']!=recipe:raise ValueError('C parity smoke')
    rows={};traces={}
    for j in jobs:
        p=out/j['job_id'];done=read(p/'completion.json');identity=binding(receipt,assigned[j['job_id']],j);bound(done,identity)
        t=lines(p/'unlabeled.jsonl');e=lines(p/'evaluation.jsonl');ordered=stream(reg,j['order'])
        rs=join(t,e,ordered,j,identity)
        if done['status']!='TRAJECTORY_COMPLETE' or done['records']!=1951 or done['adam_step']!=1951 or done['physical']!={k:v*1951 for k,v in PHYSICAL.items()} or not 0<=done['seconds']<=caps['trajectory_seconds']:raise ValueError('trajectory completion')
        if type(done['peak_allocated_bytes']) is not int or done['peak_allocated_bytes']<0:raise ValueError('measured peak memory')
        rows[j['order'],j['arm']]=rs;traces[j['order'],j['arm']]=t
    return receipt,rows,traces



def reuse_A(out,reg,expected_binding):
    receipt,rows,_=completed(out,reg,'A')
    if receipt['binding']!=expected_binding:raise PermissionError('original A execution binding')
    current=read(Path(out)/'current_result.json')
    if current.get('valid') is not True or current['status']!='R6A_COMPLETE_ELIGIBLE_FOR_REVIEW' or current.get('binding')!=receipt['binding']:raise PermissionError('qualified R6 A required')
    validate_cross_GT(rows)
    if not gate(*gate_inputs(rows),'A')['passed']:raise PermissionError('A gate replay failed')
    return receipt,rows


def validate_cross_GT(rows):
    seen={}
    for rs in rows.values():
        for r in rs:
            value=[(m['gt_pixels'],m['total_pixels'],m['gt_empty'],m['gt_full']) for m in r['metrics']['pre']]
            if seen.setdefault(r['group_id'],value)!=value:raise ValueError('cross-trajectory GT binding')


def recompute(out,reg):
    out=Path(out);raw=read(out/'receipt.json');ident=raw['binding'];scope=ident['scope']
    # Full ownership is checked BEFORE invalidating any current result pointer.
    if scope not in ('A','B_NEW') or not isinstance(ident.get('run_id'),str) or not ident['run_id'] or read(out/'R6_SCOPE.json')!=dict(schema='R6_REGIONAL_CONSISTENCY_V1',scope=scope,run_id=ident['run_id']):raise ValueError('not an owned R6 output')
    invalidate(out)
    try:
        import torch
        if torch.cuda.is_initialized():raise ValueError('CPU-only scalar analyzer')
        receipt,rows,traces=completed(out,reg,scope)
        m=read(out/'matrix.processes.json')
        result=dict(binding=ident,scope=scope,execution_started=True,next_execution_authorized=False,
            physical_formal={k:sum(rs[-1]['trace']['cumulative'][k] for rs in rows.values()) for k in PHYSICAL},
            physical_smoke={k:sum(read(out/f'device{i}/smoke.completion.json')['physical'][k] for i in range(len(receipt['devices']))) for k in PHYSICAL},
            failed_prefixes=[],wall_seconds=m['wall_seconds'],worker_seconds=m['active_seconds'],
            trajectory_costs={j['job_id']:{k:read(out/j['job_id']/'completion.json')[k] for k in ('seconds','peak_allocated_bytes')} for j in receipt['jobs']},
            limitations=['CPU verifies scalar relationships, not tensor positions, gradients or ASSD geometry.','Runtime histogram/checksum and gradient hook evidence is tested independently; summaries cannot prove spatial weights.','Equal calls do not imply equal FLOPs or time. Output logit norm matching is not parameter-gradient or Adam-displacement matching.','Exposed development contents; new orders do not create independent patients.'])
        if scope=='B_NEW':
            old,oldrows=reuse_A(receipt['reuse_A'],reg,receipt['reuse_A_binding']);rows.update(oldrows)
            if set(rows)!={(j['order'],j['arm']) for j in matrix('AB')}:raise ValueError('20 trajectory reuse coverage')
            result['reuse_A_binding']=old['binding'];result['physical_total_formal']={k:sum(rs[-1]['trace']['cumulative'][k] for rs in rows.values()) for k in PHYSICAL}
        validate_cross_GT(rows)
        values,domains,recurrence=gate_inputs(rows);selection=gate(values,domains,recurrence,scope)
        result.update(selection=selection,primary_comparisons_pp=values,domain_comparisons_pp=domains,recurrence_comparisons_pp=recurrence,
            status=('R6A_COMPLETE_ELIGIBLE_FOR_REVIEW' if selection['passed'] else 'R6A_COMPLETE_NO_ADVANCE') if scope=='A' else 'R6_EXPERIMENT_COMPLETE')
        result['trajectories']={f'{o}:{a}':summarize(rs) for (o,a),rs in rows.items()}
        orders=sorted({o for o,a in rows});result['matched_controls']={}
        for o in orders:
            result['matched_controls'][str(o)]={}
            for subset in SUBSETS:
                select=lambda a:[r for r in rows[o,a] if subset=='all_dev' or r['subset']==subset]
                selected={a:select(a) for a in ARMS};domains=sorted({r['domain'] for r in selected['C']})
                result['matched_controls'][str(o)][subset]={}
                for label,d in [('all_contents',None)]+[('domain:'+d,d) for d in domains]:
                    ss={a:[r for r in rs if d is None or r['domain']==d] for a,rs in selected.items()}
                    if any([r['group_id'] for r in rs]!=[r['group_id'] for r in ss['C']] for rs in ss.values()):raise ValueError('paired content alignment')
                    result['matched_controls'][str(o)][subset][label]={c:{ch:paired(channels([dict(metrics=r['metrics']['post']) for r in ss['R_BAL']],ch),channels([dict(metrics=r['metrics']['post']) for r in ss[c]],ch),ch=='macro') for ch in ('OD','OC','macro')} for c in CONTROLS}
        primary=[o for o in orders if o!=4]
        result['primary_order_equal']={s:{a:{ch:mean(result['trajectories'][f'{o}:{a}'][s]['domain_equal_dice_percent']['post'][ch] for o in primary) for ch in ('OD','OC','macro')} for a in ARMS} for s in SUBSETS}
        publish(out,result,'# R6 scalar results\n\n'+result['status']+'\n\nNext execution is not authorized.\n')
        return result
    except BaseException as exc:
        invalidate(out,'INCOMPLETE',type(exc).__name__+': '+str(exc));raise
