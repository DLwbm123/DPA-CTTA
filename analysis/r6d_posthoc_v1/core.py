"""R6-D scalar-only analysis. No production imports or model dependencies."""
import ast
import bisect
import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import struct
import sys

ARMS = ('C', 'R_BAL', 'R_SCALE', 'R_SHUFFLE')
CONTROLS = ('C', 'R_SCALE', 'R_SHUFFLE')
STREAMS = (0, 1, 4)
CHANNELS = ('OD', 'OC', 'macro')
PARTITIONS = ('EMPTY_FG', 'FG_MINOR_UPPER_CAP', 'UNCLIPPED', 'BG_MINOR_LOWER_CAP', 'EMPTY_BG')
N = 262144
TAG = 'POST_HOC_EXPLORATORY'


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write(p, obj):
    with Path(p).open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def disjoint(a, b):
    a, b = Path(a).resolve(), Path(b).resolve()
    if a == b or a in b.parents or b in a.parents:
        raise ValueError('source/output overlap')


def regular(p):
    p = Path(p)
    if p.is_symlink() or not p.is_file() or p.stat().st_nlink != 1:
        raise ValueError('ordinary independent file required')


def install_guard(read_files, read_roots, output):
    """Audit-hook confinement, not an OS sandbox; no native/model libraries loaded."""
    files = {str(Path(p).resolve()) for p in read_files}
    roots = [Path(p).resolve() for p in read_roots]
    output = Path(output).resolve()
    def inside(p, root):
        return p == root or root in p.parents
    def hook(event, args):
        if event == 'import' and str(args[0]).split('.')[0] in {'torch', 'tensorflow', 'cupy', 'jax', 'subprocess', 'ctypes', 'dpa_ctta'}:
            raise PermissionError('model/GPU/process import forbidden')
        if event.startswith(('subprocess.', 'socket.', 'ctypes.')) or event in {'os.system', 'os.exec', 'os.posix_spawn', 'os.fork'}:
            raise PermissionError('model/GPU/process/network entry forbidden')
        if event == 'open' and not isinstance(args[0], int):
            p = Path(args[0]).resolve()
            mode, flags = args[1], args[2]
            writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
            if writing and not inside(p, output):
                raise PermissionError('write outside new work directory')
            if not writing and str(p) not in files and not any(inside(p, r) for r in roots + [output]):
                raise PermissionError('read outside scalar/code allowlist')
        if event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.utime', 'os.truncate', 'os.rename', 'os.link', 'os.symlink'}:
            paths = args[:2] if event in {'os.rename', 'os.link', 'os.symlink'} else args[:1]
            if any(not inside(Path(p).resolve(), output) for p in paths if isinstance(p, (str, bytes))):
                raise PermissionError('source metadata mutation forbidden')
    sys.addaudithook(hook)
    return hook


def functions(path, names, env):
    """Execute only audited function ASTs; never execute a production module import."""
    tree = ast.parse(Path(path).read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in selected} != set(names):
        raise ValueError('frozen validator function coverage')
    banned = {'recompute', 'invalidate', 'publish', 'launch', 'worker', 'trajectory', 'smoke'}
    if banned.intersection(names):
        raise PermissionError('mutating/model entrypoint forbidden')
    for node in selected:
        for child in ast.walk(node):
            if isinstance(child, (ast.Import, ast.ImportFrom)):
                raise PermissionError('imports inside extracted validator forbidden')
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def validators(root, source_bytes):
    """Reuse unchanged scalar/metadata call chains at the historical fingerprint."""
    root = Path(root)
    for name, expected in [('r1_science_v1.json','e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2'),
                           ('r3_science_v1.json','73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e')]:
        if digest(root/'configs'/name)!=expected:raise ValueError('frozen metadata science bytes')
    common = dict(hashlib=hashlib, json=json, math=math, re=re, struct=struct,
                  Path=Path, Counter=collections.Counter, mean=statistics.mean,
                  ROOT=root, digest=digest)
    r1 = functions(root/'src/dpa_ctta/r1/plan.py', ['registration_digest','stream','binding','bound'],
                   dict(common, science=lambda: read(root/'configs/r1_science_v1.json'),
                        COUNTS=dict(REFUGE=400, ORIGA=650, REFUGE_Valid=800, Drishti_GS=101)))
    # Remove only type annotations from this metadata-only function by providing their types.
    kernel = functions(root/'src/dpa_ctta/r3/kernels.py', ['recurring_stream'],
                       dict(common, Mapping=dict, Sequence=list))
    r3 = functions(root/'src/dpa_ctta/r3/plan.py', ['stream','stream_summary'],
                   dict(common, science=lambda: read(root/'configs/r3_science_v1.json'),
                        registration_digest=r1['registration_digest'], primary_stream=r1['stream'],
                        recurring_stream=kernel['recurring_stream']))
    physical = dict(network_forwards=8, loss_backward_calls=1, adam_calls=1, jacobian_vjp_calls=0)
    spec = read(root/'analysis/r6d_posthoc_v1/input/R6D_ANALYSIS_SPEC.json')
    hist = spec['historical_binding']
    plan = functions(root/'src/dpa_ctta/r6_regional_consistency/plan.py',
                     ['stream','stream_summary','fingerprint','matrix','allocation'],
                     dict(common, registration_digest=r1['registration_digest'], old_stream=r3['stream'],
                          old_summary=r3['stream_summary'], REGISTRATION=hist['registration_digest'],
                          RECURRENCE=hist['stream_digest'], ARMS=ARMS))
    p2 = functions(root/'src/dpa_ctta/p2_analysis.py', ['validate_metric'], dict(common))
    r5 = functions(root/'src/dpa_ctta/r5_update_acceptance/analyze.py', ['validate_metric'],
                   dict(common, inherited_validate_metric=p2['validate_metric']))
    smoke = dict(recipe='R6_ALL_ARMS4_OLD_C4_V1', network_forwards=160, loss_backward_calls=20,
                 adam_calls=20, jacobian_vjp_calls=0, seed=20260907, pixel_indices=[0,1,2,3], rtol=1e-4, atol=1e-5)
    env = dict(common, **{k:r1[k] for k in ['bound','binding','registration_digest']})
    env.update({k:plan[k] for k in ['stream','stream_summary','fingerprint','matrix','allocation']})
    env.update(validate_metric=r5['validate_metric'], PHYSICAL=physical, SMOKE=smoke,
               SCIENCE_SHA=hist['science_sha256'], CONTROLS=CONTROLS,
               output_bytes=lambda unused: source_bytes)
    names = ['read','lines','f32','near','audit_channel','replay','join','gate','gate_inputs','completed','validate_cross_GT']
    return functions(root/'src/dpa_ctta/r6_regional_consistency/analyze.py', names, env)


def divide(a, b):
    return (a/b, None) if b else (None, 'ZERO_DENOMINATOR')


def quantile(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    h = (len(xs)-1)*q
    lo = int(h)
    return xs[lo] + (xs[min(lo+1, len(xs)-1)]-xs[lo])*(h-lo)


def partition(n, total=N):
    if type(n) is not int or not 0 <= n <= total:
        raise ValueError('partition count')
    if n == 0: return PARTITIONS[0]
    if n == total: return PARTITIONS[4]
    if 9*n < total: return PARTITIONS[1]
    if 9*n > 8*total: return PARTITIONS[3]
    return PARTITIONS[2]


def bins(value, edges):
    return 'MISSING' if value is None else 'Q'+str(bisect.bisect_left(edges, value)+1)


def decompose(bpre, bpost, cpre, cpost):
    pre, step, post = bpre-cpre, (bpost-bpre)-(cpost-cpre), bpost-cpost
    if not math.isclose(post, pre+step, rel_tol=0, abs_tol=1e-10):
        raise ValueError('decomposition identity')
    return dict(delta_pre=pre, delta_step=step, delta_post=post)


def pair(left, right):
    def index(rows):
        out = {r['group_id']:r for r in rows}
        if len(out) != len(rows): raise ValueError('duplicate content')
        return out
    a,b=index(left),index(right)
    if a.keys()!=b.keys(): raise ValueError('missing matched content')
    result=[]
    for k,x in a.items():
        y=b[k]
        if any(x[t]!=y[t] for t in ['sample_id','domain','subset']):
            raise ValueError('cross-stream metadata mismatch')
        result.append((x,y))
    return result


def timeline(rows, registered_chunks=None, registered_segments=None):
    """All arrivals determine positions, then the caller filters scoring rows."""
    totals=collections.Counter(r['domain'] for r in rows)
    seen=collections.Counter(); occurrences=collections.Counter(); last={}; result={}; segments=[]
    for visit,r in enumerate(rows,1):
        d=r['domain']
        if not segments or segments[-1]['domain']!=d:
            occurrences[d]+=1
            segments.append(dict(domain=d,first_visit=visit,count=0))
        seg=segments[-1];seg['count']+=1
        previous=seen[d];seen[d]+=1
        result[r['group_id']]=dict(visit=visit,segment=len(segments),segment_position=seg['count'],
            is_switch_segment=len(segments)>1, domain_occurrence=occurrences[d],
            prior_domain_exposures=previous, gap_since_domain_visit=visit-last[d] if d in last else None,
            within_domain_rank=seen[d],time_half='first' if seen[d]<=totals[d]//2 else 'second',
            timing='initial' if len(segments)==1 else 'early_1_8' if seg['count']<=8 else 'later_9_plus')
        last[d]=visit
    if registered_segments is not None and segments!=registered_segments:
        raise ValueError('registered contiguous segments')
    chunks=registered_chunks or [dict(s) for s in segments]
    covered=[]
    for i,chunk in enumerate(chunks,1):
        for visit in range(chunk['first_visit'],chunk['first_visit']+chunk['count']):
            if rows[visit-1]['domain']!=chunk['domain']:raise ValueError('registered chunk domain')
            result[rows[visit-1]['group_id']]['chunk']=i;covered.append(visit)
    if covered!=list(range(1,len(rows)+1)):raise ValueError('chunk coverage')
    return result,segments,chunks


def local_values(row, c):
    t=row['trace']; z=t['channels'][c]; q=row['metrics']['q'][c]
    P,G,I,total=(q[k] for k in ['pred_pixels','gt_pixels','intersection','total_pixels'])
    cells=dict(TP=I,FP=P-I,FN=G-I,TN=total-P-G+I)
    if any(type(x) is not int or x<0 for x in cells.values()) or total!=N:
        raise ValueError('infeasible q cells')
    v={k:z[k] for k in ('n_fg','n_bg','rho','applied_w_fg','applied_w_bg','S0','Sw','Sperm','a','b',
        'residual_fg_sse','residual_bg_sse','weighted_fg_sse','weighted_bg_sse','bce_sum','bce_fg_sum','bce_bg_sum',
        'weighted_bce_sum','shuffled_bce_sum','residual_weighted_dot','residual_shuffled_dot','actual_logit_gradient_l2')}
    v.update(bn_gradient_l2=t['bn_gradient_l2'],adam_affine_displacement_l2=t['adam_affine_displacement_l2'],
             rho_clipped=int(z['rho_clipped']),empty_partition_fallback=int(z['fallback'] is not None),**cells)
    ratios=dict(pi=(z['n_fg'],N), bce_fg_fraction=(z['bce_fg_sum'],z['bce_sum']),
        f0=(z['residual_fg_sse'],z['S0']),fw=(z['weighted_fg_sse'],z['Sw']),
        cos_bal_plain=(z['residual_weighted_dot'],math.sqrt(z['S0']*z['Sw'])),
        cos_perm_plain=(z['residual_shuffled_dot'],math.sqrt(z['S0']*z['Sperm'])),
        q_precision=(I,P),q_recall=(I,G),FP_fraction=(cells['FP'],N),FN_fraction=(cells['FN'],N),
        area_bias=(P-G,N),E_plain=(cells['FP']+cells['FN'],N),
        E_regional=(z['applied_w_fg']*cells['FP']+z['applied_w_bg']*cells['FN'],N))
    reasons={}
    for k,(num,den) in ratios.items():
        v[k],reasons[k]=divide(num,den)
    v['foreground_energy_shift']=None if v['f0'] is None or v['fw'] is None else v['fw']-v['f0']
    v['E_regional_minus_plain']=v['E_regional']-v['E_plain']
    arm=t['arm']
    v['actual_foreground_energy_fraction']=v['fw'] if arm=='R_BAL' else v['f0'] if arm in ('C','R_SCALE') else None
    v['actual_foreground_bce_fraction']=divide(z['applied_w_fg']*z['bce_fg_sum'],z['weighted_bce_sum'])[0] if arm=='R_BAL' else v['bce_fg_fraction'] if arm in ('C','R_SCALE') else None
    for pred in ('pre','q','post'):v[pred+'_dice']=100*row['metrics'][pred][c]['dice']
    v['step']=v['post_dice']-v['pre_dice']
    for k,val in v.items():
        if val is None:reasons[k]='NOT_RECORDED_PERMUTED_PARTITION' if arm=='R_SHUFFLE' and k.startswith('actual_') else 'ZERO_DENOMINATOR'
    return v,{k:r for k,r in reasons.items() if r}


def summarize(values, weights, ids, channel='macro', stream=4):
    finite=[(v,w) for v,w in zip(values,weights) if v is not None]
    xs=[v for v,w in finite]; ws=[w for v,w in finite]
    n=len(xs); unique=len(set(ids)); mass=math.fsum(ws)
    net=math.fsum(v*w for v,w in finite)
    tail=math.ceil(n*.1)
    factor=.5 if channel!='macro' else 1.
    result=dict(inferential_status=TAG,n_rows=len(values),n_unique=unique,n_defined=n,n_missing=len(values)-n,
        support='LOW_SUPPORT' if unique<30 else 'DESCRIPTIVE',mean=statistics.mean(xs) if xs else None,
        median=quantile(xs,.5),p10=quantile(xs,.1),p90=quantile(xs,.9),minimum=min(xs) if xs else None,
        worst_decile_mean=statistics.mean(sorted(xs)[:tail]) if tail else None,tail_count=tail,
        positive=sum(v>0 for v in xs),negative=sum(v<0 for v in xs),zero=sum(v==0 for v in xs),
        weight_sum=mass,weighted_conditional_mean=net/mass if mass else None,
        signed_net=net,positive_mass=math.fsum(w*max(v,0) for v,w in finite),negative_mass=math.fsum(w*min(v,0) for v,w in finite),
        macro_contribution=net*factor,primary_contribution=net*factor*(.5 if stream in (0,1) else 1) if stream in (0,1,'primary') else None,
        null_reason=None if n else 'EMPTY_CELL_OR_NO_DEFINED_VALUES')
    return result


PRIVATE_KEYS={'group_id','sample_id','patient_id','pid','pgid','device_uuid','image_path','mask_path'}
def public_safe(value):
    if isinstance(value,dict):
        if PRIVATE_KEYS.intersection(value):raise ValueError('private public-export key')
        for v in value.values():public_safe(v)
    elif isinstance(value,list):
        for v in value:public_safe(v)
    elif isinstance(value,str) and any(t in value for t in ['/Users/','/home/','/data_nas/','/remote-home/','GPU-']):
        raise ValueError('private public-export value')


def csv_write(path, records):
    public_safe(records)
    fields=list(dict.fromkeys(k for r in records for k in r))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(records)


def analysis(rows, stream_meta, spec, output):
    """All fixed analyses; no selection by outcome and no fitted thresholds."""
    output=Path(output); output.mkdir()
    counts=spec['inputs']['remaining_dev_domain_counts']; domains=sorted(counts)
    private=[];locals_by_key={};availability=collections.Counter();missing=collections.Counter()
    timelines={}; structure={}
    for s in STREAMS:
        tm,segs,chunks=timeline(rows[s,'C'],stream_meta['chunk_schedule'] if s==4 else None,
                               stream_meta['contiguous_domain_segments'] if s==4 else None)
        timelines[s]=tm;structure[s]=(segs,chunks)
        for arm in ARMS:
            pair(rows[s,'C'],rows[s,arm])
            for row in rows[s,arm]:
                for ci,ch in enumerate(CHANNELS[:2]):
                    vals,reasons=local_values(row,ci)
                    locals_by_key[s,arm,row['group_id'],ch]=(vals,reasons)
                    for k,v in vals.items():availability[(arm,k)]+=v is not None
                    for k,r in reasons.items():missing[(arm,k,r)]+=1
        for control in CONTROLS:
            for b,c in pair(rows[s,'R_BAL'],rows[s,control]):
                if b['visit']!=c['visit']:raise ValueError('within-stream visit mismatch')
                parts=[]
                for ci,ch in enumerate(CHANNELS[:2]):
                    vals=decompose(*[100*x for x in (b['metrics']['pre'][ci]['dice'],b['metrics']['post'][ci]['dice'],c['metrics']['pre'][ci]['dice'],c['metrics']['post'][ci]['dice'])]);parts.append(vals)
                parts.append({k:statistics.mean([v[k] for v in parts]) for k in parts[0]})
                for ch,vals in zip(CHANNELS,parts):
                    private.append(dict(stream=s,control=control,group_id=b['group_id'],sample_id=b['sample_id'],domain=b['domain'],subset=b['subset'],channel=ch,**timelines[s][b['group_id']],**vals))
    # Private joined decomposition remains outside public export.
    with (output.parent/'joined.private.jsonl').open('x') as f:
        for r in private:f.write(json.dumps(r)+'\n')
    scored=[r for r in private if r['subset']=='remaining_dev']
    tables={k:[] for k in ['MAIN_DECOMPOSITION','DOMAIN_CHANNEL','ORDER_CONTRASTS','RECURRENCE_SEGMENTS','SUPERVISION_SUMMARY','PSEUDO_TARGET_ERRORS','STRATIFIED_ASSOCIATIONS']}
    def decomp_table(dest, rr, meta, stream, channel):
        weights=[1/(4*counts[r['domain']])*(.5 if stream=='primary' else 1) for r in rr]
        for metric in ['delta_pre','delta_step','delta_post']:
            tables[dest].append(dict(meta,metric=metric,**summarize([r[metric] for r in rr],weights,[r['group_id'] for r in rr],channel,stream)))
    for s in (*STREAMS,'primary'):
        for c in CONTROLS:
            for ch in CHANNELS:
                rr=[r for r in scored if (r['stream'] in (0,1) if s=='primary' else r['stream']==s) and r['control']==c and r['channel']==ch]
                decomp_table('MAIN_DECOMPOSITION',rr,dict(stream=s,control=c,channel=ch),s,ch)
                for d in domains:
                    decomp_table('DOMAIN_CHANNEL',[r for r in rr if r['domain']==d],dict(stream=s,control=c,channel=ch,domain=d),s,ch)
    for hi,lo in [(1,0),(4,0),(4,1)]:
        for ch in CHANNELS:
            l=[r for r in scored if r['stream']==hi and r['control']=='C' and r['channel']==ch]
            r=[r for r in scored if r['stream']==lo and r['control']=='C' and r['channel']==ch]
            joined=[dict(x,**{m:x[m]-y[m] for m in ['delta_pre','delta_step','delta_post']}) for x,y in pair(l,r)]
            for d in ['ALL',*domains]:
                rr=[r for r in joined if d=='ALL' or r['domain']==d]
                decomp_table('ORDER_CONTRASTS',rr,dict(contrast=f'{hi}-{lo}',control='C',domain=d,channel=ch),4,ch)
    for s in STREAMS:
        segs,chunks=structure[s]
        for kind,blocks in [('segment',segs),('chunk',chunks)]:
            for i,block in enumerate(blocks,1):
                tm=timelines[s][rows[s,'C'][block['first_visit']-1]['group_id']]
                for phase in ['all','early_1_8','later_9_plus'] if kind=='segment' and tm['is_switch_segment'] else ['all']:
                    for ch in CHANNELS:
                        for c in CONTROLS:
                            rr=[r for r in scored if r['stream']==s and r['control']==c and r['channel']==ch and r[kind]==i and (phase=='all' or r['timing']==phase)]
                            meta=dict(stream=s,control=c,channel=ch,domain=block['domain'],block_type=kind,block=i,phase=phase,
                                      first_visit=block['first_visit'],all_arrivals=block['count'],prior_domain_exposures=tm['prior_domain_exposures'],
                                      domain_occurrence=tm['domain_occurrence'],gap_since_domain_visit=tm['gap_since_domain_visit'],is_switch_segment=tm['is_switch_segment'])
                            decomp_table('RECURRENCE_SEGMENTS',rr,meta,s,ch)
    error_keys={'TP','FP','FN','TN','q_precision','q_recall','FP_fraction','FN_fraction','area_bias','E_plain','E_regional','E_regional_minus_plain','q_dice'}
    for s in STREAMS:
        for arm in ARMS:
            for ch in CHANNELS[:2]:
                for subset in ['remaining_dev','all_dev']:
                    for d in ['ALL',*domains]:
                        rr=[r for r in rows[s,arm] if (subset=='all_dev' or r['subset']==subset) and (d=='ALL' or r['domain']==d)]
                        for k in next(iter(locals_by_key.values()))[0]:
                            vv=[locals_by_key[s,arm,r['group_id'],ch][0][k] for r in rr]
                            # all_dev descriptive weights use registered full-domain counts.
                            denominators=counts if subset=='remaining_dev' else stream_meta['domain_counts']
                            ww=[1/(4*denominators[r['domain']]) for r in rr]
                            reasons=collections.Counter(locals_by_key[s,arm,r['group_id'],ch][1].get(k) for r in rr if locals_by_key[s,arm,r['group_id'],ch][0][k] is None)
                            context='own_state_actual_BAL_regional' if arm=='R_BAL' else 'hypothetical_base_regional_at_own_state'
                            dest='PSEUDO_TARGET_ERRORS' if k in error_keys else 'SUPERVISION_SUMMARY'
                            tables[dest].append(dict(stream=s,arm=arm,channel=ch,domain=d,subset=subset,metric=k,regional_context=context,
                                null_reasons=json.dumps(dict(reasons),sort_keys=True),**summarize(vv,ww,[r['group_id'] for r in rr],ch,s)))
    # Independent C-only bins. Every class/quartile/missing and half cell is emitted, including empty cells.
    strat_metrics=['pi','a','foreground_energy_shift','adam_affine_displacement_l2','f0','fw','bce_fg_fraction','cos_bal_plain','cos_perm_plain',
                   'q_precision','q_recall','FP_fraction','FN_fraction','area_bias','E_plain','E_regional','E_regional_minus_plain',
                   'actual_foreground_energy_fraction','actual_foreground_bce_fraction']
    for s in STREAMS:
        for d in domains:
            for ch in CHANNELS[:2]:
                rr=[r for r in scored if r['stream']==s and r['domain']==d and r['channel']==ch and r['control']=='C']
                for variable in ['partition','a','foreground_energy_shift','adam_affine_displacement_l2']:
                    anchor=lambda r:locals_by_key[s,'C',r['group_id'],ch][0]
                    finite=[anchor(r)[variable] for r in rr if variable!='partition' and anchor(r)[variable] is not None]
                    edges=[quantile(finite,q) for q in [.25,.5,.75]] if finite else []
                    labels=PARTITIONS if variable=='partition' else ('Q1','Q2','Q3','Q4','MISSING')
                    label=lambda r:partition(anchor(r)['n_fg']) if variable=='partition' else bins(anchor(r)[variable],edges)
                    for group in labels:
                        for half in ['all','first','second']:
                            selected=[r for r in rr if label(r)==group and (half=='all' or r['time_half']==half)]
                            base=dict(stream=s,domain=d,channel=ch,anchor='C',variable=variable,stratum=group,time_half=half,
                                      edges=json.dumps(edges),population_n=len(rr),anchor_finite_n=len(finite) if variable!='partition' else len(rr))
                            for control in CONTROLS:
                                lookup={r['group_id']:r for r in scored if r['stream']==s and r['domain']==d and r['channel']==ch and r['control']==control}
                                decomp_table('STRATIFIED_ASSOCIATIONS',[lookup[r['group_id']] for r in selected],dict(base,control=control,arm='BAL-minus-control'),s,ch)
                            for arm in ARMS:
                                for k in strat_metrics:
                                    vals=[locals_by_key[s,arm,r['group_id'],ch][0][k] for r in selected]
                                    reasons=collections.Counter(locals_by_key[s,arm,r['group_id'],ch][1].get(k) for r in selected if locals_by_key[s,arm,r['group_id'],ch][0][k] is None)
                                    tables['STRATIFIED_ASSOCIATIONS'].append(dict(base,arm=arm,control='NOT_APPLICABLE',metric=k,
                                        state_context='secondary_post_treatment' if arm=='R_BAL' else 'own_state',null_reasons=json.dumps(dict(reasons),sort_keys=True),
                                        **summarize(vals,[1/(4*counts[d])]*len(vals),[r['group_id'] for r in selected],ch,s)))
    for name,rr in tables.items():csv_write(output/(name+'.csv'),rr)
    fields=dict(inferential_status=TAG,stored_GT_metrics_used=True,available=[dict(arm=a,field=k,n_defined=n,total_rows=5853*2) for (a,k),n in sorted(availability.items())],
        nulls=[dict(arm=a,field=k,reason=r,count=n) for (a,k,r),n in sorted(missing.items())],
        NOT_AVAILABLE=spec['analysis']['D3']['unavailable']+['C0 matched no-update forgetting quantity','SHUFFLE partition-specific actually applied gradient/BCE foreground share'])
    write(output/'FIELD_AVAILABILITY.json',fields)
    # Machine-readable full tables remain in CSV; compact JSON preserves key decomposition and coverage.
    write(output/'AGGREGATE.json',dict(inferential_status=TAG,main=tables['MAIN_DECOMPOSITION'],domains=tables['DOMAIN_CHANNEL'],
        order_contrasts=tables['ORDER_CONTRASTS'],table_rows={k:len(v) for k,v in tables.items()}))
    # Contribution partitions must add back exactly within scalar tolerance.
    for main in tables['MAIN_DECOMPOSITION']:
        selected=[r for r in tables['DOMAIN_CHANNEL'] if all(r[k]==main[k] for k in ['stream','control','channel','metric'])]
        if not math.isclose(sum(r['signed_net'] for r in selected),main['signed_net'],rel_tol=0,abs_tol=1e-10):raise ValueError('domain contribution sum')
    for s in STREAMS:
        for d in domains:
            for ch in CHANNELS[:2]:
                for variable in ['partition','a','foreground_energy_shift','adam_affine_displacement_l2']:
                    for c in CONTROLS:
                        for metric in ['delta_pre','delta_step','delta_post']:
                            cells=[r for r in tables['STRATIFIED_ASSOCIATIONS'] if (r['stream'],r['domain'],r['channel'],r['variable'],r['control'],r['metric'],r['time_half'])==(s,d,ch,variable,c,metric,'all')]
                            target=next(r['signed_net'] for r in tables['DOMAIN_CHANNEL'] if (r['stream'],r['domain'],r['channel'],r['control'],r['metric'])==(s,d,ch,c,metric))
                            if not math.isclose(sum(r['signed_net'] for r in cells),target,rel_tol=0,abs_tol=1e-10):raise ValueError('stratum contribution sum')
    for s in STREAMS:
        for c in CONTROLS:
            for ch in CHANNELS:
                for metric in ['delta_pre','delta_step','delta_post']:
                    target=next(r['signed_net'] for r in tables['MAIN_DECOMPOSITION'] if (r['stream'],r['control'],r['channel'],r['metric'])==(s,c,ch,metric))
                    for kind in ['segment','chunk']:
                        total=sum(r['signed_net'] for r in tables['RECURRENCE_SEGMENTS'] if (r['stream'],r['control'],r['channel'],r['metric'],r['block_type'],r['phase'])==(s,c,ch,metric,kind,'all'))
                        if not math.isclose(total,target,rel_tol=0,abs_tol=1e-10):raise ValueError('segment/chunk contribution sum')
    return dict(table_rows={k:len(v) for k,v in tables.items()},joined_rows=len(private),identity_atol_pp=1e-10)
