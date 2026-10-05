"""Deidentified campaign deliverables and content-clustered paired uncertainty."""
import csv,json,statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from .runtime import read,save,sha,ledger
from .score import load_rows,summary,cells
import re

def public(value):
    if isinstance(value,dict):return {k:public(v) for k,v in value.items() if k not in ('path','values_path','output_root','target_root','checkpoint_path','protocol_path','argv','content','group_id','sample_id','image_path','mask_path')}
    if isinstance(value,(list,tuple)):return [public(v) for v in value]
    if isinstance(value,str):return re.sub(r"/(?:data_nas|Users|tmp)/[^\s'\"]+",'[PRIVATE_PATH]',value)
    return value

def csvout(path,rows):
    columns=list(dict.fromkeys(k for r in rows for k in r)) or ['status']
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list,tuple)) else v for k,v in r.items()})

def mean(xs):return st.mean(xs) if xs else None

def aggregate(rows):
    main=[];domains=[];groups=defaultdict(list)
    for r in rows:
        if r['subset']!='remaining_dev':continue
        groups[r['condition'],r['seed'],r['order'],r['role']].append(r)
        groups[r['condition'],r['seed'],r['order'],'PRIMARY_ALL'].append(r)
    for (condition,seed,order,role),rs in sorted(groups.items()):
        s=summary(rs);soft=[m for r in rs for m in r['metrics'] if m.get('soft_dice') is not None]
        main.append(dict(condition=condition,seed=seed,order=order,role=role,observations=len(rs),content_identities=len({r['content'] for r in rs}),macro_Dice_percent=s['macro'],imageweighted_Dice_percent=s['imageweighted'],soft_Dice_percent=None if not soft else 100*st.mean(m['soft_dice'] for m in soft),Brier=mean([m['Brier'] for m in soft]),soft_channel_observations=len(soft),containment_violations=sum(r['containment_violations'] for r in rs),empty_OD=sum(r['empty_foreground'][0] for r in rs),empty_OC=sum(r['empty_foreground'][1] for r in rs)))
        for (domain,channel),dice in s['cells'].items():
            ms=[m for r in rs if r['domain']==domain for m in r['metrics'] if m['channel']==channel];assd=[m['assd'] for m in ms if m['assd'] is not None]
            domains.append(dict(condition=condition,seed=seed,order=order,role=role,domain=domain,channel=channel,observations=len(ms),Dice_percent=dice,ASSD_defined=len(assd),ASSD_undefined=len(ms)-len(assd),ASSD_conditional_mean_pixels=mean(assd),pred_empty=sum(m['pred_empty'] for m in ms),gt_empty=sum(m['gt_empty'] for m in ms)))
    return main,domains

def bootstrap_pair(rows,a,b,seeds,role='SEALED_REVIEW',repeats=2000):
    index={(r['condition'],r['seed'],r['order'],r['content']):r for r in rows if r['role']==role and r['seed'] in seeds};contents=sorted({k[3] for k in index if k[0]==a});bydomain=defaultdict(list);trajectories=[];single_cells=[];paired=[]
    for seed in seeds:
        for order in (0,1):
            aa=[r for k,r in index.items() if k[:3]==(a,seed,order)];bb=[r for k,r in index.items() if k[:3]==(b,seed,order)]
            if not aa or {r['content'] for r in aa}!={r['content'] for r in bb}:return dict(status='INCOMPLETE',candidate=a,baseline=b,role=role,seeds=seeds),[],[]
            sa,sb=summary(aa),summary(bb);trajectories.append(dict(seed=seed,order=order,delta_pp=sa['macro']-sb['macro'],imageweighted_delta_pp=sa['imageweighted']-sb['imageweighted']))
            for k in sa['cells']:single_cells.append(dict(seed=seed,order=order,domain=k[0],channel=k[1],delta_pp=sa['cells'][k]-sb['cells'][k]))
    for content in contents:
        vectors=[];domain=None
        for seed in seeds:
            for order in (0,1):
                ar,br=index[a,seed,order,content],index[b,seed,order,content];domain=ar['domain'];d=[100*(x['dice']-y['dice']) for x,y in zip(ar['metrics'],br['metrics'])];vectors.append(d);paired.append(dict(seed=seed,order=order,domain=domain,delta_pp=st.mean(d)))
        bydomain[domain].append(np.mean(vectors,axis=0))
    rng=np.random.default_rng(20261005);draws=[];cis=[]
    for domain,v in sorted(bydomain.items()):
        arr=np.asarray(v);idx=rng.integers(0,len(arr),size=(repeats,len(arr)));samples=arr[idx].mean(axis=1);draws.append(samples.mean(axis=1))
        for k,channel in enumerate(('OD','OC')):cis.append(dict(candidate=a,baseline=b,role=role,domain=domain,channel=channel,identities=len(arr),delta_pp=float(arr[:,k].mean()),CI_low_pp=float(np.quantile(samples[:,k],.025)),CI_high_pp=float(np.quantile(samples[:,k],.975))))
    boot=np.mean(draws,axis=0);seedavg=defaultdict(list)
    for x in single_cells:seedavg[x['order'],x['domain'],x['channel']].append(x['delta_pp'])
    values=[p['delta_pp'] for p in paired];result=dict(status='COMPLETE',candidate=a,baseline=b,role=role,seeds=seeds,content_identities=len(contents),paired_observations=len(paired),delta_pp=st.mean(x['delta_pp'] for x in trajectories),imageweighted_delta_pp=st.mean(x['imageweighted_delta_pp'] for x in trajectories),order_delta_pp=[st.mean(x['delta_pp'] for x in trajectories if x['order']==o) for o in (0,1)],positive_trajectories=sum(x['delta_pp']>0 for x in trajectories),trajectory_count=len(trajectories),worst_seed_averaged_cell_pp=min(st.mean(v) for v in seedavg.values()),worst_single_seed_cell_pp=min(x['delta_pp'] for x in single_cells),CI_low_pp=float(np.quantile(boot,.025)),CI_high_pp=float(np.quantile(boot,.975)),improved=sum(v>0 for v in values),tied=sum(v==0 for v in values),degraded=sum(v<0 for v in values),worst_observation_delta_pp=min(values),tenth_percentile_pp=float(np.quantile(values,.1)),resampling_unit='content identity; all orders and seeds coupled',bootstrap_repeats=repeats,patient_dependence='UNKNOWN',trajectories=trajectories)
    negative=[dict(candidate=a,baseline=b,role=role,aggregation='individual_seed',**x) for x in single_cells if x['delta_pp']<0]
    negative += [dict(candidate=a,baseline=b,role=role,aggregation='seed_mean',order=k[0],domain=k[1],channel=k[2],delta_pp=st.mean(v)) for k,v in seedavg.items() if st.mean(v)<0]
    return result,cis,negative

def historical_parity(c,rows):
    root=Path(c['output_root']);binding=read(root/'scorer/HISTORICAL_BINDING.json');out=[]
    for order in (0,1):
        import hashlib
        raw=Path(binding['R19_G'][order]['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=binding['R19_G'][order]['receipt']['scalar_sha256']:raise ValueError('historical scalar seal mismatch')
        refs=[json.loads(x) for x in raw.splitlines()];new=sorted([x for x in rows if x['condition']=='G' and x['seed']==c['native_seed'] and x['order']==order],key=lambda x:x['visit']);manifest=read(root/'scorer'/f'FULL_o{order}.json')
        if len(new)!=1951:out.append(dict(order=order,status='INCOMPLETE'));continue
        differences=0
        for a,b,m in zip(new,refs,manifest):
            if b['content']!=m['group_id'] or a['content']!=m['image_sha256'] or a['domain']!=b['domain'] or a['subset']!=b['subset']:raise ValueError('historical identity/role mismatch')
            for x,y in zip(a['metrics'],b['metrics']):
                if x['channel']!=y['channel'] or any(x[k]!=y[k] for k in ('intersection','pred_pixels','gt_pixels','total_pixels','dice','assd')):differences+=1
        out.append(dict(order=order,status='PASS' if differences==0 else 'MISMATCH',visits=len(new),channel_metric_mismatches=differences))
    return out

def flat_diagnostics(values,prefix=''):
    out={}
    for k,v in values.items():
        name=prefix+k
        if k in ('weights','logit_gradient_regions'):
            for channel,row in zip(('OD','OC'),v):
                for region,value in zip(('foreground','boundary','background'),row):out[name+'.'+channel+'.'+region]=value
        elif isinstance(v,dict):out.update(flat_diagnostics(v,name+'.'))
        elif isinstance(v,(list,tuple)):
            for i,value in enumerate(v):out[name+'.'+str(i)]=value
        else:out[name]=v
    return out

def diagnostics(rows):
    groups=defaultdict(list);index={(r['condition'],r['seed'],r['order'],r['content']):r for r in rows};out=[]
    for r in rows:
        for k,v in flat_diagnostics(r['diagnostics']).items():
            if isinstance(v,(int,float)):
                base=index.get(('G',r['seed'],r['order'],r['content']));delta=None if base is None else 100*st.mean(x['dice']-y['dice'] for x,y in zip(r['metrics'],base['metrics']));groups[r['condition'],r['role'],k].append((v,delta))
    from scipy.stats import spearmanr
    for (condition,role,metric),vals in groups.items():
        paired=[(x,y) for x,y in vals if y is not None];rho=None
        if len(paired)>2 and len({x for x,y in paired})>1 and len({y for x,y in paired})>1:rho=float(spearmanr([x for x,y in paired],[y for x,y in paired]).statistic)
        out.append(dict(condition=condition,role=role,diagnostic=metric,count=len(vals),mean=st.mean(x for x,y in vals),Spearman_actual_Dice_change=rho,interpretation='post-hoc association only; no online feedback'))
    return out

def report(c,state):
    root=Path(c['output_root']);out=root/'public';out.mkdir(exist_ok=True);resource=ledger(c,state)
    for name,value in [('RUN_STATE.json',state),('RESOURCE_LEDGER.json',resource),('DATA_SPLIT_AUDIT.json',read(root/'DATA_SPLIT_AUDIT.json'))]:save(out/name,public(value))
    registered={x['id']:x for x in c['candidates']}
    for p in (root/'stages').glob('*.jobs.json'):
        for j in read(p):registered[j['candidate']['id']]=j['candidate']
    save(out/'CANDIDATES.json',public(list(registered.values())))
    search=[];history=[]
    for p in sorted((root/'stages').glob('*.selection.json')):
        for i,r in enumerate(read(p)['results']):search.append(dict(stage=p.name.split('.')[0],**r));history.append(dict(stage=p.name.split('.')[0],rank=i+1,**r))
    csvout(out/'SEARCH_RESULTS.csv',search);csvout(out/'SEARCH_HISTORY.csv',history)
    for p in [root/'FROZEN.json',root/'MECHANICAL_TESTS.json']:
        if p.exists():save(out/p.name,public(read(p)))
    rows=[];freeze=read(root/'FROZEN.json') if (root/'FROZEN.json').exists() else None
    # Partial scorer rows never become complete results.
    if (root/'FINAL_JOBS.json').exists():
        js=[j for j in read(root/'FINAL_JOBS.json') if (root/'scores/final'/(j['id']+'.complete.json')).exists()];rows=load_rows(root,'FINAL',js,True)
    main,domain=aggregate(rows);csvout(out/'SEALED_REVIEW_RESULTS.csv',[r for r in main if r['role']=='SEALED_REVIEW']);csvout(out/'FULL_RESULTS.csv',main);csvout(out/'DOMAIN_CHANNEL.csv',domain);csvout(out/'ABLATIONS.csv',[r for r in main if r['condition'].startswith('ABL_')]);csvout(out/'MECHANISM_DIAGNOSTICS.csv',diagnostics(rows))
    pairs=[];cis=[];negative=[];signal=None;parity=[];ablation_pairs=[];cost_pairs=[]
    if rows and freeze:
        parity=historical_parity(c,rows);save(out/'NATIVE_HISTORICAL_PARITY.json',parity)
        for baseline in dict.fromkeys(['G',freeze['best_simple_control']]):
            for role in ('SEARCH','SEALED_REVIEW'):
                result,ci,neg=bootstrap_pair(rows,freeze['primary'],baseline,freeze['seeds'],role);pairs.append(result);cis+=ci;negative+=neg
        review=next((x for x in pairs if x['baseline']=='G' and x['role']=='SEALED_REVIEW'),None);control=next((x for x in pairs if x['baseline']==freeze['best_simple_control'] and x['role']=='SEALED_REVIEW'),None)
        if review and control and review['status']==control['status']=='COMPLETE' and all(p['status']=='PASS' for p in parity):
            signal=review['delta_pp']>=.5 and min(review['order_delta_pp'])>0 and review['positive_trajectories']>=(5 if len(freeze['seeds'])==3 else 3) and review['imageweighted_delta_pp']>=0 and review['worst_seed_averaged_cell_pp']>=-2 and control['delta_pp']>0
        for ablation in [a['id'] for a in freeze['ablations']]+freeze['reused_ablations']:
            for role in ('SEARCH','SEALED_REVIEW'):
                result,_,_=bootstrap_pair(rows,freeze['primary'],ablation,[c['native_seed']],role);ablation_pairs.append(result)
        if freeze.get('extra_compute_control'):
            for role in ('SEARCH','SEALED_REVIEW'):
                result,_,_=bootstrap_pair(rows,freeze['primary'],freeze['extra_compute_control'],[c['native_seed']],role);cost_pairs.append(result)
        save(out/'FINAL_DECISION.json',dict(strong_development_signal=signal,primary_frozen=freeze['primary'],backup_not_substituted=True,review_historically_exposed=True,native_parity=parity,clinical_validation=False,follow_on_authorized=False))
    csvout(out/'ABLATION_COMPARISONS.csv',ablation_pairs);csvout(out/'EXTRA_COMPUTE_COMPARISONS.csv',cost_pairs)
    csvout(out/'PAIRED_SUMMARY.csv',pairs);csvout(out/'CONTENT_BOOTSTRAP_CI.csv',cis);csvout(out/'ALL_NEGATIVE_CELLS.csv',negative)
    costs=[dict(phase=a['phase'],attempt=a['attempt'],status=a['status'],wall_seconds=a['wall_seconds'],**a['cost']) for a in resource['attempts']];csvout(out/'COST.csv',costs)
    audit=dict(original_checkpoint_sha256=c['checkpoint_sha256'],permitted_assets=['original segmentation checkpoint and its BN statistics','pinned architecture and preprocessing','current image and causal prior target memory'],source_reads=0,online_target_label_reads=0,selection_mode=c['selection_mode'],ROI_crop_center_provenance='UNKNOWN',patient_dependence='UNKNOWN',prediction_and_state_assets_private=True,access_logs=len(list((root/'private').glob('*-access-audit.json'))),disk_bytes=resource['disk_bytes'],module_initialization='fresh original model; BN moments empty; teacher copy; zero-U/random-V adapter; empty prototype FIFO',native_BN_policy='current-image statistics; frozen f0 retains checkpoint inference statistics')
    save(out/'ASSET_AUDIT.json',audit)
    protocol=Path(c['protocol_path']).read_text();(out/'PROTOCOL.md').write_text(protocol+'\n\n## Execution amendments\n\n'+c['registration_notes']+'\n')
    lines=['# R20 model-only search', '',f"Status: **{state['status']}**. T0: {c['origin']['T0']}. Final delivery: {state.get('delivery','PENDING')}. Scientific results are complete only for sealed, fully scored trajectories.",'',f"GPU-worker time: {resource['gpu_worker_seconds']/3600:.3f} h / 48 h; CPU-worker time: {resource['cpu_worker_seconds']/3600:.3f} h; campaign elapsed: {resource['wall_seconds']/3600:.3f} h / 24 h. Measured disk: {resource['disk_bytes']/1024**3:.3f} GiB.",'','Outer selection uses SEARCH development labels in a separate CPU process. Online losses and memory are unlabeled. SEALED_REVIEW was historically exposed; this is a campaign-held review, not independent clinical or unseen test evidence. Orders and seeds reuse content; bootstrap keeps them coupled. Patient linkage and original ROI crop-center provenance remain UNKNOWN.','']
    if state.get('reason'):lines+=['Failure / incomplete reason: '+public(state['reason']),'']
    if freeze:
        lines += [f"Frozen primary: **{freeze['primary']}**; backup: {freeze['backup']}; strongest full-SEARCH simple control: **{freeze['best_simple_control']}**. Seeds: {freeze['seeds']}. The backup cannot replace the main conclusion after review.",'',f"Best full SEARCH ranked configuration: **{freeze['SEARCH_results'][0]['id']}**. The selected new-module primary is reported separately from a potentially stronger simple baseline.",'']
        for p in pairs:
            if p['status']=='COMPLETE':lines += [f"{p['role']}: {p['candidate']} minus {p['baseline']}: **{p['delta_pp']:+.4f} pp**, paired content bootstrap 95% CI [{p['CI_low_pp']:+.4f}, {p['CI_high_pp']:+.4f}] pp; {p['positive_trajectories']}/{p['trajectory_count']} positive order-seed trajectories; worst seed-averaged cell {p['worst_seed_averaged_cell_pp']:+.4f} pp; worst single-seed cell {p['worst_single_seed_cell_pp']:+.4f} pp.",'']
            else:lines += [f"{p['role']} comparison against {p['baseline']}: INCOMPLETE; no complete robustness claim.",'']
        for comparison in ablation_pairs+cost_pairs:
            if comparison['status']=='COMPLETE' and comparison['role']=='SEALED_REVIEW':lines += [f"Frozen base-seed mechanism/cost check: primary minus {comparison['baseline']} = {comparison['delta_pp']:+.4f} pp on review. A positive removal contrast supports an incremental effect only under this fixed configuration; it is not independent replication.",'']
        lines += [f"Predeclared strong development signal: **{signal}** (null means unqualified/incomplete, not a negative result).",'', 'Mechanism attribution: ABLATIONS.csv contains the frozen mechanism on C and key-component removal, or both component removals for a two-module primary. These are fixed-parameter transfer/ablation checks, not a fully tuned non-G route. COST.csv records every physical attempt, including profiling, failures and recovery. G_K2 and G_H025 remain registered screen controls; exact cost matching is not asserted.','']
    lines += ['Soft Dice/Brier use genuine saved float32 probabilities only; missing probabilities are NA. Every registered trajectory retains bit-packed masks and per-arrival scalars privately. Deployment requires the original checkpoint plus frozen algorithm configuration, fresh moments/memory and new-module initialization; adapted terminal weights are not initialization for another trajectory.','', 'History: full-stream R19 G 77.229693%, G_HALF 76.403522%, G_VAL 76.641735%, G_MEM 75.725186%. These are historical references, not comparators for compressed SEARCH streams.','', 'No automatic R21 or post-review retuning is authorized. Negative results and incomplete conditions remain in the ledger.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');return dict(public_files=[p.name for p in out.iterdir()],status=state['status'])
