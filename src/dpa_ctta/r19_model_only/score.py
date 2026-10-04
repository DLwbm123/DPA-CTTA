"""Independent terminal-only CPU labels; complete paired and proxy diagnostics."""
import json,math,statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr,rankdata
from ..r10_12h_core.run import read,save,sha
from ..r8_ba.journal import verify_online_complete,_digest
from ..r7_target_screen.runner import TargetReader
from ..r16_evidence_correction.score import metrics
from ..r16_evidence_correction.structure import topology,containment
from ..r16_evidence_correction.report import cells,mean,write_csv,public
ARMS=('G','G_HALF','G_VAL','G_MEM');WIDTH=4*65536;SOFT_WIDTH=2*512*512*4

def dice(mask,label):
    out=[]
    for p,y in zip(mask,label):
        d=int(p.sum())+int(y.sum());out.append(1. if not d else float(2*(p&y).sum()/d))
    return float(np.mean(out))

def score(c,guard):
    r=Path(c['output_root']);state=read(r/'RUN_STATE.json');jobs=[f'{a}_o{o}' for a in ARMS for o in (0,1)]
    if state['status']!='TARGET_MATRIX_TERMINAL' or any(state['jobs'].get(j)!='COMPLETE' for j in jobs):raise ValueError('all eight complete before labels')
    if any(read(p)['active'] for p in (r/'processes').glob('online_*.json')):raise ValueError('online process not retired')
    dest=r/'score';dest.mkdir(exist_ok=False);reader=TargetReader(c['target_root'],256*1024**2,'mask');scalars=[];proxies=[];receipts=[]
    refs=read(r/'scorer/references.json')
    for ref in refs:
        values=read(ref['values_path'])
        if sha(values)!=ref['values_sha256']:raise ValueError('sealed historical scalar changed')
        scalars.extend(dict(x,condition='G_HISTORICAL' if ref['condition']=='G' else ref['condition'],origin='SEALED_MATCHED_REFERENCE') for x in values)
    for arm in ARMS:
        for o in (0,1):
            jid=f'{arm}_o{o}';path=r/'target'/jid;rows=read(r/'scorer'/f'FULL_o{o}.json');on=read(path/'online_complete.json');soft=read(path/'soft_complete.json')
            verify_online_complete(path,jid,on['identity']['context_sha256'],c['manifests'][o]['sha256'],1951,WIDTH)
            if soft['bytes']!=1951*SOFT_WIDTH or _digest(path/'probabilities.f32')!=soft['sha256']:raise ValueError('soft seal')
            traces=[json.loads(x) for x in (path/'visits.jsonl').open()];out=dest/(jid+'.private.jsonl')
            with (path/'predictions.bits').open('rb') as f,(path/'probabilities.f32').open('rb') as sf,out.open('x') as handle:
                for i,(m,t) in enumerate(zip(rows,traces)):
                    guard();masks=np.unpackbits(np.frombuffer(f.read(WIDTH),dtype=np.uint8)).reshape(4,2,512,512).astype(bool);q=np.frombuffer(sf.read(SOFT_WIDTH),dtype='<f4').reshape(2,512,512);y=reader.read(m);gt=y.numpy()[0];vals=metrics(masks[0],y)
                    for ch,met in enumerate(vals):
                        met['soft_dice']=float((2*np.sum(q[ch]*gt[ch],dtype=np.float64)+1e-6)/(np.sum(q[ch],dtype=np.float64)+np.sum(gt[ch],dtype=np.float64)+1e-6));met['Brier']=float(np.mean((q[ch].astype(np.float64)-gt[ch])**2))
                    row=dict(condition=arm,order=o,seed=c['science']['native_seed'],visit=i+1,content=m['group_id'],domain=m['domain'],subset=m['subset'],metrics=vals,accepted=t['accepted'],reason=t['reason'],seconds=t['seconds'],empty_foreground=[not x.any() for x in masks[0]],containment_violations=containment(masks[0]),topology=topology(masks[0]),origin='NEW_FULL_ONLINE')
                    scalars.append(row);handle.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
                    if t['diagnostic']:
                        d=t['diagnostic'];ds=[dice(a,gt.astype(bool)) for a in masks];v0=d['V_skip'];v1=d['V_full'];vh=d['V_half']
                        proxies.append(dict(order=o,visit=i+1,subset=m['subset'],full_delta_pp=100*(ds[2]-ds[1]),half_delta_pp=100*(ds[3]-ds[1]),full_proxy=None if v0 is None or v1 is None else float(np.mean(v0)-np.mean(v1)),half_proxy=None if v0 is None or vh is None else float(np.mean(v0)-np.mean(vh)),accept_full=d['accept_full'],accept_half=d['accept_half'],local_best_minus_full_pp=100*(max(ds[1:])-ds[2]),local_best_minus_half_pp=100*(max(ds[1:])-ds[3])))
                if f.read(1) or sf.read(1):raise ValueError('output trailing rows')
            receipt=dict(job=jid,visits=1951,principal=sum(m['subset']=='remaining_dev' for m in rows),scalar_sha256=_digest(out),online_sha256=sha(on));save(dest/(jid+'.complete.json'),receipt);receipts.append(receipt)
    f=dest/'all-scalars.private.jsonl'
    with f.open('x') as handle:
        for x in scalars:handle.write(json.dumps(x,sort_keys=True,allow_nan=False)+'\n')
    save(dest/'proxy.private.json',proxies);result=dict(status='COMPLETE',new_visits=15608,new_principal=13560,trajectories=receipts,all_scalar_sha256=_digest(f),all_eight_retired_before_labels=True,target_soft='real float32 probabilities',source_reads=0)
    save(r/'SCORER_RECEIPT.json',result);return result

def report(c,state):
    r=Path(c['output_root']);out=r/'public';out.mkdir(exist_ok=True)
    for name in ('MODEL_ONLY_AUDIT.json','RESOURCE_LEDGER.json','RUN_STATE.json','PREFLIGHT.json','PROFILE_ADMISSION.json','SCORER_RECEIPT.json'):
        if (r/name).exists():save(out/name,public(read(r/name)))
    main=[];domain=[];paired=[];diagnostics=[];groups=defaultdict(list)
    f=r/'score/all-scalars.private.jsonl'
    if f.exists() and (r/'SCORER_RECEIPT.json').exists():
        for line in f.open():
            x=json.loads(line);groups[x['condition'],x['order']].append(x)
    for (a,o),rs in sorted(groups.items()):
        cs=cells(rs);primary=[x for x in rs if x['subset']=='remaining_dev'];v=mean(rs);g=groups['G',o]
        main.append(dict(condition=a,order=o,visits=len(rs),principal=len(primary),macro_Dice_percent=100*v,imageweighted_Dice_percent=100*st.mean(st.mean(m['dice'] for m in x['metrics']) for x in primary),delta_G_pp=100*(v-mean(g)),acceptance_rate=st.mean(x['accepted'] for x in rs) if 'accepted' in rs[0] else None,seconds_per_image=st.mean(x['seconds'] for x in rs) if 'seconds' in rs[0] else None,empty_foreground_rate=st.mean(any(x['empty_foreground']) for x in rs) if 'empty_foreground' in rs[0] else None,components_mean=st.mean(sum(ch[0] for ch in x['topology']) for x in primary) if 'topology' in rs[0] else None,holes_mean=st.mean(sum(ch[1] for ch in x['topology']) for x in primary) if 'topology' in rs[0] else None,containment_violations=sum(x['containment_violations'] for x in primary) if 'containment_violations' in rs[0] else None))
        for (d,ch),xs in sorted(cs.items()):
            bs=[x['assd'] for x in xs if x.get('assd') is not None and math.isfinite(x['assd'])];soft=[x['soft_dice'] for x in xs if x.get('soft_dice') is not None];br=[x['Brier'] for x in xs if x.get('Brier') is not None]
            domain.append(dict(condition=a,order=o,domain=d,channel=ch,n=len(xs),Dice_percent=100*st.mean(x['dice'] for x in xs),delta_G_pp=100*(st.mean(x['dice'] for x in xs)-st.mean(x['dice'] for x in cells(g)[d,ch])),ASSD_mean=st.mean(bs) if bs else None,ASSD_valid=len(bs),ASSD_undefined=len(xs)-len(bs),soft_Dice=st.mean(soft) if soft else None,Brier=st.mean(br) if br else None))
        if a in ARMS:
            trace=[json.loads(x) for x in (r/'target'/f'{a}_o{o}'/'visits.jsonl').open()]
            diagnostics.append(dict(condition=a,order=o,visits=len(trace),accepted=sum(x['accepted'] for x in trace),unavailable=sum(x['reason']=='unavailable' for x in trace),area_rejected=sum(x['reason']=='area' for x in trace),validation_rejected=sum(x['reason']=='validation' for x in trace),memory_veto=sum(x['memory_veto'] for x in trace),empty_memory=sum(x['memory_empty'] for x in trace),writes=sum(x['writes'] for x in trace),evictions=sum(x['evictions'] for x in trace),expired=sum(x['expired'] for x in trace),mean_memory_age=st.mean(x['memory_age'] for x in trace if x['memory_age'] is not None) if any(x['memory_age'] is not None for x in trace) else None))
            for control in ('G','G_HALF','G_VAL','C0','DS','G_HISTORICAL'):
                b=groups[control,o];ix={x['content']:x for x in b};ds=[]
                for x in primary:
                    y=ix[x['content']]
                    if (x['domain'],x['subset'])!=(y['domain'],y['subset']):raise ValueError('pairing role drift')
                    ds.append(100*st.mean(u['dice']-v['dice'] for u,v in zip(x['metrics'],y['metrics'])))
                paired.append(dict(condition=a,control=control,order=o,n=len(ds),macro_delta_pp=100*(v-mean(b)),imageweighted_delta_pp=st.mean(ds),improved=sum(x>0 for x in ds),unchanged=sum(x==0 for x in ds),worsened=sum(x<0 for x in ds),median_pp=st.median(ds),p05_pp=float(np.quantile(ds,.05))))
    proxy=[]
    p=r/'score/proxy.private.json'
    if p.exists() and (r/'SCORER_RECEIPT.json').exists():
        raw=read(p)
        for o in (0,1):
            for action in ('full','half'):
                xs=[x for x in raw if x['order']==o and x['subset']=='remaining_dev' and x[action+'_proxy'] is not None];pred=np.array([-x[action+'_proxy'] for x in xs]);bad=np.array([x[action+'_delta_pp']<0 for x in xs]);auc=None
                if bad.any() and (~bad).any():auc=float((rankdata(pred)[bad].sum()-bad.sum()*(bad.sum()+1)/2)/(bad.sum()*(~bad).sum()))
                corr=spearmanr([x[action+'_proxy'] for x in xs],[x[action+'_delta_pp'] for x in xs]).statistic if len(xs)>1 else float('nan')
                proxy.append(dict(order=o,action=action,n=len(xs),sampled_positions=sum(x['order']==o for x in raw),spearman=float(corr) if np.isfinite(corr) else None,harm_AUROC=auc,harmful_accepted=sum(x['accept_'+action] and x[action+'_delta_pp']<0 for x in xs),beneficial_rejected=sum(not x['accept_'+action] and x[action+'_delta_pp']>0 for x in xs),local_best_minus_action_pp=st.mean(x['local_best_minus_'+action+'_pp'] for x in xs) if xs else None))
    summaries=[]
    for a in ('G_HALF','G_VAL','G_MEM'):
        rs=[x for x in paired if x['condition']==a and x['control']=='G'];ds=[x for x in domain if x['condition']==a]
        if len(rs)==2:summaries.append(dict(condition=a,mean_delta_G_pp=st.mean(x['macro_delta_pp'] for x in rs),priority_signal=st.mean(x['macro_delta_pp'] for x in rs)>=.5 and all(x['macro_delta_pp']>0 for x in rs) and st.mean(x['imageweighted_delta_pp'] for x in rs)>=0 and min(x['delta_G_pp'] for x in ds)>=-2,worst_G_pp=min(x['delta_G_pp'] for x in ds)))
    for item in summaries:
        control='G_VAL' if item['condition']=='G_MEM' else 'G_HALF' if item['condition']=='G_VAL' else None
        if control:
            q=[x for x in paired if x['condition']==item['condition'] and x['control']==control]
            item['mechanism_control']=control;item['mechanism_delta_pp']=st.mean(x['macro_delta_pp'] for x in q)
            item['mechanism_signal']=(item['mechanism_delta_pp']>=.2 and all(x['macro_delta_pp']>=0 for x in q)) if control=='G_VAL' else all(x['macro_delta_pp']>0 for x in q)
    reproduction=[]
    if groups:
        for o in (0,1):
            old={x['content']:x for x in groups['G_HISTORICAL',o]};new=groups['G',o]
            diffs=[abs(a['dice']-b['dice']) for x in new for a,b in zip(x['metrics'],old[x['content']]['metrics'])]
            reproduction.append(dict(order=o,visits=len(new),max_hard_Dice_difference=max(diffs),exact_hard_metrics=all(all(a[k]==b[k] for k in ('dice','intersection','pred_pixels','gt_pixels')) for x in new for a,b in zip(x['metrics'],old[x['content']]['metrics']))))
    save(out/'HISTORICAL_G_REPRODUCTION.json',reproduction)
    save(out/'SUMMARY.json',summaries)
    for name,rows in [('main.csv',main),('domain-channel.csv',domain),('paired.csv',paired),('proxy_diagnostics.csv',proxy),('acceptance-memory.csv',diagnostics)]:
        if rows:write_csv(out/name,rows,list(rows[0]))
    costs=read(r/'RESOURCE_LEDGER.json');costrows=[dict(phase=x['phase'],attempt=x['attempt'],status=x['status'],wall_seconds=x.get('wall_seconds',x['ended']-x['started']),**x['cost']) for x in costs['attempts']]
    if costrows:write_csv(out/'cost.csv',costrows,list(dict.fromkeys(k for x in costrows for k in x)))
    text=['# R19 model-only validation and memory','',f'Status: {state["status"]}. Execution `{c["code_sha"]}`.','',
        'Only the supplied segmentation checkpoint is loaded as a learned asset. No source images, source labels, actor, carrier, fitted scaler, Fisher or correction head. Existing 800x800 ROI inputs are retained by delegated user decision; original crop centers are unverified. Results are conditional on supplied ROI inputs, not proof that original ROI preparation was label-free.',
        '', 'Eight new full physical trajectories, four conditions times two orders;1951 arrivals and1695 principal observations each. Other arrivals retain128 legacy_dev and128 p1_extension_dev roles. C0 uses historical current-image BN statistics, not source-running-statistics N. C0/DS references reused only after sealed provenance and pairing checks. All development content was previously exposed; orders share content, patient dependence unknown.',
        '', 'GPU budget8h including profile/diagnostics/failures; CPU scoring billed separately. Native updates cost9F2BP1Adam; this implementation explicitly adds an original-image pre-output forward. G_VAL adds4 validation forwards; G_MEM adds up to2 history forwards. Rejected attempts still incur the candidate computation. All model/Adam/temporary-gradient state is rolled back; the main augmentation stream advances once per arrival even on rejection. Diagnostic half branches use cloned native RNG and never change the full main trajectory.',
        '', 'Stages: IMPLEMENTED; TESTED see synthetic and preflight receipts; RUN/SCORED only when corresponding complete receipts exist. Missing metrics are NA, never inferred from hard masks. No R20 authorization.', '',f'GPU-worker seconds: {costs["gpu_seconds"]:.3f}.', '', '| Condition | Order | Macro Dice % | vs new G pp |','|---|---:|---:|---:|']
    text += [f'| {x["condition"]} | {x["order"]} | {x["macro_Dice_percent"]:.6f} | {x["delta_G_pp"]:+.6f} |' for x in main]
    text += ['', 'Priority: >=+0.5pp versus new matched G, both orders positive, nonnegative image-weighted mean, and no domain/channel/order loss >2pp. Compare G_VAL with G_HALF; memory added-value signal requires >=+0.2pp versus G_VAL with neither order negative. Sparse counterfactual opportunities are local to sampled states, not a trajectory oracle or independent validation.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')
