"""One independent CPU scorer after the complete registered matrix is terminal."""
import json,math,statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
from ..r10_12h_core.run import read,save,sha
from ..r8_ba.journal import verify_online_complete,_digest
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader
from ..r16_evidence_correction.score import metrics,WIDTH
from ..r16_evidence_correction.structure import topology,containment
from ..r16_evidence_correction.report import public,cells,mean,write_csv
from .method import ARMS,SEED
TARGET_SEED=SEED


def score(c,guard):
    root=Path(c['output_root']);state=read(root/'RUN_STATE.json');lock=read(root/'EXPERIMENT_LOCK.json');p=lock['payload']
    if state['status']!='TARGET_MATRIX_TERMINAL' or any(v in ('RUNNING','NOT_RUN') for v in state['jobs'].values()):raise ValueError('target matrix not terminal')
    if lock['sha256']!=sha(p):raise ValueError('target lock changed')
    dest=root/'score';dest.mkdir(exist_ok=False);scalars=[];completed=[];failed=[]
    for ref in p['reused']:
        values=read(ref['values_path'])
        if sha(values)!=ref['values_sha256']:raise ValueError('reference changed')
        scalars.extend(dict(x,origin='SEALED_R17_REFERENCE',soft_metrics='NA_MASK_ONLY') for x in values)
    reader=TargetReader(c['bindings']['target_root'],256*1024**2,'mask')
    try:
        for arm in ARMS:
            for order in (0,1):
                jid=f'{arm}_o{order}';status=state['jobs'].get(jid,'NOT_RUN_STOPPED')
                if status!='COMPLETE':failed.append(dict(job=jid,status=status));continue
                path=root/'target'/jid;on=read(path/'online_complete.json');rows=read(p['manifests'][order]['path'])
                verify_online_complete(path,jid,on['identity']['context_sha256'],rows_sha(rows),len(rows),WIDTH)
                output=dest/(jid+'.private.jsonl')
                with (path/'predictions.bits').open('rb') as f,output.open('x') as out:
                    for i,m in enumerate(rows):
                        guard();raw=f.read(WIDTH)
                        if len(raw)!=WIDTH:raise ValueError('prediction truncation')
                        mask=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool);label=reader.read(m)
                        if (label[:,1]>label[:,0]).any():raise ValueError('label containment protocol changed')
                        row=dict(condition=arm,order=order,seed=TARGET_SEED,visit=i+1,content=m['group_id'],domain=m['domain'],subset=m['subset'],metrics=metrics(mask,label),containment_violations=containment(mask),fragments_holes=topology(mask),origin='NEW_STATEFUL_INTEGRATED',soft_metrics='NA_MASK_ONLY')
                        out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');scalars.append(row)
                    if f.read(1):raise ValueError('prediction tail')
                receipt=dict(job=jid,visits=len(rows),principal=sum(m['subset']=='remaining_dev' for m in rows),online_sha256=sha(on),scalar_sha256=_digest(output),independent_CPU=True)
                save(dest/(jid+'.complete.json'),receipt);completed.append(receipt)
    finally:reader.after_check()
    f=dest/'all-scalars.private.jsonl'
    with f.open('x') as out:
        for row in scalars:out.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    result=dict(status='COMPLETE',completed=completed,failed=failed,scalar_rows=len(scalars),scalar_sha256=_digest(f),source_lock_sha256=lock['sha256'],mask_reads=dict(reader.counts),embargo='all eight registered trajectories terminal',target_soft_metrics='NA: packed masks are not probabilities')
    save(root/'SCORER_RECEIPT.json',result);return result


def report(c,state):
    root=Path(c['output_root']);out=root/'public';out.mkdir(exist_ok=True);groups=defaultdict(list)
    for name in ('RESOURCE_LEDGER.json','RUN_STATE.json','EXPERIMENT_LOCK.json','PROFILE_ADMISSION.json','SOURCE_PRECHECK.json','SOURCE_RL.json','SOURCE_CACHE_STATE.json','SCORER_RECEIPT.json','SOURCE_CACHE_RETIRED.json'):
        if (root/name).exists():save(out/name,public(read(root/name)))
    f=root/'score/all-scalars.private.jsonl'
    if f.exists():
        for line in f.open():
            r=json.loads(line);groups[r['condition'],r['order']].append(r)
    main=[];domain=[];paired=[];structures=[];summary=[]
    for (arm,order),rows in sorted(groups.items()):
        values=cells(rows);v=mean(rows);entry=dict(condition=arm,order=order,seed=rows[0].get('seed'),visits=len(rows),principal=sum(x['subset']=='remaining_dev' for x in rows),Dice_percent=v*100)
        for ref in ('C0','G','RL_ORIGINAL','FIXED_GRATA','WARM_GRATA'):
            control=groups.get((ref,order));entry['delta_'+ref+'_pp']=100*(v-mean(control)) if control else None
        main.append(entry)
        for (d,ch),xs in sorted(values.items()):
            b=[x['assd'] for x in xs if x.get('assd') is not None and math.isfinite(x['assd'])]
            r=dict(condition=arm,order=order,domain=d,channel=ch,n=len(xs),Dice_percent=100*st.mean(x['dice'] for x in xs),ASSD_mean=st.mean(b) if b else None,ASSD_valid_n=len(b),ASSD_undefined_n=len(xs)-len(b),TP=sum(x['intersection'] for x in xs),FP=sum(x['pred_pixels']-x['intersection'] for x in xs),FN=sum(x['gt_pixels']-x['intersection'] for x in xs))
            for ref in ('C0','G','RL_ORIGINAL','FIXED_GRATA','WARM_GRATA'):
                ys=cells(groups[ref,order])[d,ch] if (ref,order) in groups else None;r['delta_'+ref+'_pp']=100*st.mean(x['dice'] for x in xs)-100*st.mean(x['dice'] for x in ys) if ys else None
            domain.append(r)
        rs=[x for x in rows if x['subset']=='remaining_dev']
        if all('containment_violations' in x for x in rs):structures.append(dict(condition=arm,order=order,n=len(rs),containment_violations=sum(x['containment_violations'] for x in rs),topology='available per-image privately; aggregate below',fragments_holes_aggregate=_sum_topology([x['fragments_holes'] for x in rs])))
    contrasts=[('RL_GRATA',x) for x in ('G','RL_ORIGINAL','FIXED_GRATA','WARM_GRATA')]+[('WARM_GRATA','FIXED_GRATA')]
    for a,b in contrasts:
        for order in (0,1):
            aa=groups.get((a,order));bb=groups.get((b,order))
            if not aa or not bb:paired.append(dict(condition=a,control=b,order=order,status='NA_INCOMPLETE'));continue
            ix={x['content']:x for x in bb};delta=[]
            for x in aa:
                y=ix[x['content']]
                if (x['domain'],x['subset'])!=(y['domain'],y['subset']):raise ValueError('paired content/role mismatch')
                if x['subset']=='remaining_dev':delta.append(100*st.mean(u['dice']-v['dice'] for u,v in zip(x['metrics'],y['metrics'])))
            paired.append(dict(condition=a,control=b,order=order,status='COMPLETE',macro_delta_pp=100*(mean(aa)-mean(bb)),n=len(delta),image_weighted_delta_pp=st.mean(delta),p05_pp=float(np.quantile(delta,.05)),median_pp=st.median(delta),p95_pp=float(np.quantile(delta,.95)),improved=sum(x>0 for x in delta),unchanged=sum(x==0 for x in delta),worsened=sum(x<0 for x in delta)))
    for arm in ARMS:
        a=[x for x in main if x['condition']==arm];ds=[x for x in domain if x['condition']==arm]
        complete=len(a)==2;gain=st.mean(x['delta_G_pp'] for x in a) if complete else None;worst=min(x['delta_G_pp'] for x in ds) if ds else None
        summary.append(dict(condition=arm,status='COMPLETE' if complete else 'INCOMPLETE',mean_delta_G_pp=gain,both_orders_positive=complete and all(x['delta_G_pp']>0 for x in a),priority_signal=complete and gain>=.5 and all(x['delta_G_pp']>0 for x in a),worst_domain_channel_G_pp=worst,risk_over_2pp=worst is not None and worst<-2))
    save(out/'SUMMARY.json',summary);save(out/'STRUCTURE.json',structures)
    for name,rows in [('main.csv',main),('domain-channel.csv',domain),('paired.csv',paired)]:
        if rows:write_csv(out/name,rows,list(dict.fromkeys(k for x in rows for k in x)))
    text=['# R18 full RL use/write FiLM with GraTa','',f"Status: {state['status']}. Execution source: `{c['code_sha']}`.",'',
    'Native GraTa adapts BN affine parameters on the current image; its updated BN state is copied to the identical frozen FiLM model, whose zero-modulation logits are checked exactly against GraTa on every image. The original carrier, 10-action actor, full FiLM use and recurrent m/q/h write memory are retained. Final output includes FiLM. GraTa updates do not depend on RL actions; this coupling is not end-to-end policy control of the GraTa optimizer.','',
    'RL_ORIGINAL is the unchanged historical R10 POST actor on new complete streams. RL_GRATA retrains WARM and GR_RET_EMA in the GraTa-adapted source environment. FIXED_GRATA fixes ALL actions: gain .8, residual zero, write .5. WARM_GRATA uses the same newly supervised WARM actor without RL refinement. C0 and G are exact sealed full historical references. Eight new streams each have 1951 visits and 1695 primary observations.','',
    'Source uses 16 fixed episodes per fit/val fold with four balanced modes and 32 visits each. GraTa state and observations are cached because their evolution is independent of policy actions. The frozen original carrier and normalization are reused. WARM runs 1000 supervised updates and GR_RET_EMA 1024 rounds with 2 optimizer epochs; endpoints are fixed, no checkpoint search. Retention probes hold BN at the final query state of the four-step window for both memory branches. The source simulator pool differs from historical R10, so RL_ORIGINAL versus RL_GRATA alone cannot isolate the GraTa mechanism.','',
    'Primary priority signal: RL_GRATA minus G at least +0.5 pp averaged across orders, both orders positive. Any domain/channel loss greater than 2 pp is a risk. Attribution to RL additionally requires positive matched gains over WARM_GRATA and FIXED_GRATA in both orders. All conditions and adverse results are retained.','',
    'The actor is trained only on source labels, then frozen on target; this is not online reward learning. Target updates are native BN adaptation plus recurrent unlabeled controller memory. Source hard/soft and action diagnostics are retained. Target soft metrics are NA because target outputs store masks only. ASSD includes valid/undefined denominators. Macro and image-weighted paired effects must both be read.','',
    'All development images were exposed in prior research. The two orders contain the same images, patient dependence is unknown, and one policy fit seed is not independent replication. Native GraTa seed 20260907 and policy/source seed 20260924 have different roles. No clinical or held-out claim.','',
    '| Condition | Mean delta vs GraTa (pp) | Worst domain/channel (pp) | Priority signal |','|---|---:|---:|---|']
    for x in summary:text.append(f"| {x['condition']} | {x['mean_delta_G_pp']} | {x['worst_domain_channel_G_pp']} | {x['priority_signal']} |")
    text+=['','Full order/domain/paired aggregates, source hard/soft metrics and fixed endpoints are in the accompanying files. RESOURCE_LEDGER includes failed attempts and source qualification. R17 historical cost remains separate. No automatic successor is authorized.']
    (out/'REPORT.md').write_text('\n'.join(text)+'\n')


def _sum_topology(rows):
    if not rows:return None
    if isinstance(rows[0],dict):return {k:_sum_topology([r[k] for r in rows]) for k in rows[0]}
    if isinstance(rows[0],list):return [_sum_topology([r[i] for r in rows]) for i in range(len(rows[0]))]
    if isinstance(rows[0],(int,float)):return sum(rows)
    return None
