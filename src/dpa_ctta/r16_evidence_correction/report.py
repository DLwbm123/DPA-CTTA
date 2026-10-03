"""Anonymous aggregates and matched distributions; no individual content identities."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import statistics as st
import numpy as np
from ..r10_12h_core.run import read,save
from .methods import STATIC
from .zero_order import KINDS

def write_csv(p,rows,fields):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def public(x):
    if isinstance(x,dict):return {k:public(v) for k,v in x.items() if k not in ('bindings','file','path','values_path','output_root','target_root','source_root','checkout','private_root','remote_root','argv','reason','traceback','group','content','sample_id','group_id','mask_path','image_path','host')}
    if isinstance(x,list):return [public(v) for v in x]
    return x

def cells(rows):
    out=defaultdict(list)
    for r in rows:
        if r['subset']=='remaining_dev':
            for m in r['metrics']:out[r['domain'],m['channel']].append(m)
    return out

def mean(rows):
    c=cells(rows)
    if len(c)!=8:raise ValueError('four domains two channels incomplete')
    return st.mean(st.mean(m['dice'] for m in a) for a in c.values())

def report(c,state):
    root=Path(c['output_root']);out=root/'public';out.mkdir(exist_ok=True)
    ledger=read(root/'RESOURCE_LEDGER.json');save(out/'RESOURCE_LEDGER.json',public(ledger));save(out/'RUN_STATE.json',public(state));save(out/'RUN_RECEIPTS.json',public(dict(attempts=ledger['attempts'])))
    lock=root/'EXPERIMENT_LOCK.json'
    if lock.exists():save(out/'EXPERIMENT_LOCK.json',public(read(lock)))
    else:save(out/'EXPERIMENT_LOCK.json',dict(status='NO_TARGET_LOCK',target_accesses=0,code_sha=c['code_sha'],config_sha256=ledger['attempts'][0]['config_sha256'] if ledger['attempts'] else None))
    for name in ('PROFILE_ADMISSION.json','SCORER_RECEIPT.json','SOURCE_CACHE_RETIRED.json'):
        if (root/name).exists():save(out/name,public(read(root/name)))
    main=[];domain=[];pairs=[];mechanisms=[];na=[];summary=[];groups=defaultdict(list)
    p=root/'score/all-scalars.private.jsonl'
    rows=[json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []
    for r in rows:groups[r['condition'],r['order'],r.get('seed')].append(r)
    normalized={}
    for (arm,order,seed),rs in groups.items():
        m=mean(rs);cels=cells(rs);ref={k:next((values for (a,o,s),values in groups.items() if a==k and o==order),[]) for k in ('C0','DS','G')}
        row=dict(condition=arm,order=order,seed=seed,status='COMPLETE',origin=rs[0]['origin'],visits=len(rs),principal=sum(r['subset']=='remaining_dev' for r in rs),Dice_percent=100*m)
        for k in ref:row['delta_'+k+'_pp']=100*(m-mean(ref[k])) if ref[k] else None
        row['OD_percent']=100*st.mean(st.mean(x['dice'] for x in values) for (d,ch),values in cels.items() if ch=='OD')
        row['OC_percent']=100*st.mean(st.mean(x['dice'] for x in values) for (d,ch),values in cels.items() if ch=='OC')
        main.append(row)
        for (d,ch),values in sorted(cels.items()):
            controls={k:cells(v).get((d,ch),[]) if v else [] for k,v in ref.items()};ds=controls['DS'];v=st.mean(x['dice'] for x in values)
            boundary=[x['assd'] for x in values if x.get('assd') is not None and math.isfinite(x['assd'])]
            domain.append(dict(condition=arm,order=order,seed=seed,domain=d,channel=ch,n=len(values),Dice_percent=100*v,**{'delta_'+k+'_pp':100*(v-st.mean(x['dice'] for x in a)) if a else None for k,a in controls.items()},
                  ASSD_mean=st.mean(boundary) if boundary else None,ASSD_valid_n=len(boundary),ASSD_undefined_n=len(values)-len(boundary),
                  TP=sum(x['intersection'] for x in values),FP=sum(x['pred_pixels']-x['intersection'] for x in values),FN=sum(x['gt_pixels']-x['intersection'] for x in values),
                  correct_to_error=sum(x.get('correct_to_error',0) for x in values) if all('correct_to_error' in x for x in values) else None,
                  error_to_correct=sum(x.get('error_to_correct',0) for x in values) if all('error_to_correct' in x for x in values) else None))
        normalized[arm,order,seed]=rs
    # Z seed averaging occurs within the same order, including per-image paired outcomes.
    compared={}
    for arm in (*STATIC,*KINDS,'G'):
        for order in (0,1):
            candidates=[rs for (a,o,s),rs in groups.items() if a==arm and o==order]
            if not candidates:continue
            if arm in KINDS and len(candidates)!=2:
                na.append(dict(condition=arm,order=order,status='INCOMPLETE_SEED_PAIR',metric='aggregate',reason_code='both registered Z seeds required'));continue
            index={}
            for rs in candidates:
                for r in rs:index.setdefault(r['content'],[]).append(r)
            averaged=[]
            for identity,items in index.items():
                if len(items)!=len(candidates):raise ValueError('seed pairing mismatch')
                averaged.append(dict(items[0],metrics=[dict(m,dice=st.mean(x['metrics'][j]['dice'] for x in items)) for j,m in enumerate(items[0]['metrics'])]))
            compared[arm,order]=averaged
    contrast=[('P','DS'),('S','DS'),('P_VERIFY','P'),('P_VERIFY','P_SIMPLE'),('Z_CORE','Z_PROBE'),('Z_BOUND','Z_CORE'),('D_CONTEXT','D_LOGIT'),('D_VERIFY','D_CONTEXT')]
    for a,b in contrast:
        for o in (0,1):
            ra=compared.get((a,o));rb=compared.get((b,o))
            mechanisms.append(dict(condition=a,control=b,order=o,status='COMPLETE' if ra and rb else 'NA',delta_pp=100*(mean(ra)-mean(rb)) if ra and rb else None))
    for (arm,o),rs in compared.items():
        if arm in ('C0','DS','G'):continue
        ref={k:{r['content']:r for r in compared.get((k,o),[])} for k in ('C0','DS','G')}
        delta=[]
        for r in rs:
            if r['subset']!='remaining_dev':continue
            item=dict(condition=arm,order=o,domain=r['domain'])
            for k,lookup in ref.items():
                base=lookup.get(r['content']);item['delta_'+k+'_pp']=100*(st.mean(m['dice'] for m in r['metrics'])-st.mean(m['dice'] for m in base['metrics'])) if base else None
            delta.append(item)
        delta.sort(key=lambda x:(x['delta_DS_pp'] if x['delta_DS_pp'] is not None else 0,x['domain']))
        for rank,r in enumerate(delta,1):pairs.append(dict(r,anonymous_sorted_rank=rank,worst_10_percent=rank<=math.ceil(.1*len(delta))))
    for arm in (*STATIC,*KINDS,'G'):
        for o in (0,1):
            if (arm,o) not in compared:na.append(dict(condition=arm,order=o,status='NOT_RUN_OR_INCOMPLETE',metric='target',reason_code=state.get('status')))
        a=[compared[arm,o] for o in (0,1) if (arm,o) in compared]
        controls={k:[compared[k,o] for o in (0,1) if (k,o) in compared] for k in ('C0','DS','G')}
        if len(a)==2 and all(len(v)==2 for v in controls.values()):
            changes={k:100*st.mean(mean(x)-mean(y) for x,y in zip(a,v)) for k,v in controls.items()}
            dc=[];c0loss=[]
            for rs,ds in zip(a,controls['DS']):
                ca,cb=cells(rs),cells(ds);dc.extend(100*(st.mean(x['dice'] for x in ca[k])-st.mean(x['dice'] for x in cb[k])) for k in ca)
            for rs,base in zip(a,controls['C0']):
                ca,cb=cells(rs),cells(base);c0loss.extend(100*(st.mean(x['dice'] for x in ca[k])-st.mean(x['dice'] for x in cb[k])) for k in ca)
            worst=min(dc);signal=changes['DS']>0 and all(mean(x)>=mean(y) for x,y in zip(a,controls['DS']))
            summary.append(dict(condition=arm,Dice_percent=100*st.mean(mean(x) for x in a),delta_C0_pp=changes['C0'],delta_DS_pp=changes['DS'],delta_G_pp=changes['G'],worst_domain_channel_DS_pp=worst,
                                worst_domain_channel_C0_pp=min(c0loss),status='DEVELOPMENT_SIGNAL' if signal else 'NO_DEVELOPMENT_GAIN',new_data_confirmation_candidate=bool(changes['C0']>=.5 and changes['DS']>0 and worst>=-2 and min(c0loss)>=-2)))
    save(out/'SUMMARY.json',summary)
    write_csv(out/'MAIN_RESULTS.csv',main,('condition','order','seed','status','origin','visits','principal','Dice_percent','OD_percent','OC_percent','delta_C0_pp','delta_DS_pp','delta_G_pp'))
    write_csv(out/'DOMAIN_CHANNEL_RESULTS.csv',domain,('condition','order','seed','domain','channel','n','Dice_percent','delta_C0_pp','delta_DS_pp','delta_G_pp','ASSD_mean','ASSD_valid_n','ASSD_undefined_n','TP','FP','FN','correct_to_error','error_to_correct'))
    write_csv(out/'PAIRED_RESULTS.csv',pairs,('condition','order','domain','anonymous_sorted_rank','delta_C0_pp','delta_DS_pp','delta_G_pp','worst_10_percent'))
    write_csv(out/'MECHANISM_RESULTS.csv',mechanisms,('condition','control','order','status','delta_pp'))
    for arm in (*STATIC,*KINDS,'G'):na.append(dict(condition=arm,status='NA_MASK_ONLY',metric='target_soft_Dice_Brier_ECE',reason_code='real target probabilities not stored'))
    for a in ledger['attempts']:
        if a['status']!='COMPLETE':na.append(dict(condition=a['phase'],status=a['status'],metric='execution',reason_code=a.get('failure',{}).get('classification')))
    write_csv(out/'FAILURES_AND_NA.csv',na,('condition','order','status','metric','reason_code'))
    # Preserve source hard AND soft, all four Z choices and all D checkpoints.
    srcrows=[]
    for name in ('SOURCE_STATIC','SOURCE_Z','SOURCE_D'):
        if not (root/(name+'.json')).exists():continue
        doc=read(root/(name+'.json'));raw=doc.get('final_rows',doc.get('rows',[]));bins=defaultdict(list)
        for r in raw:bins[r['condition'],r.get('mu'),r.get('eta'),r.get('step'),r.get('mode')].append(r)
        for (arm,mu,eta,step,mode),rs in bins.items():
            srcrows.append(dict(condition=arm,mu=mu,eta=eta,step=step,mode=mode,n=len(rs),**{k:st.mean(r[k] for r in rs) if all(k in r for r in rs) else None for k in ('hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice')}))
        if name=='SOURCE_D':
            bins=defaultdict(list)
            for r in doc['rows']:bins[r['condition'],r['step'],r['mode']].append(r)
            for (arm,step,mode),rs in bins.items():srcrows.append(dict(condition=arm,step=step,mode=mode,n=len(rs),**{k:st.mean(r[k] for r in rs) for k in ('hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice')}))
            save(out/'SOURCE_SELECTION.json',public(dict(D=doc['selected'],Z=read(root/'SOURCE_Z.json').get('selected'))))
    write_csv(out/'SOURCE_RESULTS.csv',srcrows,('condition','mu','eta','step','mode','n','hard_OD','hard_OC','hard_Dice','soft_OD','soft_OC','soft_Dice'))
    lines=['# R16 four-direction finite development experiment',f"Status: {state['status']}; execution SHA {c['code_sha']}.",
           'All source choice is completed before the target lock. Target masks are scored by a separate CPU process after the prediction matrix is terminal.',
           'The cohort is previously exposed DEVELOPMENT. Orders share images; Z seeds describe algorithm randomness, not patient replication. No next experiment is authorized.',
           f"GPU-worker {ledger['gpu_seconds']:.3f} s; wall from original T0 {ledger['wall_seconds']:.3f} s. Failed attempts remain charged.",
           '| Condition | Dice % | DS delta pp | GraTa delta pp | Worst domain/channel vs DS pp | Interpretation |','|---|---:|---:|---:|---:|---|']
    for r in summary:lines.append(f"| {r['condition']} | {r['Dice_percent']:.6f} | {r['delta_DS_pp']:+.6f} | {r['delta_G_pp']:+.6f} | {r['worst_domain_channel_DS_pp']:+.6f} | {r['status']} |")
    if not summary:lines.append('Target results NOT_RUN or unavailable; failure/status package is the real outcome. No target values are inferred from historical means.')
    lines+=['','Full condition/order/seed/domain/channel tables, source hard and soft, all source choices, paired adverse tails, undefined boundary denominators, failures and NA accompany this report.',
            'C0/H025 references are exact historical scalar records checked against the new shared static hard pixel counts. DS hard equals H025 by construction. Static order 1 is REORDERED_REUSE. GraTa uses an intact historical stateful trajectory with its own seed; not a stitched trajectory.',
            'Target soft Dice, Brier and ECE are NA because real probabilities were not retained. Hard-mask values are not probability calibration evidence.',
            'These are Inspired/Adapted modules, not reproductions of CLIP TTA, TopoOT, EVA-0 or ORCA. Added source supervision for D and target parameter updates for Z are distinct costs.',
            'Publication pending local proxy push and anonymous verification. Private images, labels, predictions, features, content identifiers, weights and credentials are excluded.']
    (out/'REPORT.md').write_text('\n\n'.join(lines)+'\n')
    return out
