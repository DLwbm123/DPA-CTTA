"""CPU-only aggregate review of sealed R16 receipts. Never opens images or labels.

Usage: call aggregate(Path(private_run_root)) after SCORER_RECEIPT is COMPLETE.
The output contains only aggregates; supports, identities, features and weights stay private.
"""
import csv
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path


def read(p):
    return json.loads(p.read_text())


def save(p, x):
    p.write_text(json.dumps(x, indent=2, sort_keys=True, allow_nan=False)+'\n')


def write(p, rows):
    if not rows:
        return
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def latency(values):
    a = sorted(values)
    return dict(n=len(a), mean_seconds=st.mean(a), median_seconds=st.median(a),
                p95_seconds=a[math.ceil(.95*len(a))-1], max_seconds=max(a))


def aggregate(root):
    root = Path(root); out = root/'public'
    state = read(root/'RUN_STATE.json'); receipt = read(root/'SCORER_RECEIPT.json')
    assert state['status']=='COMPLETE' and receipt['status']=='COMPLETE' and not receipt['failed']
    assert receipt['scalar_rows']==66334 and len(receipt['completed'])==13
    assert all(r['visits']==1951 and r['principal']==1695 for r in receipt['completed'])
    # Streaming the existing journal is observation, not another model run or scoring pass.
    counts=Counter(); coverage=[[],[],[]]; cosines=[[],[],[]]; times=[]; verifies={}
    for line in (root/'target/STATIC_o0/visits.jsonl').open():
        row=json.loads(line); d=row['diagnostics']; times.append(row['seconds']); p=d['prototype']
        counts['visits']+=1; counts['fallback_'+str(p['fallback'])]+=1
        for i,x in enumerate(p['seed_coverage']): coverage[i].append(x)
        if 'prototype_cosines' in p:
            for i,(a,b) in enumerate(((0,1),(0,2),(1,2))): cosines[i].append(p['prototype_cosines'][a][b])
        for k in ('P_SIMPLE','P_VERIFY','S','D_VERIFY'):
            a=d[k]; v=verifies.setdefault(k,dict(candidates=0,accepted=0,rejections=Counter(),edited_pixels=0,max_spatial_edit_fraction=0.0,zero_accept_visits=0))
            v['candidates']+=a['candidates']; v['accepted']+=a['accepted']; v['rejections'].update(a['rejections']); v['edited_pixels']+=a['edited_pixels']
            v['zero_accept_visits']+=a['accepted']==0
            v['max_spatial_edit_fraction']=max(v['max_spatial_edit_fraction'],a['edited_pixels']/(512*512))
    assert counts['visits']==1951 and all(v['max_spatial_edit_fraction']<=.02 for v in verifies.values())
    save(out/'PROTOTYPE_CANDIDATE_DIAGNOSTICS.json',dict(scope='1951 unique online visits including 256 warmup; order1 reused',prototype=dict(counts=counts,
         class_order=['background','disc_without_cup','cup'],mean_seed_coverage=[st.mean(a) for a in coverage],cosine_pairs=['background_disc','background_cup','disc_cup'],
         mean_prototype_cosines=[st.mean(a) if a else None for a in cosines],valid_prototype_visits=len(cosines[0])),selectors=verifies,shared_static_latency=latency(times)))
    z=[]
    for p in sorted((root/'target').glob('Z_*/visits.jsonl')):
        rows=[json.loads(line) for line in p.open()]; assert len(rows)==1951
        z.append(dict(job=p.parent.name,updates=sum(r['updates'] for r in rows),update_skipped=sum(r['update_skipped'] for r in rows),
             skipped_fraction=sum(r['update_skipped'] for r in rows)/len(rows),final_adapter_rms=rows[-1]['adapter_rms'],max_adapter_rms=max(r['adapter_rms'] for r in rows),
             mean_gradient_norm=st.mean(r['gradient_norm'] for r in rows),max_gradient_norm=max(r['gradient_norm'] for r in rows),
             mean_absolute_loss_difference=st.mean(abs(r['loss_plus']-r['loss_minus']) for r in rows),**latency([r['seconds'] for r in rows])))
    write(out/'ADAPTER_LATENCY_DIAGNOSTICS.csv',z)
    bins=defaultdict(list); primary=0
    for line in (root/'score/all-scalars.private.jsonl').open():
        r=json.loads(line)
        if r['subset']=='remaining_dev': bins[r['condition'],r['order'],r.get('seed')].append(r); primary+=1
    assert primary==57630 and len(bins)==34
    structure=[]
    for (k,o,s),rows in sorted(bins.items(),key=lambda item:str(item[0])):
        for c,ch in enumerate(('OD','OC')):
            available=all('fragments_holes' in r for r in rows)
            structure.append(dict(condition=k,order=o,seed=s,channel=ch,n=len(rows),
                fragments_total=sum(r['fragments_holes'][c][0] for r in rows) if available else None,
                holes_total=sum(r['fragments_holes'][c][1] for r in rows) if available else None,
                containment_pixels_total=sum(r['containment_violations'] for r in rows) if available else None,
                containment_images=sum(r['containment_violations']>0 for r in rows) if available else None,
                corrected_pixels=sum(r['metrics'][c]['error_to_correct'] for r in rows) if all('error_to_correct' in r['metrics'][c] for r in rows) else None,
                harmed_pixels=sum(r['metrics'][c]['correct_to_error'] for r in rows) if all('correct_to_error' in r['metrics'][c] for r in rows) else None))
    write(out/'TARGET_STRUCTURE_RESULTS.csv',structure)
    source=[]; source_means={}; shuffled=[]
    for name in ('SOURCE_STATIC','SOURCE_D'):
        doc=read(root/(name+'.json')); rows=doc.get('final_rows',doc['rows']); ds={(r['episode'],r['visit']):r for r in rows if r['condition']=='DS'}
        groups=defaultdict(list)
        for r in rows: groups[r['condition']].append(r)
        for k,rs in groups.items():
            source_means[k]=dict(n=len(rs),hard_percent=100*st.mean(r['hard_Dice'] for r in rs),soft_percent=100*st.mean(r['soft_Dice'] for r in rs))
            if k=='P_SHUFFLED_SOURCE_DIAGNOSTIC':
                lookup={(r['episode'],r['visit']):r for r in groups['P']}; matches=[lookup[r['episode'],r['visit']] for r in rs]
                shuffled=dict(n=len(rs),shuffled_hard_percent=source_means[k]['hard_percent'],matched_P_hard_percent=100*st.mean(r['hard_Dice'] for r in matches),
                     shuffled_minus_matched_P_pp=100*st.mean(r['hard_Dice']-m['hard_Dice'] for r,m in zip(rs,matches)))
                continue
            for mode in sorted({r['mode'] for r in rs}):
                a=[r for r in rs if r['mode']==mode]; b=[ds[r['episode'],r['visit']] for r in a]
                source.append(dict(condition=k,mode=mode,n=len(a),hard_minus_DS_pp=100*st.mean(r['hard_Dice']-v['hard_Dice'] for r,v in zip(a,b)),
                    corrected_pixels=sum(r['error_to_correct'] for r in a),harmed_pixels=sum(r['correct_to_error'] for r in a),edited_channel_pixels=sum(r['edited'] for r in a),
                    containment_pixels=sum(r['containment'] for r in a),fragment_change=sum(sum(x[0] for x in r['topology'])-sum(x[0] for x in v['topology']) for r,v in zip(a,b)),
                    hole_change=sum(sum(x[1] for x in r['topology'])-sum(x[1] for x in v['topology']) for r,v in zip(a,b))))
    write(out/'SOURCE_ERROR_ANALYSIS.csv',source)
    save(out/'SOURCE_MEANS.json',dict(all_modes_equal_512_visits=source_means,shuffled_32_source=shuffled,
         boundary_displacement='NA: source per-pixel boundary distances not retained; hard edit counts do not measure directed boundary offsets'))
    # Z soft_Dice is the documented channel average of already recorded soft_OD/OC.
    p=out/'SOURCE_RESULTS.csv'; rows=list(csv.DictReader(p.open()))
    for r in rows:
        if not r['soft_Dice'] and r['soft_OD'] and r['soft_OC']: r['soft_Dice']=str((float(r['soft_OD'])+float(r['soft_OC']))/2)
    write(p,rows)
    ledger=read(root/'RESOURCE_LEDGER.json'); charges={}
    for a in ledger['attempts']:
        key=f"{a['phase']}.{a['attempt']}"; assert key not in charges; charges[key]=a['cost'].get('gpu_seconds',0)
    assert abs(sum(charges.values())-ledger['gpu_seconds'])<1e-8
    online=[a for a in ledger['attempts'] if a['phase'].startswith('online_')]
    assert len(online)==13 and all(a['status']=='COMPLETE' and a['cost']['model_forwards']==3902 and all(a['cost'][k]==0 for k in ('backward_calls','optimizer_steps','vjp_calls')) for a in online)
    save(out/'FINAL_AUDIT.json',dict(status='PASS',online_physical_jobs=13,logical_streams=34,scalar_rows=66334,principal_rows=primary,
         per_stream_visits=1951,per_stream_principal=1695,source_lock_sha256=receipt['source_lock_sha256'],
         target_BP_optimizer_VJP=0,shared_C0_H025_hard_pixel_parity='PASS in independent scorer',DS_H025_hard_identity='PASS all static visits',
         static_order1='CONTENT_ROLE_VERIFIED_REORDERED_REUSE',reference_G='MATCHED_INTACT_HISTORICAL_STATEFUL',GPU_seconds=ledger['gpu_seconds'],
         source_qualification='sealed v2 native32 reused with exact CPU v3 equivalence; new v3 mechanical8 only',recovery_attempts=0,
         target_score_embargo='all13 registered online jobs terminal before independent CPU label reads',
         source_soft_Dice_Z='derived as arithmetic mean of sealed channel soft Dice; no new inference',
         source_boundary_displacement='NA_NOT_RECORDED',standalone_latency='NA_NOT_SEPARATELY_MEASURED',peak_device_memory='NA_NOT_INSTRUMENTED'))
    return dict(status='COMPLETE',primary=primary,charges=charges)
