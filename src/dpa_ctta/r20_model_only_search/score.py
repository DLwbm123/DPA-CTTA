"""Independent CPU metrics; stage SEARCH barriers and single final review release."""
import json,statistics as st
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from ..r10_12h_core.run import read,save,sha
from ..r8_ba.journal import verify_online_complete,_digest
from ..r8_ba.streams import rows_sha
from ..r7_target_screen.runner import TargetReader
from ..r16_evidence_correction.score import metrics
from ..r16_evidence_correction.structure import containment

WIDTH=65536;SOFT_WIDTH=2097152

def score_job(c,guard,request):
    root=Path(c['output_root']);job=request['job'];stage=request['stage'];final=request.get('final',False);barrier=read(root/'stages'/f'{stage}.barrier.json')
    if barrier['status']!='ALL_WORKERS_RETIRED' or any(read(p).get('active') for p in (root/'processes').glob('online_*.json')):raise ValueError('online worker retirement barrier')
    if final:
        lock=read(root/'FROZEN.json');release=read(root/'REVIEW_RELEASE.json')
        if release['frozen_sha256']!=sha(lock) or release['all_formal_workers_retired'] is not True:raise ValueError('review release identity')
    rows=read(root/'scorer'/f'{"SCREEN" if job["stream"]=="screen" else "FULL"}_o{job["order"]}.json');assignment=read(root/'scorer/SPLIT.private.json');online=read(root/'private'/f'{"SCREEN" if job["stream"]=="screen" else "ONLINE"}_o{job["order"]}.json')
    dest=root/'target'/job['id'];on=read(dest/'online_complete.json');verify_online_complete(dest,job['id'],on['identity']['context_sha256'],rows_sha(online),len(rows),WIDTH)
    outdir=root/'scores'/('final' if final else stage);outdir.mkdir(parents=True,exist_ok=True);out=outdir/(job['id']+'.private.jsonl');reader=TargetReader(c['target_root'],256*1024**2,'mask');count=0;soft=None
    if job.get('soft'):
        receipt=read(dest/'soft_complete.json')
        if receipt['bytes']!=len(rows)*SOFT_WIDTH or (dest/'probabilities.f32').stat().st_size!=receipt['bytes']:raise ValueError('probability coverage')
        soft=(dest/'probabilities.f32').open('rb')
    try:
        with (dest/'predictions.bits').open('rb') as bits,(dest/'visits.jsonl').open() as traces,out.open('x') as target:
            for i,m in enumerate(rows):
                guard();bits.seek(i*WIDTH);raw=bits.read(WIDTH);trace=json.loads(next(traces));role=assignment.get(m['image_sha256'],'CONTEXT')
                if not final and role!='SEARCH':continue # Label path never opened for sealed review/context during selection.
                label=reader.read(m);count+=1;mask=np.unpackbits(np.frombuffer(raw,dtype=np.uint8)).reshape(2,512,512).astype(bool);values=metrics(mask,label)
                if soft:
                    soft.seek(i*SOFT_WIDTH);p=np.frombuffer(soft.read(SOFT_WIDTH),dtype='<f4').reshape(2,512,512);gt=label.numpy()[0]
                    for k,v in enumerate(values):v.update(soft_dice=float((2*np.sum(p[k]*gt[k],dtype=np.float64)+1e-6)/(np.sum(p[k],dtype=np.float64)+np.sum(gt[k],dtype=np.float64)+1e-6)),Brier=float(((p[k].astype(np.float64)-gt[k])**2).mean()))
                row=dict(job=job['id'],condition=job['candidate']['id'],order=job['order'],seed=job['seed'],visit=i+1,content=m['image_sha256'],domain=m['domain'],subset=m['subset'],role=role,metrics=values,seconds=trace['seconds'],diagnostics=trace['diagnostics'],empty_foreground=[not x.any() for x in mask],containment_violations=containment(mask))
                target.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
        if count!=(len(rows) if final else sum(assignment.get(m['image_sha256'])=='SEARCH' for m in rows)):raise ValueError('score denominator')
    finally:
        if soft:soft.close()
        reader.after_check()
    result=dict(status='SCORED',job=job['id'],rows=count,independent_CPU=True,review_labels_read=final,scalar_sha256=_digest(out),source_reads=0);save(outdir/(job['id']+'.complete.json'),result);return result

def load_rows(root,stage,jobs,final=False):
    directory=Path(root)/'scores'/('final' if final else stage);rows=[]
    for j in jobs:
        p=directory/(j['id']+'.private.jsonl')
        if p.exists():rows.extend(json.loads(x) for x in p.read_text().splitlines())
    return rows

def cells(rows):
    groups=defaultdict(list)
    for r in rows:
        for m in r['metrics']:groups[r['domain'],m['channel']].append(m['dice']*100)
    return {k:st.mean(v) for k,v in groups.items()}

def summary(rows):
    cs=cells(rows)
    return dict(macro=st.mean(cs.values()),imageweighted=st.mean(st.mean(m['dice'] for m in r['metrics'])*100 for r in rows),cells=cs)

def rank_results(rows,jobs,attempts):
    groups=defaultdict(list)
    for r in rows:
        if r['role']=='SEARCH':groups[r['condition'],r['order']].append(r)
    baseline={o:summary(groups['G',o]) for o in (0,1)};out=[]
    for candidate in sorted({r['condition'] for r in rows}):
        if any((candidate,o) not in groups for o in (0,1)):continue
        ss={o:summary(groups[candidate,o]) for o in (0,1)};delta=[ss[o]['macro']-baseline[o]['macro'] for o in (0,1)];iw=st.mean(ss[o]['imageweighted']-baseline[o]['imageweighted'] for o in (0,1));worst=min(ss[o]['cells'][k]-baseline[o]['cells'][k] for o in (0,1) for k in baseline[o]['cells']);jids={j['id'] for j in jobs if j['candidate']['id']==candidate};seconds=sum(a['cost']['gpu_seconds'] for a in attempts if a['phase'].removeprefix('online_') in jids);n=sum(len(groups[candidate,o]) for o in (0,1));actual_visits=sum(384 if j['stream']=='screen' else 1951 for j in jobs if j['candidate']['id']==candidate)
        out.append(dict(id=candidate,macro_Dice_percent=st.mean(ss[o]['macro'] for o in (0,1)),order_delta_pp=delta,delta_pp=st.mean(delta),imageweighted_delta_pp=iw,worst_cell_pp=worst,nonrisk=all(d>0 for d in delta) and iw>=0 and worst>=-2,seconds_per_image=seconds/max(1,actual_visits),scored_observations=n))
    return out
