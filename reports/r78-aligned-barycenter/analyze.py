"""Derive anonymous aggregate contrasts only; no images, labels or model calls."""
import csv,json,time
from pathlib import Path
from statistics import mean

root=Path(__file__).resolve().parent;started=time.time()
def load(name):
    with (root/name).open() as f:return list(csv.DictReader(f))
def write(name,rows):
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
results=json.loads((root/'RESULTS.json').read_text());cells=load('ALL_CELLS.csv');errors=load('DOMAIN_CHANNEL_ERRORS.csv')
assert results['formal_updates']==3456 and len(cells)==384 and len(errors)==48
contrasts={s['contrast']:s for s in results['contrasts']};domain=[];worst=[];violations=[];trajectories=[]
for name,s in contrasts.items():
    relevant=[r for r in cells if r['contrast']==name]
    for d in ('D0','D1','D2','D3'):
        for ch in ('OD','OC'):
            group=[r for r in relevant if r['domain']==d and r['channel']==ch]
            value=mean(float(r['delta_pp']) for r in group)
            domain.append(dict(contrast=name,domain=d,channel=ch,delta_pp=value))
            if name=='BARY/O-NATIVE/O' and value<-.5:
                violations.append(dict(gate='domain_channel_mean',domain=d,channel=ch,order='ALL',delta_pp=value,threshold_pp=-.5))
            for order in (0,1):
                values=[float(r['delta_pp']) for r in group if int(r['order'])==order]
                assert len(values)==3
                if name=='BARY/O-NATIVE/O' and mean(values)<-2:
                    violations.append(dict(gate='domain_channel_order_seedmean',domain=d,channel=ch,order=order,delta_pp=mean(values),threshold_pp=-2))
    r=min(relevant,key=lambda r:float(r['delta_pp']))
    worst.append(dict(contrast=name,kind='individual',domain=r['domain'],channel=r['channel'],order=r['order'],seed=r['seed'],delta_pp=float(r['delta_pp'])))
    grouped=[]
    for d in ('D0','D1','D2','D3'):
        for ch in ('OD','OC'):
            for order in (0,1):
                values=[float(r['delta_pp']) for r in relevant if r['domain']==d and r['channel']==ch and int(r['order'])==order]
                grouped.append(dict(contrast=name,kind='seed_mean',domain=d,channel=ch,order=order,seed='MEAN',delta_pp=mean(values)))
    worst.append(min(grouped,key=lambda r:r['delta_pp']))
    for seed in (20260907,17011,29009):
        for order in (0,1):
            v={ch:mean(float(r['delta_pp']) for r in relevant if int(r['seed'])==seed and int(r['order'])==order and r['channel']==ch) for ch in ('OD','OC')}
            trajectories.append(dict(contrast=name,seed=seed,order=order,OD_delta_pp=v['OD'],OC_delta_pp=v['OC'],macro_delta_pp=mean(v.values())))
    for ch in ('OD','OC'):
        assert abs(mean(r['delta_pp'] for r in domain if r['contrast']==name and r['channel']==ch)-s[ch+'_delta_pp'])<1e-9
write('DOMAIN_CHANNEL_DELTAS.csv',domain);write('WORST_CELLS.csv',worst);write('HARM_GATE_VIOLATIONS.csv',violations);write('TRAJECTORY_DELTAS.csv',trajectories)
summary=[]
for arm in ('NATIVE','VERTEX','BARY'):
    rows=[r for r in load('MECHANISM.csv') if r['arm']==arm]
    summary.append(dict(arm=arm,mean_trajectory_seconds=mean(float(r['seconds']) for r in rows),
        mean_arrival_seconds=mean(float(r['seconds']) for r in rows)/192,
        mean_BN_gradient=mean(float(r['mean_BN_gradient']) for r in rows),mean_BN_update=mean(float(r['mean_BN_update']) for r in rows)))
write('MECHANISM_SUMMARY.csv',summary)
temporal=load('TEMPORAL_AGGREGATES.csv');lookup={(r['cell'],r['seed'],r['order'],r['half'],r['domain']):r for r in temporal}
rows=[]
for a,b in (('VERTEX/O','NATIVE/O'),('BARY/O','NATIVE/O'),('BARY/O','VERTEX/O'),('NATIVE/A','NATIVE/O'),('VERTEX/A','VERTEX/O'),('BARY/A','BARY/O')):
    for seed in (20260907,17011,29009):
        for order in (0,1):
            for half in (0,1):
                for d in ('D0','D1','D2','D3'):
                    x=lookup[a,str(seed),str(order),str(half),d];y=lookup[b,str(seed),str(order),str(half),d]
                    assert x['n']==y['n']
                    if int(x['n'])==0:continue
                    for ch in ('OD','OC'):
                        rows.append(dict(contrast=a+'-'+b,seed=seed,order=order,half=half,domain=d,channel=ch,n=int(x['n']),delta_pp=float(x[ch])-float(y[ch])))
write('TEMPORAL_DELTAS.csv',rows)
for ch in ('macro','OC','OD'):
    key=ch+'_delta_pp'
    assert abs(contrasts['BARY/A-NATIVE/O'][key]-contrasts['BARY/O-NATIVE/O'][key]-contrasts['BARY/A-BARY/O'][key])<1e-9
receipt=dict(status='PASS',analysis_seconds=time.time()-started,inputs='anonymous aggregate CSV/JSON only',
    new_model_calls=0,new_image_reads=0,new_label_reads=0,formal_trajectories=18,formal_updates=3456,predictions=6912,
    arithmetic_consistency=True,missingness='empty temporal domain cells retained as n=0 in original table; excluded from paired temporal deltas')
(root/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(violations=violations,worst=worst,mechanism=summary),indent=2))
