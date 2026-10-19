"""Aggregate existing anonymous result tables only, without model/data calls."""
import csv,json,time
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parent;start=time.time()
def load(name):
    with (root/name).open() as f:return list(csv.DictReader(f))
def write(name,rows):
    with (root/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
result=json.loads((root/'RESULTS.json').read_text());cells=load('ALL_CELLS.csv');errors=load('ERROR_COMPONENT_CELLS.csv')
assert result['physical_updates']==8640 and result['prediction_units']==13824
assert all(len([r for r in cells if r['contrast']==s['contrast']])==48 for s in result['contrasts'])
domain=[]
for name in dict.fromkeys(r['contrast'] for r in cells):
    for d in ('D0','D1','D2','D3'):
        for ch in ('OD','OC'):
            rows=[r for r in cells if r['contrast']==name and r['domain']==d and r['channel']==ch]
            values=[float(r['delta_pp']) for r in rows]
            domain.append(dict(contrast=name,domain=d,channel=ch,mean_delta_pp=mean(values),
                min_anchor_delta_pp=min(values),max_anchor_delta_pp=max(values),positive_anchors=sum(v>0 for v in values),n_anchors=len(values)))
write('DOMAIN_DECOMPOSITION.csv',domain)
index={(r['contrast'],r['domain'],r['channel']):r for r in domain}
focal=[]
for a,d,ch in (('VERTEX-NATIVE','D1','OC'),('VERTEX-NATIVE','D1','OD'),('BARY-NATIVE','D1','OC'),
    ('BARY-NATIVE','D0','OD'),('BARY-NATIVE','D2','OD'),('BARY-NATIVE','D3','OD')):
    v={t:index[a+'/'+t,d,ch]['mean_delta_pp'] for t in ('EPISODIC','HISTORY','INTERACTION','CURRENT','TOTAL')}
    assert abs(v['EPISODIC']+v['HISTORY']+v['INTERACTION']-v['TOTAL'])<1e-9
    assert abs(v['CURRENT']+v['HISTORY']-v['TOTAL'])<1e-9
    focal.append(dict(contrast=a,domain=d,channel=ch,**v))
write('FOCAL_MECHANISM.csv',focal)
error_rows=[]
for name in dict.fromkeys(r['contrast'] for r in errors):
    for d in ('D0','D1','D2','D3'):
        for ch in ('OD','OC'):
            for metric in ('FP','FN','FOREGROUND'):
                values=[float(r['delta_pixel_pp']) for r in errors if r['contrast']==name and r['domain']==d and r['channel']==ch and r['metric']==metric]
                assert len(values)==6
                error_rows.append(dict(contrast=name,domain=d,channel=ch,metric=metric,mean_delta_pixel_pp=mean(values)))
write('ERROR_DOMAIN_DECOMPOSITION.csv',error_rows)
scores=load('DOMAIN_CHANNEL_ERRORS.csv');source=[]
for a in ('NATIVE','VERTEX','BARY'):
    for d in ('D0','D1','D2','D3'):
        for ch in ('OD','OC'):
            x={r['output']:r for r in scores if r['arm']==a and r['domain']==d and r['channel']==ch}
            source.append(dict(arm=a,domain=d,channel=ch,
                initial_P0=float(x['P0']['Dice_percent']),reset_R=float(x['R']['Dice_percent']),
                reset_absolute_delta_pp=float(x['R']['Dice_percent'])-float(x['P0']['Dice_percent']),
                incoming_PRE=float(x['PRE']['Dice_percent']),history_absolute_delta_pp=float(x['PRE']['Dice_percent'])-float(x['P0']['Dice_percent']),
                committed_K=float(x['K']['Dice_percent']),total_absolute_delta_pp=float(x['K']['Dice_percent'])-float(x['P0']['Dice_percent'])))
write('ABSOLUTE_REFERENCE_EFFECTS.csv',source)
for r in source:
    key=r['arm']+'/TOTAL',r['domain'],r['channel']
    assert abs(index[key]['mean_delta_pp']-r['total_absolute_delta_pp'])<1e-9
receipt=dict(status='PASS',analysis_seconds=time.time()-start,inputs='existing anonymous aggregate CSV and JSON only',
    new_model_calls=0,new_image_reads=0,new_label_reads=0,identities_verified=True,
    physical_updates=8640,qualification_updates=29,independent_continuous_trajectories=18,
    domain_intervals='not computed; means and six anchor ranges retained',
    reset_anchors='three seed controls reused across two orders, not six independent resets')
(root/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'focal':focal,'table_counts':dict(cells=len(cells),error_cells=len(errors),domain=len(domain))},indent=2))
