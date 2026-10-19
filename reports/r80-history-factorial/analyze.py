"""Zero-model-call checks and aggregation of existing anonymous terminal tables."""
import csv,json,time
from collections import defaultdict
from pathlib import Path
from statistics import mean
root=Path(__file__).resolve().parent;start=time.time()
def load(name):
 with (root/name).open() as f:return list(csv.DictReader(f))
def write(name,rows):
 with (root/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
r=json.loads((root/'RESULTS.json').read_text());cells=load('ALL_CELLS.csv');errors=load('ERROR_COMPONENT_CELLS.csv')
assert r['new_physical_updates']==10368 and r['new_prediction_units']==20736
by=defaultdict(list)
for v in cells:by[v['contrast'],v['domain'],v['channel']].append(v)
domain=[]
for (name,d,ch),vs in by.items():
 assert len(vs)==6
 worst=min(vs,key=lambda x:float(x['delta_pp']));v=[float(x['delta_pp']) for x in vs]
 domain.append(dict(contrast=name,domain=d,channel=ch,mean_delta_pp=mean(v),min_anchor_delta_pp=min(v),max_anchor_delta_pp=max(v),
  positive_anchors=sum(x>0 for x in v),n_anchors=6,worst_seed=worst['seed'],worst_order=worst['order']))
write('DOMAIN_DECOMPOSITION.csv',domain);idx={(v['contrast'],v['domain'],v['channel']):v['mean_delta_pp'] for v in domain}
parts=('BASE_BN','ADAM_AT_Q0','FEEDBACK_AT_A0','INTERACTION','ORIGINAL','ADAM_AT_Q1','FEEDBACK_AT_A1')
focal=[]
for a in ('VERTEX-NATIVE','BARY-NATIVE','BARY-VERTEX'):
 for d,ch in (('D1','OC'),('D1','OD'),('D0','OD'),('D2','OD'),('D3','OD')):
  for phase in ('HISTORY','CURRENT','TOTAL'):
   v={p:idx[a+'/'+phase+'/'+p,d,ch] for p in parts}
   assert abs(sum(v[p] for p in parts[:4])-v['ORIGINAL'])<1e-9
   focal.append(dict(contrast=a,domain=d,channel=ch,phase=phase,**v))
write('FOCAL_MECHANISM.csv',focal)
for (name,d,ch),v in idx.items():
 if '/HISTORY/' in name:
  assert abs(v+idx[name.replace('/HISTORY/','/CURRENT/'),d,ch]-idx[name.replace('/HISTORY/','/TOTAL/'),d,ch])<1e-9
byerr=defaultdict(list)
for v in errors:byerr[v['contrast'],v['domain'],v['channel'],v['metric']].append(float(v['delta_pixel_pp']))
er=[]
for (n,d,ch,m),vs in byerr.items():
 assert len(vs)==6
 er.append(dict(contrast=n,domain=d,channel=ch,metric=m,mean_delta_pixel_pp=mean(vs)))
write('ERROR_DOMAIN_DECOMPOSITION.csv',er)
receipt=dict(status='PASS',analysis_seconds=time.time()-start,inputs='existing anonymous public CSV and JSON only',new_model_calls=0,
 new_image_reads=0,new_label_reads=0,factorial_and_H_C_T_mean_identities_verified=True,new_physical_updates=10368,qualification_updates=30,
 new_trajectories=54,reused_11_trajectories=18,domain_intervals='not computed; six coupled anchor ranges are descriptive')
(root/'ANALYSIS_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(dict(focal_history=[v for v in focal if v['phase']=='HISTORY'],counts=dict(cells=len(cells),negative_cells=len(load('NEGATIVE_CELLS.csv')),error_cells=len(errors),contrasts=len(r['contrasts']))),indent=2))
