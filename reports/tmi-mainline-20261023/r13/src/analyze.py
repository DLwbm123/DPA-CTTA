"""Aggregate private rows by content; public outputs contain no content identifiers."""
import json,sys,statistics
from pathlib import Path
from collections import defaultdict
R=Path(sys.argv[1]);mean=statistics.mean
f=json.loads((R/'functional.private.json').read_text());c=json.loads((R/'credit.private.json').read_text());m=json.loads((R/'metrics.private.json').read_text())
assert len(f)==1440 and len(c)==320 and len(m)==2880
out={'scope':'Repeated source-development content only; not independent patient or target validation','functional':[],'credit':[],'metrics':[],'counts':json.loads((R/'FINISHED.json').read_text())}
def cm(rows,fun):
 d=defaultdict(list)
 for r in rows:
  v=fun(r)
  if v is not None:d[r['content']].append(v)
 return mean(mean(v) for v in d.values()) if d else None
for seed in [10223,10224]:
 for subset in ['all','clean','shift']:
  filt=lambda r:r['seed']==seed and (subset=='all' or (r['shift']=='clean')==(subset=='clean'))
  for mode in ['GLOBAL','SHUFFLE']:
   rows=[r for r in f if filt(r) and r['mode']==mode]
   row={'seed':seed,'subset':subset,'mode':mode,'full_minus_readout_pp':cm(rows,lambda r:mean(r['full_minus_readout'])),'od_delta_pp':cm(rows,lambda r:r['full_minus_readout'][0]),'oc_delta_pp':cm(rows,lambda r:r['full_minus_readout'][1]),'max_expert_mean_shift':max(r['expert_mean_shift'] for r in rows)}
   if mode=='SHUFFLE':row['permutation_mean_delta_pp']=[cm([r for r in rows if r['permutation']==p],lambda r:mean(r['full_minus_readout'])) for p in range(8)]
   out['functional'].append(row)
  for head in ['OD','OC']:
   rows=[r for r in c if filt(r) and r['head']==head];groups=defaultdict(list)
   for r in rows:groups[r['content']].append(r)
   def rate(a,b):
    vals=[sum(r[a] for r in rs)/sum(r[b] for r in rs) for rs in groups.values() if sum(r[b] for r in rs)>0]
    return {'mean_content_rate':mean(vals) if vals else None,'eligible_contents':len(vals),'pooled_action_rate_descriptive':sum(r[a] for r in rows)/max(sum(r[b] for r in rows),1)}
   out['credit'].append({'seed':seed,'subset':subset,'head':head,'positive_to_negative':rate('positive_to_negative','old_positive'),'positive_to_negative_changed':rate('positive_to_negative_changed','old_positive_changed'),'mean_regret_pp':cm(rows,lambda r:r['mean_regret_pp']),'topset_intersection_fraction':cm(rows,lambda r:r['topset_intersection']/64),'old_tied_fraction':cm(rows,lambda r:r['old_tied_regions']/64),'uninformative_fraction':cm(rows,lambda r:r['uninformative_regions']/64),'unchanged_action_fraction':cm(rows,lambda r:r['unchanged']/256),'changed_near_zero_fraction':cm(rows,lambda r:r['changed_near_zero']/256),'old_positive_count':sum(r['old_positive'] for r in rows),'positive_to_negative_count':sum(r['positive_to_negative'] for r in rows)})
  for resolution in [256,512]:
   for arm in sorted(set(r['arm'] for r in m)):
    rows=[r for r in m if filt(r) and r['resolution']==resolution and r['arm']==arm]
    v={'seed':seed,'subset':subset,'resolution':resolution,'arm':arm}
    for key in ['dice','od','oc','vcdr_error','empty_heads','outside_pixels','outside_occurs','outside_fraction']:v[key]=cm(rows,lambda r:r[key])
    for h in range(2):
     for key in ['hd95','assd']:v[f'{h}_{key}_finite_mean']=cm(rows,lambda r:r['boundary'][h][key])
     v[f'{h}_one_empty_count']=sum(r['boundary'][h]['empty_case']=='one' for r in rows);v[f'{h}_finite_count']=sum(r['boundary'][h]['hd95'] is not None for r in rows)
    out['metrics'].append(v)
(R/'REPORT.json').write_text(json.dumps(out,indent=2)+'\n')
for key in ['functional','credit']:
 print(key,json.dumps([r for r in out[key] if r['subset']=='all'],indent=2))
print('METRICS',json.dumps([r for r in out['metrics'] if r['subset']=='all' and r['arm'] in ['FULL','CONTROL','DIRECT']],indent=2))
# Compare only matching finite boundary observations; retain failure coverage.
paired=[]
for seed in [10223,10224]:
 for resolution in [256,512]:
  d={(v['content'],v['shift'],v['arm']):v for v in m if v['seed']==seed and v['resolution']==resolution}
  for h in [0,1]:
   for key in ['hd95','assd']:
    rows=[]
    for (i,k,a),v in d.items():
     if a!='FULL':continue
     other=d[i,k,'CONTROL'];x=v['boundary'][h][key];y=other['boundary'][h][key]
     if x is not None and y is not None:rows.append({'content':i,'delta':x-y})
    paired.append({'seed':seed,'resolution':resolution,'head':['OD','OC'][h],'metric':key,'paired_finite_conditions':len(rows),'full_minus_control_content_mean':cm(rows,lambda r:r['delta'])})
out['paired_boundary']=paired
active=[]
for line in (R/'actions.private.jsonl').read_text().splitlines():
 v=json.loads(line);n=hits=0;regrets=[];harms=[]
 for a in range(8):
  for b in range(8):
   old=[v['old_credit'][e][a][b] for e in range(4)];g=[v['exec_credit'][e][a][b] for e in range(4)];ch=[v['changed_pixels'][e][a][b] for e in range(4)]
   harms.extend(g[e] for e in range(4) if old[e]>1e-6 and g[e]<-1e-6)
   if max(abs(x) for x in g)<=1e-6:continue
   n+=1;best=max(0,max(g));ids=[e for e in range(4) if abs(old[e]-max(old))<=1e-6];hits+=any(abs(g[e]-best)<=1e-6 for e in ids);regrets.append(mean(best-g[e] for e in ids))
 active.append({'seed':v['seed'],'content':v['content'],'head':v['head'],'active':n,'hits':hits,'regret':mean(regrets) if regrets else None,'negative_action_mean':mean(harms) if harms else None})
out['active_region_credit']=[]
for seed in [10223,10224]:
 for h in [0,1]:
  rows=[v for v in active if v['seed']==seed and v['head']==h];groups=defaultdict(list)
  for v in rows:groups[v['content']].append(v)
  rates=[sum(v['hits'] for v in vs)/sum(v['active'] for v in vs) for vs in groups.values() if sum(v['active'] for v in vs)]
  out['active_region_credit'].append({'seed':seed,'head':['OD','OC'][h],'eligible_contents':len(rates),'active_region_topset_intersection_content_mean':mean(rates),'active_region_regret_pp':cm(rows,lambda r:r['regret']),'positive_to_negative_gain_pp':cm(rows,lambda r:r['negative_action_mean'])})
out['functional_content_distributions']=[]
for seed in [10223,10224]:
 for mode in ['GLOBAL','SHUFFLE']:
  rows=[v for v in f if v['seed']==seed and v['mode']==mode];groups=defaultdict(list)
  for v in rows:groups[v['content']].append(mean(v['full_minus_readout']))
  vals=sorted(mean(v) for v in groups.values());out['functional_content_distributions'].append({'seed':seed,'mode':mode,'content_delta_min_pp':min(vals),'content_delta_median_pp':statistics.median(vals),'content_delta_max_pp':max(vals),'content_delta_quartiles_pp':statistics.quantiles(vals,n=4,method='inclusive'),'positive_contents':sum(x>0 for x in vals),'negative_contents':sum(x<0 for x in vals)})
(R/'REPORT.json').write_text(json.dumps(out,indent=2)+'\n')
print('ACTIVE',json.dumps(out['active_region_credit']))
print('PAIRED',json.dumps(paired))
