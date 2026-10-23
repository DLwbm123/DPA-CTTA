"""Aggregate frozen 2x2 readouts; no model calls or private identifiers in output."""
import json,sys,statistics
from pathlib import Path
from collections import defaultdict
R=Path(sys.argv[1]);mean=statistics.mean

def read(p):return json.loads(p.read_text())
def quantile(xs,q):
 xs=sorted(xs);a=(len(xs)-1)*q;i=int(a);return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(a-i)
def content_mean(rows,key):
 g=defaultdict(list)
 for row in rows:
  v=row[key] if isinstance(key,str) else key(row)
  if v is not None:g[row['content']].append(v)
 return mean(mean(v) for v in g.values()) if g else None

def index(rows):
 assert len(rows)==80
 d={(r['content'],r['shift']):r for r in rows};assert len(d)==80
 assert {i for i,k in d}==set(range(20))
 assert {k for i,k in d}=={'clean','gamma','color','blur'}
 return d
assert read(R/'TRAINING_FINISHED.json')['status']=='COMPLETE'
assert read(R/'TRAINING_FINISHED.json')['counts']['outer_updates']==16000
assert read(R/'EXIT.json')['code']==0 and read(R/'EXIT.json')['reason'] is None
out={'scope':'Repeated source-development content; patient and backbone membership UNKNOWN; not independent validation','primary_checkpoint':'last4000','arms':{'A':'r12 learned routing + credit (reused)','B':'learned routing, no credit','C':'r12 fixed DIRECT routing + credit (reused)','D':'fixed DIRECT routing, no credit'},'results':[],'coverage':{'contents':20,'conditions_per_seed_arm_checkpoint':80,'new_training_runs':4,'new_outer_updates':16000,'reuse_readout_conditions_actual':640,'reuse_metadata_typo_correction':'Original REUSE_GATE reported 320; eight saved readouts x80 =640; no rerun'},'runtime':read(R/'EXIT.json'),'counts':read(R/'TRAINING_FINISHED.json')['counts'],'decision_rule':'last A-B both seeds >0; seed mean >=0.15pp, plus preregistered harm gates','r15_authorized':False}
for tag in ['last','best']:
 for seed in [10223,10224]:
  arms={}
  for arm,folder,old in [('A','full',True),('B','full',False),('C','control',True),('D','control',False)]:
   run=R/('reused' if old else 'runs')/f'{folder}_{seed}';rows=read(run/f'{tag}_detailed.private.json');arms[arm]=index(rows)
   if not old:
    c=read(run/'completion.json');assert c['status']=='COMPLETE' and c['steps']==4000
  assert all(set(d)==set(arms['A']) for d in arms.values())
  result={'checkpoint':tag,'seed':seed,'metrics':{},'comparisons':{},'paired_boundaries':[]}
  for a,d in arms.items():
   rows=list(d.values());m={k:content_mean(rows,k) for k in ['dice','od','oc','vcdr_error','empty_heads','outside_cup','outside_fraction']}
   m.update(clean_dice=content_mean([v for v in rows if v['shift']=='clean'],'dice'),outside_conditions=sum(v['outside_cup']>0 for v in rows),empty_conditions=sum(v['empty_heads']>0 for v in rows),boundary_coverage={})
   for h,name in enumerate(['OD','OC']):
    m['boundary_coverage'][name]={k:sum(v['boundary'][h]['empty_case']==k for v in rows) for k in ['neither','one','both']}
    for metric in ['hd95','assd']:
     m[f'{name}_{metric}_finite_mean']=content_mean(rows,lambda r:r['boundary'][h][metric])
   result['metrics'][a]=m
  pairs=[('A-C','A','C'),('A-B','A','B'),('C-D','C','D'),('B-D','B','D')]
  for name,a,b in pairs:
   diffs=[{'content':i,'shift':k,**{metric:arms[a][i,k][metric]-arms[b][i,k][metric] for metric in ['dice','od','oc','vcdr_error']}} for i,k in arms[a]]
   groups=defaultdict(list)
   for v in diffs:groups[v['content']].append(v['dice'])
   vals=[mean(v) for v in groups.values()]
   result['comparisons'][name]={k+'_delta':content_mean(diffs,k) for k in ['dice','od','oc','vcdr_error']}
   result['comparisons'][name].update(clean_dice_delta=content_mean([v for v in diffs if v['shift']=='clean'],'dice'),content_dice_p05=quantile(vals,.05),content_dice_min=min(vals),content_dice_median=statistics.median(vals),content_dice_max=max(vals),positive_contents=sum(v>0 for v in vals),negative_contents=sum(v<0 for v in vals),per_condition={k:content_mean([v for v in diffs if v['shift']==k],'dice') for k in ['clean','gamma','color','blur']})
   for h,head in enumerate(['OD','OC']):
    for metric in ['hd95','assd']:
     finite=[{'content':i,'delta':arms[a][i,k]['boundary'][h][metric]-arms[b][i,k]['boundary'][h][metric]} for i,k in arms[a] if arms[a][i,k]['boundary'][h][metric] is not None and arms[b][i,k]['boundary'][h][metric] is not None]
     result['paired_boundaries'].append({'comparison':name,'head':head,'metric':metric,'finite_conditions':len(finite),'content_mean_delta':content_mean(finite,'delta')})
  interaction=[]
  for i,k in arms['A']:
   interaction.append({'content':i,'delta':arms['A'][i,k]['dice']-arms['B'][i,k]['dice']-arms['C'][i,k]['dice']+arms['D'][i,k]['dice']})
  result['interaction_dice_pp']=content_mean(interaction,'delta')
  ab=result['comparisons']['A-B'];new_empty=new_outside=0
  for key,a in arms['A'].items():
   b=arms['B'][key]
   # With the same GT, one-empty A against neither-empty B identifies an A-only empty head.
   new_empty+=any(a['boundary'][h]['empty_case']=='one' and b['boundary'][h]['empty_case']=='neither' for h in [0,1])
   new_outside+=a['outside_fraction']>.01 and b['outside_fraction']<=.01
  result['harm_gates']={'clean_delta_ge_minus_0_3':ab['clean_dice_delta']>=-.3,'oc_delta_ge_0':ab['oc_delta']>=0,'content_p05_ge_minus_2':ab['content_dice_p05']>=-2,'new_A_empty_conditions_vs_B':new_empty,'new_A_outside_above_1pct_vs_B':new_outside}
  result['harm_gates_pass']=all(result['harm_gates'][k] for k in ['clean_delta_ge_minus_0_3','oc_delta_ge_0','content_p05_ge_minus_2']) and new_empty==0 and new_outside==0
  out['results'].append(result)
primary=[v for v in out['results'] if v['checkpoint']=='last'];ab=[v['comparisons']['A-B']['dice_delta'] for v in primary]
out['primary_decision']={'A_minus_B_per_seed_pp':ab,'mean_A_minus_B_pp':mean(ab),'both_positive':all(v>0 for v in ab),'mean_ge_0_15':mean(ab)>=.15,'all_harm_gates_pass':all(v['harm_gates_pass'] for v in primary),'interaction_both_positive':all(v['interaction_dice_pp']>0 for v in primary),'passes':all(v>0 for v in ab) and mean(ab)>=.15 and all(v['harm_gates_pass'] for v in primary)}
# Algebraic cross-check of independent interaction definitions.
for v in out['results']:
 c=v['comparisons'];assert abs(v['interaction_dice_pp']-(c['A-C']['dice_delta']-c['B-D']['dice_delta']))<1e-9
(R/'REPORT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({'decision':out['primary_decision'],'summary':[{'checkpoint':v['checkpoint'],'seed':v['seed'],'dice':{a:m['dice'] for a,m in v['metrics'].items()},'diff':{a:m['dice_delta'] for a,m in v['comparisons'].items()},'interaction':v['interaction_dice_pp'],'harm':v['harm_gates']} for v in out['results']]},indent=2))
