"""Fixed content-level analysis, JSON only; no model or data access."""
import json,sys,statistics,itertools,math
from pathlib import Path
R=Path(sys.argv[1]);ARMS=['INVARIANT','SHUFFLED','SHARED']
def avg(a):return statistics.mean(a)
def quant(a,q=.05):
 a=sorted(a);t=(len(a)-1)*q;i=int(t);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(t-i)
end=json.loads((R/'CONTROL_EXIT.json').read_text());finish=json.loads((R/'TRAINING_FINISHED.json').read_text())
assert end['wrapper_exit_code']==0 and end['termination_confirmed'] and end['within_deadline']
assert finish['status']=='COMPLETE' and finish['completed_seeds']==[10225,10226] and finish['counts']['updates']==12000 and finish['counts']['training_encoder_images']==192000 and finish['counts']['evaluation_encoder_images']==480
results=[]
for seed in [10225,10226]:
 run=R/'runs'/str(seed);rows=json.loads((run/'detailed.private.json').read_text());done=json.loads((run/'completion.json').read_text())
 assert len(rows)==80 and len({(x['content'],x['condition']) for x in rows})==80
 assert len({x['content'] for x in rows})==20 and done['optimizer_steps_per_arm']==2000 and done['source_training_from_scratch'] and done['readout_weights_frozen']
 out={'seed':seed,'completion':done,'arms':{},'comparisons':{}}
 for a in ARMS:
  out['arms'][a]={k:avg([x[a][k] for x in rows]) for k in ['dice','od','oc','vcdr_error']}
  out['arms'][a]['conditions']={c:avg([x[a]['dice'] for x in rows if x['condition']==c]) for c in ['clean','gamma','color','blur']}
  out['arms'][a]['empty_od']=sum(x[a]['empty_od'] for x in rows);out['arms'][a]['empty_oc']=sum(x[a]['empty_oc'] for x in rows)
  out['arms'][a]['cup_outside_gt1pct']=sum(x[a]['cup_outside']>.01 for x in rows)
 for b,a in itertools.combinations(ARMS,2):
  pair={}
  for k in ['dice','od','oc','vcdr_error']:
   values=[avg([x[a][k]-x[b][k] for x in rows if x['content']==i]) for i in sorted({x['content'] for x in rows})]
   pair[k]={'mean':avg(values),'q05':quant(values),'minimum':min(values)}
  pair['clean']=avg([x[a]['dice']-x[b]['dice'] for x in rows if x['condition']=='clean'])
  diff=[x[a]['dice']-x[b]['dice'] for x in rows];pair['condition_win_tie_loss']=[sum(v>0 for v in diff),sum(v==0 for v in diff),sum(v<0 for v in diff)]
  pair['finite_boundaries']={}
  for k in ['od_hd95','oc_hd95','od_assd','oc_assd']:
   valid=[x for x in rows if x[a][k] is not None and x[b][k] is not None and math.isfinite(x[a][k]) and math.isfinite(x[b][k])]
   ids=sorted({x['content'] for x in valid});vals=[avg([x[a][k]-x[b][k] for x in valid if x['content']==i]) for i in ids]
   pair['finite_boundaries'][k]={'paired_conditions':len(valid),'total_conditions':80,'paired_contents':len(ids),'mean_content_delta':avg(vals) if vals else None}
  out['comparisons'][a+'-'+b]=pair
 out['new_structure_events']=sum((x['SHARED']['empty_od'] and not x['INVARIANT']['empty_od']) or (x['SHARED']['empty_oc'] and not x['INVARIANT']['empty_oc']) or (x['SHARED']['cup_outside']>.01 and x['INVARIANT']['cup_outside']<=.01) for x in rows)
 results.append(out)
ps=[r['comparisons']['SHARED-INVARIANT'] for r in results]
primary=all(p['dice']['mean']>0 and p['od']['mean']>=0 and p['oc']['mean']>=0 and p['clean']>=-.3 and p['dice']['q05']>=-2 for p in ps) and avg([p['dice']['mean'] for p in ps])>=.2
safety=all(r['new_structure_events']==0 for r in results)
qualification=all(r['arms'][arm]['conditions']['clean']>=75 for r in results for arm in ARMS)
primary=primary and qualification
def gate(name,threshold):
 values=[r['comparisons'][name]['dice']['mean'] for r in results];return all(v>0 for v in values) and avg(values)>=threshold
gates={'medical_primary':primary and safety,'structure_safety':safety,'source_training_qualified':qualification,'paired_action_mechanism':gate('SHARED-SHUFFLED',.1)}
report={'results':results,'gates':gates,'finish':finish,'control_exit':end,'limitations':['Repeated development and patient identity UNKNOWN','From-scratch source representation only; not a generative model or CTTA','Gamma interventions preserve mathematical ordering, not guaranteed clinical sufficiency','Ridge support solve is not proof of rank or group identifiability','INVARIANT computes unused matrices; gradient FLOPs are not exactly matched','No target-domain or paper superiority evidence'],'decision':'Mechanism supported within frozen source-development test only' if all(gates.values()) else 'Reject fixed shared-action configuration; no dimension/ridge/gamma/temperature/steps/seed search'}
(R/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
lines=['# r13：源训练跨内容作用迁移','',report['decision'],'',json.dumps(gates),'','|seed|INVARIANT|SHUFFLED|SHARED|','|---|---:|---:|---:|']
for r in results:lines.append('|'+str(r['seed'])+'|'+'|'.join(f"{r['arms'][a]['dice']:.6f}" for a in ARMS)+'|')
for r in results:
 lines += ['',f"## seed {r['seed']} 全部三组成对差",'']
 for name,v in r['comparisons'].items():lines.append('- '+name+': '+json.dumps(v))
 lines += ['', '四条件、单头、结构事件：',json.dumps(r['arms']), '新增结构事件：'+str(r['new_structure_events'])]
lines += ['', '## 限制','',*['- '+v for v in report['limitations']],'','作用拟合/训练loss下降不是分割支持；固定末步一次读出，无best。']
(R/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gates))
