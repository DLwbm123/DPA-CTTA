"""Fixed content-level analysis, JSON only; no model or data access."""
import json,sys,statistics,itertools,math
from pathlib import Path
R=Path(sys.argv[1]);ARMS=['ZERO','DIRECT1','DIRECT3','DENOISE1','DENOISE3_NO_EXTRA_IMAGE','DENOISE3']
def avg(a):return statistics.mean(a)
def quant(a,q=.05):
 a=sorted(a);t=(len(a)-1)*q;i=int(t);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(t-i)
end=json.loads((R/'CONTROL_EXIT.json').read_text());finish=json.loads((R/'TRAINING_FINISHED.json').read_text())
assert end['wrapper_exit_code']==0 and end['termination_confirmed'] and end['within_deadline']
assert finish['status']=='COMPLETE' and finish['completed_seeds']==[10225,10226] and finish['extra_counts']['optimizer_steps']==4000
results=[]
for seed in [10225,10226]:
 run=R/'runs'/str(seed);rows=json.loads((run/'detailed.private.json').read_text());done=json.loads((run/'completion.json').read_text())
 assert len(rows)==80 and len({(x['content'],x['condition']) for x in rows})==80
 assert len({x['content'] for x in rows})==20 and done['episodes_per_arm']==1000 and done['base_frozen'] and done['readout_weights_frozen']
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
 out['new_structure_events']=sum((x['DENOISE3']['empty_od'] and not x['ZERO']['empty_od']) or (x['DENOISE3']['empty_oc'] and not x['ZERO']['empty_oc']) or (x['DENOISE3']['cup_outside']>.01 and x['ZERO']['cup_outside']<=.01) for x in rows)
 results.append(out)
ps=[r['comparisons']['DENOISE3-ZERO'] for r in results]
primary=all(p['dice']['mean']>0 and p['od']['mean']>=0 and p['oc']['mean']>=0 and p['clean']>=-.3 and p['dice']['q05']>=-2 for p in ps) and avg([p['dice']['mean'] for p in ps])>=.2
safety=all(r['new_structure_events']==0 for r in results)
def gate(name,threshold):
 values=[r['comparisons'][name]['dice']['mean'] for r in results];return all(v>0 for v in values) and avg(values)>=threshold
gates={'medical_primary':primary,'structure_safety':safety,'versus_equal_capacity_direct3':gate('DENOISE3-DIRECT3',.10),'iteration':gate('DENOISE3-DENOISE1',.05),'extra_image_diagnostic':gate('DENOISE3-DENOISE3_NO_EXTRA_IMAGE',.05)}
report={'results':results,'gates':gates,'finish':finish,'control_exit':end,'limitations':['Repeated source development; patient identity UNKNOWN','Additional source training; matched direct control reported','No target-domain validation or persistent CTTA state','No-image diagnostic retains image-derived initial SDF and is out of distribution','r01 migration RNG lacks prior byte snapshot','Not a normalized posterior or implemented full diffusion model'],'decision':'Retain for separately controlled next hypothesis; no paper-superiority claim' if primary and safety and gates['versus_equal_capacity_direct3'] else 'Reject current denoising mechanism configuration; do not increase width, steps or search noise grid'}
(R/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
lines=['# r06：医学条件去噪与等容量直接纠错','',report['decision'],'','预注册门：`'+json.dumps(gates)+'`。','', '|seed|ZERO|DIRECT1|DIRECT3|DENOISE1|NO_EXTRA_IMAGE|DENOISE3|','|---|---:|---:|---:|---:|---:|---:|']
for r in results:lines.append('|'+str(r['seed'])+'|'+'|'.join(f"{r['arms'][a]['dice']:.6f}" for a in ARMS)+'|')
lines+=['','## 全部15组成对差','', '|seed|比较|Dice pp|OD pp|OC pp|clean pp|内容q05 pp|vCDR误差差|','|---|---|---:|---:|---:|---:|---:|---:|']
for r in results:
 for name,p in r['comparisons'].items():lines.append(f"|{r['seed']}|{name}|{p['dice']['mean']:+.6f}|{p['od']['mean']:+.6f}|{p['oc']['mean']:+.6f}|{p['clean']:+.6f}|{p['dice']['q05']:+.6f}|{p['vcdr_error']['mean']:+.6f}|")
lines+=['','## 四条件与结构事件','']
for r in results:
 lines.append(f"seed {r['seed']} DENOISE3对ZERO新增结构事件：{r['new_structure_events']}。")
 for a in ARMS:lines.append(f"- {a}: "+json.dumps(r['arms'][a]))
lines+=['','## 共同有限边界覆盖与差值','']
for r in results:
 for n,p in r['comparisons'].items():lines.append(f"- seed {r['seed']} {n}: "+json.dumps(p['finite_boundaries']))
lines+=['','## 证据边界','',*['- '+x for x in report['limitations']],'','训练loss下降不作分割证据；所有读出固定末步、按内容聚合，无best或测试标签选择。原r01-r05负结果与停止条件不变。']
(R/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n');print(json.dumps(gates))
