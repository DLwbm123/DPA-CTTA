"""JSON-only fit versus previously sealed dev diagnostic."""
import json,sys,statistics,math
from pathlib import Path
R=Path(sys.argv[1]);D=Path(sys.argv[2]);out=[]
avg=statistics.mean
ARMS=['ZERO','DETACHED1','DETACHED3','FULL1','FULL3_NO_EXTRA_IMAGE','FULL3']
def q05(v):
 v=sorted(v);p=(len(v)-1)*.05;i=int(p);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(p-i)
def aggregate(rows):
 result={'arms':{},'versus_zero':{}}
 for arm in ARMS:
  result['arms'][arm]={k:avg([r[arm][k] for r in rows]) for k in ['dice','od','oc','vcdr_error']}
  result['arms'][arm]['conditions']={c:avg([r[arm]['dice'] for r in rows if r['condition']==c]) for c in ['clean','gamma','color','blur']}
  delta={k:avg([r[arm][k]-r['ZERO'][k] for r in rows]) for k in ['dice','od','oc','vcdr_error']}
  vals=[avg([r[arm]['dice']-r['ZERO']['dice'] for r in rows if r['content']==i]) for i in sorted({r['content'] for r in rows})]
  delta['content_q05']=q05(vals);delta['clean']=avg([r[arm]['dice']-r['ZERO']['dice'] for r in rows if r['condition']=='clean'])
  delta['new_structure_events']=sum((r[arm]['empty_od'] and not r['ZERO']['empty_od']) or (r[arm]['empty_oc'] and not r['ZERO']['empty_oc']) or (r[arm]['cup_outside']>.01 and r['ZERO']['cup_outside']<=.01) for r in rows)
  delta['boundaries']={}
  for k in ['od_hd95','oc_hd95','od_assd','oc_assd']:
   valid=[r for r in rows if r[arm][k] is not None and r['ZERO'][k] is not None and math.isfinite(r[arm][k]) and math.isfinite(r['ZERO'][k])];ids=sorted({r['content'] for r in valid})
   delta['boundaries'][k]={'conditions':len(valid),'total_conditions':len(rows),'contents':len(ids),'content_mean_delta':avg([avg([r[arm][k]-r['ZERO'][k] for r in valid if r['content']==i]) for i in ids]) if ids else None}
  result['versus_zero'][arm]=delta
 return result
exit=json.loads((R/'CONTROL_EXIT.json').read_text());finish=json.loads((R/'TRAINING_FINISHED.json').read_text());assert exit['wrapper_exit_code']==0 and exit['termination_confirmed'] and exit['within_deadline'];assert finish['source_conditions']==632 and finish['extra_counts']['optimizer_steps']==0
for seed in [10225,10226]:
 f=json.loads((R/f'runs/{seed}/detailed.private.json').read_text());d=json.loads((D/f'runs/{seed}/detailed.private.json').read_text());assert len(f)==316 and len(d)==80 and len({(r['content'],r['condition']) for r in f})==316
 z={'seed':seed,'source_fit':aggregate(f),'sealed_dev':aggregate(d)};z['gain_gap_pp']=z['source_fit']['versus_zero']['FULL3']['dice']-z['sealed_dev']['versus_zero']['FULL3']['dice'];out.append(z)
passed=all(z['source_fit']['versus_zero']['FULL3']['dice']>0 and z['gain_gap_pp']>=.5 for z in out)
report={'results':out,'registered_training_generalization_gap':passed,'not_causal_attribution':True,'r07_failure_unchanged':True,'new_dev_or_target_images':0,'training_updates':0}
(R/'REPORT.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
(R/'REPORT_ZH.md').write_text('# r08 冻结纠错器训练内外诊断\n\n预注册差距门：'+str(passed)+'。仅训练内外描述，不能归因于某一个网络，也不是适应成功。r07失败不变。\n\n'+ '\n'.join(f"seed {z['seed']}: fit gain {z['source_fit']['versus_zero']['FULL3']['dice']:+.6f} pp; dev gain {z['sealed_dev']['versus_zero']['FULL3']['dice']:+.6f} pp; gap {z['gain_gap_pp']:+.6f} pp" for z in out)+'\n\n全部六臂、条件、单头、尾部、vCDR、结构事件及共同有限边界见REPORT.json。患者UNKNOWN、重复dev、额外源训练和r01 RNG限制保留。\n')
print(json.dumps({'registered_gap':passed,'gaps':[z['gain_gap_pp'] for z in out]}))
