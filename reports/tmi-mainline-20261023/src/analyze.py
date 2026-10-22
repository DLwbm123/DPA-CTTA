"""Aggregate source-development evidence; never consumes target labels."""
import json,sys
from pathlib import Path
import random, statistics
def quantile(v,q):
 v=sorted(v); a=(len(v)-1)*q; i=int(a); return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(a-i)
r=Path(sys.argv[1]);out=r/'analysis';out.mkdir(exist_ok=True)
report={'scope':'source development only; patient independence and backbone membership unknown','seeds':{},'passed':False}
for seed in [10223,10224]:
 d=r/f'direct_{seed}';f=r/f'full_{seed}'
 if not all((v/'completion.json').exists() for v in [d,f]):
  report['seeds'][seed]={'status':'INCOMPLETE'};continue
 if any(json.loads((v/'completion.json').read_text())['status']!='COMPLETE' for v in [d,f]):
  report['seeds'][seed]={'status':'STOPPED'};continue
 dm=json.loads((d/'best_metrics.json').read_text());fm=json.loads((f/'best_metrics.json').read_text())
 direct=dm['metrics']['NO_UPDATE'];full=fm['metrics']['FULL'];source=fm['metrics']['SOURCE']
 fr=json.loads((f/'best_rows.private.json').read_text());dr=json.loads((d/'best_rows.private.json').read_text())
 key=lambda v:(v['content'],v['order'],v['shift'])
 dd={key(v):v for v in dr if v['arm']=='NO_UPDATE'}
 delta=[(v, v['dice']-dd[key(v)]['dice']) for v in fr if v['arm']=='FULL']
 # Aggregate repeated shifts/orders within each content; not patient level.
 content=sorted(set(v['content'] for v,x in delta));vals=[statistics.mean([x for v,x in delta if v['content']==i]) for i in content]
 rng=random.Random(20261023);boot=[statistics.mean(rng.choices(vals,k=len(vals))) for _ in range(2000)]
 tail=float(quantile(vals,.05));gain=full['dice']-direct['dice'];sgain=full['dice']-source['dice'];clean=full['clean_dice']-direct['clean_dice'];oc=full['oc']-direct['oc']
 gates={'direct_ge_1pp':gain>=1,'source_ge_1pp':sgain>=1,'clean_ge_minus_0_3pp':clean>=-.3,'oc_nonnegative':oc>=0,'content_tail5_ge_minus2pp':tail>=-2}
 report['seeds'][seed]={'status':'COMPLETE','direct_step':dm['step'],'full_step':fm['step'],'direct_metrics':direct,'full_metrics':fm['metrics'],'full_minus_direct_pp':gain,'full_minus_source_pp':sgain,'clean_delta_pp':clean,'oc_delta_pp':oc,'content_tail5_pp':tail,'content_bootstrap_descriptive_95':[quantile(boot,.025),quantile(boot,.975)],'full_minus_episodic_pp':(full['dice']-fm['metrics']['EPISODIC']['dice'] if 'EPISODIC' in fm['metrics'] else None),'full_minus_random_pp':(full['dice']-fm['metrics']['RANDOM']['dice'] if 'RANDOM' in fm['metrics'] else None),'full_minus_no_update_pp':(full['dice']-fm['metrics']['NO_UPDATE']['dice'] if 'NO_UPDATE' in fm['metrics'] else None),'gates':gates,'pass':all(gates.values())}
 if 'TEACHER' in fm['metrics']:
  report['seeds'][seed]['full_minus_teacher_pp']=full['dice']-fm['metrics']['TEACHER']['dice']
 u=r/f'uniform_{seed}'/'best_metrics.json'
 if u.exists():
  um=json.loads(u.read_text())['metrics']['MATCHED_UNIFORM'];report['seeds'][seed]['matched_uniform_metrics']=um;report['seeds'][seed]['full_minus_matched_uniform_pp']=full['dice']-um['dice']
report['passed']=all(x.get('pass',False) for x in report['seeds'].values())
(out/'REPORT.json').write_text(json.dumps(report,indent=2)+'\n')
s=[f'# {r.parent.name}源开发结果','',f"开发门：{'PASS' if report['passed'] else 'NOT PASSED'}。不是独立患者/真实目标验证。",'', '|种子|FULL−DIRECT pp|FULL−SOURCE pp|FULL−NO_UPDATE pp|FULL−EPISODIC pp|FULL−RANDOM pp|门|','|---|---:|---:|---:|---:|---:|---|']
for seed,x in report['seeds'].items():
 if x['status']!='COMPLETE':s.append(f'|{seed}|未完成|||||{x["status"]}|');continue
 s.append('|'+str(seed)+'|'+ '|'.join((f'{x[k]:.6f}' if x[k] is not None else 'NA') for k in ['full_minus_direct_pp','full_minus_source_pp','full_minus_no_update_pp','full_minus_episodic_pp','full_minus_random_pp'])+'|'+str(x['pass'])+'|')
for seed,x in report['seeds'].items():
 if 'full_minus_teacher_pp' in x:s.append(f"\n种子{seed} FULL−TEACHER: {x['full_minus_teacher_pp']:.6f}pp；不能把独立视图收益全部归给适应。")
for seed,x in report['seeds'].items():
 if 'full_minus_matched_uniform_pp' in x:s.append(f"\n种子{seed} FULL−独立训练均匀融合: {x['full_minus_matched_uniform_pp']:.6f}pp；这是区域选择的必要对照。")
s+=['','检查点仅据开发集选取；bootstrap以内容为单位合并顺序和扰动，仅描述性。两个种子共享开发集，不作为独立队列。未通过门时保留负结果，不修改门。历史接口失败与工程重试分别归档，见各轮协议和运行回执。']
(out/'REPORT_ZH.md').write_text('\n'.join(s)+'\n');print(json.dumps(report,indent=2))
