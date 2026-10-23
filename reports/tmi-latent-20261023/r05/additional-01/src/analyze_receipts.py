"""Validate existing JSON aggregates only; no images, fitting or model calls."""
import json, math, statistics, sys
from pathlib import Path
r=Path(sys.argv[1]);v=json.loads((r/'collected.private.json').read_text())
d=v['REPORT.json'];rows=v['detailed.private.json'];c=v['TRAINING_FINISHED.json']['counts']
assert len(rows)==24 and len({x['content'] for x in rows})==12
assert all(x['qualified'] for x in rows)
assert c['source_images_read']==c['source_masks_read']==79
assert c['synthetic_solves']==6 and c['calibration_solves']==67 and c['probe_solves']==144
assert all(c[k]==0 for k in ['dev_images_read','target_images_read','neural_model_calls','network_training_updates'])
def near(a,b): assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-12),(a,b)
def q05(a):
 a=sorted(a);i=(len(a)-1)*.05;lo=int(i);return a[lo]+(a[min(lo+1,len(a)-1)]-a[lo])*(i-lo)
for h in ['OD','OC']:
 names=[h+'_SHRINK',h+'_EXPAND'];vals=[statistics.mean(x['scores'][n]-x['scores']['TRUE'] for x in rows if x['content']==i for n in names) for i in sorted({x['content'] for x in rows})]
 rec=d['heads'][h];near(statistics.mean(vals),rec['mean_margin']);near(statistics.median(vals),rec['median_margin']);near(q05(vals),rec['q05_margin'])
 assert [sum(x>1e-6 for x in vals),sum(abs(x)<=1e-6 for x in vals),sum(x < -1e-6 for x in vals)]==rec['content_win_tie_loss']
 assert rec['ranking_gate']==(sum(x>1e-6 for x in vals)>=9 and statistics.mean(vals)>1e-6 and all(statistics.mean(x['scores'][n]-x['scores']['TRUE'] for x in rows if x['view']==view for n in names)>1e-6 for view in ['original','photometric']))
for x in rows:assert x['truth_rank']==1+sum(val<x['scores']['TRUE']-1e-10 for k,val in x['scores'].items() if k!='TRUE')
assert d['observability_gate']==all(d['heads'][h]['ranking_gate'] for h in ['OD','OC'])
e=v['control']['EXIT.json'];clock=v['control']['CLOCK.json'];wall=e['observed_monotonic']-(clock['deadline_monotonic']-900)
assert e['wrapper_exit_code']==0 and e['reason']=='NORMAL_EXIT' and e['within_deadline'] and e['termination_confirmed'] and e['probe_termination_confirmed']
assert v['controller_retired'] and v['wrapper_retired'] and wall<900 and v['allocated_parent_plus_spool_bytes']<100000000
extra={'registered_gate':d['observability_gate'],'truth_best_views':sum(x['truth_rank']==1 for x in rows),'views':24,'direction_win_tie_loss':{n:[sum(x['scores'][n]-x['scores']['TRUE']>1e-6 for x in rows),sum(abs(x['scores'][n]-x['scores']['TRUE'])<=1e-6 for x in rows),sum(x['scores'][n]-x['scores']['TRUE'] < -1e-6 for x in rows)] for n in ['OD_SHRINK','OD_EXPAND','OC_SHRINK','OC_EXPAND']},'interpretation':'Registered mean gate passes, but truth is never best; directional bias prevents treating likelihood minimization as segmentation support','posthoc_gate_change':False,'segmentation_improvement_established':False}
b={'explicit_additional_budget_seconds':900,'additional_wall_seconds':wall,'within_additional_budget':True,'original_receipt_wall_seconds':1173.2566542625427,'original_budget_compliance':False,'original_actual_solve_count':'UNKNOWN','allocated_parent_and_spool_bytes':v['allocated_parent_plus_spool_bytes'],'storage_soft_bytes':50000000,'storage_hard_bytes':100000000,'storage_is_sampled_not_peak_proof':True,'exit_code':0,'all_owned_processes_retired':True,'counts':c,'scientific_conditions_unchanged':True}
(r/'INTERPRETATION.json').write_text(json.dumps(extra,indent=2)+'\n');(r/'BUDGET_RECEIPT.json').write_text(json.dumps(b,indent=2)+'\n')
lines=['# r05 明确追加15分钟后的结果','',f'追加尝试正常结束，用时 {wall:.3f} 秒，计量父r05目录加本地控制记录 {v["allocated_parent_plus_spool_bytes"]:,} 字节。原轮−9、1173.257秒超限和实际求解数UNKNOWN均保留；本次不是重写原预算。','', '**预注册观测门通过，但当前似然不能作为可靠的分割更新目标。** 真值在24个视图中成为五候选最优的次数为0；平均门合并收缩与扩张，掩盖了方向性偏好。不得改写预注册门为失败，也不得把门通过写成Dice收益。','', '|头|平均margin|中位数|5%分位|内容胜/平/负|原图|光度图|','|---|---:|---:|---:|---|---:|---:|']
for h,a in d['heads'].items():lines.append(f'|{h}|{a["mean_margin"]:+.9f}|{a["median_margin"]:+.9f}|{a["q05_margin"]:+.9f}|{a["content_win_tie_loss"]}|{a["view_mean_margin"]["original"]:+.9f}|{a["view_mean_margin"]["photometric"]:+.9f}|')
lines+=['','margin=NLL错误候选−NLL真值，正数有利于真值。两头均达到≥9/12内容为正，均值及两视图均值>1e-6；12/12内容合格。','', '|错误方向|平均margin|视图胜/平/负|','|---|---:|---|']
for h,a in d['heads'].items():
 for n,m in a['direction_mean_margin'].items():lines.append(f'|{n}|{m:+.9f}|{extra["direction_win_tie_loss"][n]}|')
lines+=['','真值五候选名次频数：'+str(d['truth_rank_aggregate'])+'。OD扩张平均更受似然偏好；OC收缩平均也更有利。故模型能对边界改变产生反应，但没有证明它偏好的改变与正确医学边界一致。不能据此启动无约束似然下降式分割适应。','', '本次完成79对源图像/掩膜、67校准、144探针与6合成求解，总217；12内容×2视图。0dev/target读取、0神经网络调用/训练，不占GPU。原轮实际源读取/求解数UNKNOWN，本次217不是累计已知总成本。','', '额外源预算、重复源fit、患者身份UNKNOWN、同图棋盘空间留出和固定有限候选限制均保留；非独立患者泛化、非目标域CTTA、非分割Dice改善或论文超越证据。无解剖负对照候选无关属于结构性质。','', '本次只执行用户明确授权的一次追加尝试，不自动扩样/改阈值/改噪声/加阶数或启动后继实验。源码、冻结协议、追加授权、匿名聚合与去标识回执公开；逐内容、路径配置、数据与权重不公开。']
(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'registered_gate':extra['registered_gate'],'truth_best_views':extra['truth_best_views'],'wall_seconds':wall,'direction_counts':extra['direction_win_tie_loss']}))
