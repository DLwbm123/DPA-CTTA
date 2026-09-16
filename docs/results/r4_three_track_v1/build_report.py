"""Rebuild report/tables from the public export, standard library only."""
import csv,datetime,gzip,json
from pathlib import Path
from statistics import mean
from export_scalars import table

root=Path(__file__).resolve().parent
with gzip.open(root/'aggregate.json.gz','rt') as f:a=json.load(f)
e=json.loads((root/'execution.json').read_text())
assert a['status']==e['status']=='R4T_EXPERIMENT_COMPLETE' and a['binding']==e['binding']
assert len(e['completions'])==70 and e['process_exit_codes']==[0]*73
P=a['primary_four_order_equal']['remaining_dev']['arms'];S=a['secondary_recurrence']['remaining_dev']['domain_equal_dice_percent'];arms=list(P)
comparisons=[]
for pair,delta in a['primary_four_order_equal']['remaining_dev']['comparisons_pp'].items():
    orders=[a['target'][str(o)]['remaining_dev']['comparisons_pp'][pair] for o in range(4)]
    assert abs(mean(orders)-delta)<1e-10
    comparisons.append(dict(pair=pair,primary_delta_pp=delta,positive_orders=sum(v>0 for v in orders),worst_same_order_delta_pp=min(orders),secondary_delta_pp=a['secondary_recurrence']['remaining_dev']['comparisons_pp'][pair]))
table(root,'comparisons.csv',comparisons)

def md(head,rows):return '\n'.join(['| '+' | '.join(head)+' |','| '+' | '.join(['---']*len(head))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def aux(arm,key,orders=range(4)):
    values=[a['mechanism'][f'o{o}a{arms.index(arm)}'][key] for o in orders]
    return sum(x['mean']*x['n'] for x in values)/sum(x['n'] for x in values)

zone=datetime.timezone(datetime.timedelta(hours=8));stamp=lambda t:datetime.datetime.fromtimestamp(t,zone).strftime('%Y-%m-%d %H:%M:%S')
report=f'''# R4T 三方向冻结实验完成报告

**R4T_EXPERIMENT_COMPLETE**：70/70 条完整轨迹、136,570 条正式记录；3 次机械 smoke 和 70 个正式进程均退出 0，未发生失败或重试。原 CPU 标量重算有效，公开导出核对全部记录与物理计数通过。

结论：保留 C 为参考，不晋级三条新增路线，不追加组合、调参或 GPU 诊断。主表最高的已有 RP 比 C 高 0.169pp，但低于 +0.5pp 筛选参考，回访流略低于 C；不把多臂中的小正差称为新方法成功。

## 时间、提交与生命周期

- 北京时间 {stamp(e['launcher']['started_unix'])} 启动，{stamp(e['launcher']['ended_unix'])} 完成计算及 CPU 校验；含前后处理约 {(e['launcher']['ended_unix']-e['launcher']['started_unix'])/3600:.3f} 小时。
- 监督器墙钟 {e['wall_seconds']/3600:.3f} 小时，累计 active-worker {e['active_worker_seconds']/3600:.3f} 小时；物理 GPU 5、6、7，最多三个 worker，RTX 3090。共享设备耗时不是受控速度基准。
- 实际运行 SHA：`{a['binding']['code_sha']}`。
- Science SHA256：`{a['binding']['science_sha256']}`。
- Registration digest：`{a['binding']['registration_digest']}`。
- Recurrence digest：`{a['binding']['stream_digest']}`。
- 生命周期：阶段 I 全部实现及 CPU 验收 → 用户明确免除本轮外部 review → 启动前补接既有 neutral_subprocesses 入口保护，本地/服务器各 5 个针对性测试通过 → 三 GPU smoke → 全部 70 条正式轨迹 → COMPUTE_COMPLETE → CPU 标量校验 → R4T_EXPERIMENT_COMPLETE → 公开导出。
- 入口补接不改变科学配置。没有 GPU 失败前缀、额外正式尝试或自动重试。免审是用户授权，不是外部 review 通过。阶段 I 的历史失败 CPU 日志继续保留在 review 目录，旧 R3 失败历史也未覆盖。

## 终点和全部 14 臂

主终点为 remaining_dev（1,695 内容/轨迹）：每图 OD/OC 平均、域内平均、四域等权、四主序等权。Dice 单位 %，差值 pp。回访流单列，禁止五流平均；它回访环境但不重复旧图，也不是独立患者队列或固定旧域遗忘测量。Drishti_GS 的 37 内容仍占每序 25% 域权重。

'''
report+=md(['臂','OD','OC','主 macro','主 ΔC pp','回访 macro','回访 ΔC pp'],[[arm,f"{v['OD']:.4f}",f"{v['OC']:.4f}",f"{v['macro']:.4f}",f"{v['macro']-P['C']['macro']:+.4f}",f"{S[arm]['macro']:.4f}",f"{S[arm]['macro']-S['C']['macro']:+.4f}"] for arm,v in P.items()])
report+='\n\n## 匹配控制与选择\n\n'+md(['配对','主 Δpp','正主序 /4','最差同序 Δpp','回访 Δpp'],[[r['pair'],f"{r['primary_delta_pp']:+.6f}",r['positive_orders'],f"{r['worst_same_order_delta_pp']:+.4f}",f"{r['secondary_delta_pp']:+.4f}"] for r in comparisons])
report+='''

- **A 教师：不晋级。** MT 比 C 低 0.537pp，MT_RP 低 0.340pp；FT 与 FT_RP 更低。RP 在相同教师下改善 MT 约 0.196pp、FT 约 0.216pp，但 MT_RP/FT_RP 均低于原 RP，不能把这些增益归给 EMA。MT 优于 FT 并不能支持 MT 优于 C。
- **B Kernel：不晋级。** KDG 比 C 高 0.037pp，3/4 主序正向，但比 K_ALL/K_FREE 仅高 0.00684/0.00893pp，未达到 +0.5pp 参考。局部 kernel 改变和不变量成立不能替代最终分割增益，也不证明旋转约束必要。
- **C 边界图：不晋级。** G_BOUND 比 C 低 0.071pp，低于 G_GLOBAL 约 0.00361pp；比常量和打乱边权控制的小优势不足以建立有用机制。回访比 C 高 0.011pp，而 G_GLOBAL 仍更高，不能宣称边界限制优于全域。

+0.5pp 且至少 3/4 主序同向是资源选择参考，不是显著性、临床或发表门槛。域均值低于 C 2pp、最差同序低于 C 0.5pp仅为风险提示。四序共享内容且 seed 固定，不作为独立患者重复；结论只覆盖本 checkpoint、冻结配置与开发流。

## 域差异、配对尾部和 ASSD

以下是主四序平均域内差值，REFUGE_Valid 的 OC 单列。所有臂逐域、逐序见 remaining_domain_scores.csv，全部直接控制的配对分布和 ASSD 分母见 remaining_pairs.csv。

'''
candidates=['MT','MT_RP','FT','FT_RP','KDG','G_BOUND'];domains=[]
for arm in candidates:
    for domain in a['target']['0']['remaining_dev']['domains']:
        domains.append([arm,domain]+[f"{mean(a['target'][str(o)]['remaining_dev']['domains'][domain]['paired'][arm+'-C'][ch]['dice_delta_pp']['mean'] for o in range(4)):+.4f}" for ch in ('OD','OC','macro')])
report+=md(['臂−C','域','OD Δpp','OC Δpp','macro Δpp'],domains)
risk=[];assd=[]
for arm in candidates:
    rows=[a['target'][str(o)]['remaining_dev']['pooled_content_paired'][arm+'-C'] for o in range(4)]
    x=[r['macro']['dice_delta_pp'] for r in rows];n=sum(v['n'] for v in x);assert n==6780
    risk.append([arm]+[f"{sum(v[k] for v in x)/n*100:.2f}" for k in ('positive','zero','negative')]+[f"{mean(v[k] for v in x):+.4f}" for k in ('median','worst_decile_mean')]+[f"{min(v['minimum'] for v in x):+.4f}"])
    for ch in ('OD','OC'):
        n=sum(r[ch]['assd_common_valid'] for r in rows)
        assd.append([arm,ch,n,sum(r[ch]['assd_not_jointly_defined'] for r in rows),f"{sum(r[ch]['assd_delta_mean_px']*r[ch]['assd_common_valid'] for r in rows)/n:+.4f}"])
report+='\n\n内容加权尾部诊断不等于四域等权主终点。正/零/负按 6,780 个内容-顺序配对计；中位数与下侧最差10%为各序统计量的平均，最差内容取四序最小值。\n\n'+md(['臂−C','正 %','零 %','负 %','中位数均值 pp','最差10%均值 pp','最差内容 pp'],risk)
report+='\n\nASSD 只按共同有效配对汇总，单位为预测栅格像素，负值较好，undefined 不填0。逐臂 undefined 及上侧不利10%见 CSV。\n\n'+md(['臂−C','通道','共同有效','非共同有效','ASSD Δpx'],assd)
report+='\n\n## 辅助机制（全部 1,951 内容，四主序汇总）\n\n这些辅助量为所有内容的图均值，不是 remaining_dev 的域等权主指标。q 是每种方法自己的更新前教师；只在状态提交后用 GT 评分。各通道 Dice、Brier、前景/背景 Brier、分母及分布完整保留在 mechanism_distributions.csv / aggregate.json.gz。\n\n'
report+=md(['臂','q OD Dice %','q OC Dice %','q OD Brier','q OC Brier'],[[arm]+[f"{100*aux(arm,f'auxiliary.q.{ch}.dice'):.4f}" for ch in (0,1)]+[f"{aux(arm,f'auxiliary.q.{ch}.brier'):.6f}" for ch in (0,1)] for arm in arms])
graph=[]
for arm in ('G_BOUND','G_CONST','G_SHUFFLE','G_GLOBAL'):
    for ch,name in enumerate(('OD','OC')):
        graph.append([arm,name,f"{100*(aux(arm,f'auxiliary.qstar.{ch}.dice')-aux(arm,f'auxiliary.q.{ch}.dice')):+.4f}",f"{aux(arm,f'auxiliary.qstar.{ch}.brier')-aux(arm,f'auxiliary.q.{ch}.brier'):+.7f}",f"{aux(arm,f'auxiliary.graph.transitions.{ch*3}.wrong_to_correct'):.2f}",f"{aux(arm,f'auxiliary.graph.transitions.{ch*3}.correct_to_wrong'):.2f}"])
report+='\n\n图纠错/误纠正为每图平均像素数。只改善 q* 不能晋级；最终输出仍是学生更新后的原图预测。\n\n'+md(['图臂','通道','q*−q Dice pp','q*−q Brier','错→对像素/图','对→错像素/图'],graph)
report+='\n\nRP token 质量统计来自实际提交的学生 pre 特征位置；以下为四主序全图所有 bank 的计数比值。\n\n'+md(['臂','实际提交 token 正确率 %'],[[arm,f"{100*sum(aux(arm,f'auxiliary.memory.regions.{i}.correct_tokens') for i in range(4))/sum(aux(arm,f'auxiliary.memory.regions.{i}.selected_tokens') for i in range(4)):.4f}"] for arm in ('RP','MT_RP','FT_RP')])
report+='\n\n实际有效 kernel 位移为每步 ||W_eff−W_source|| 的平均值；不同参数化坐标数和有效函数空间并不等价。\n\n'+md(['臂','层','有效位移 L2'],[[arm,layer,f"{aux(arm,'r4t.kernels.'+layer+'.effective_change_l2'):.7f}"] for arm in ('KDG','K_ALL','K_MAG','K_FREE') for layer in ('res.conv1','res.layer1.2.conv2')])
costs=list(csv.DictReader((root/'costs_by_trajectory.csv').open()));costrows=[]
for arm in arms:
    rows=[r for r in costs if r['arm']==arm]
    costrows.append(dict(arm=arm,mean_minutes=mean(float(r['worker_seconds']) for r in rows)/60,mean_host_ms=1000*sum(float(r['host_seconds']) for r in rows)/(1951*5),peak_allocated_MiB=max(int(r['peak_allocated_bytes']) for r in rows)/1024**2))
table(root,'costs_by_arm.csv',costrows)
report+='\n\n## 成本与边界\n\n正式 1,112,070 forward、136,570 backward/Adam、0 VJP；三次 smoke 合计 780 forward、96 backward/Adam。总计 1,112,850 forward、136,666 backward/Adam。没有离线源模型训练、额外 GPU 诊断或重复矩阵。下表 peak allocated 为 PyTorch 张量分配峰值，不是 reserved、进程 RSS 或 nvidia-smi 总显存。\n\n'+md(['臂','平均轨迹 min','host ms/图','peak allocated MiB'],[[r['arm'],f"{r['mean_minutes']:.2f}",f"{r['mean_host_ms']:.1f}",f"{r['peak_allocated_MiB']:.1f}"] for r in costrows])
report+='''

记录只支持标量关系重算；未重建图像、完整概率图、协方差、Jacobian、图解或 ASSD 几何。原始 RGB/mask/权重、私有 registration、机器身份及逐内容记录不公开。GT 未参与方法或状态更新，未访问源数据、代理、原型或重训源模型。

## 复现与证据索引

- [完整聚合](aggregate.json.gz)：所有 subset、四主序与回访分开、全部配对与辅助分布。
- [执行证据](execution.json)：去身份的 73 个退出码、70 个完成记录、三次 smoke 及 backend，实际执行 SHA 与 science/registration/stream 摘要。
- [全部终点表](all_subset_domain_equal.csv)、[配对控制](comparisons.csv)、[逐域分数](remaining_domain_scores.csv)、[配对分布及 ASSD](remaining_pairs.csv)。
- [辅助机制分布](mechanism_distributions.csv)、[逐轨迹成本](costs_by_trajectory.csv)、[各臂成本](costs_by_arm.csv)。
- [公开导出代码](export_scalars.py)、[真实导出 CPU 日志](export_cpu.log)、[报告复现代码](build_report.py)、[报告检查日志](tables_cpu.log)。在本目录用 Python 执行 build_report.py 可从公开聚合重建报告与派生表；无需 Torch、原始数据或 GPU。
- [阶段 I 材料](../../review/r4_three/REVIEW_INDEX.md)与[入口修复说明](../../review/r4_three/EXECUTION_ADDENDUM.md)保留原历史。

实验结束后停止，无后续搜索或组合授权。
'''
(root/'REPORT.md').write_text(report)
print('PASS 70 trajectories / 73 exits; matched aggregates, 25 comparison rows; report rebuilt using public scalars only')
