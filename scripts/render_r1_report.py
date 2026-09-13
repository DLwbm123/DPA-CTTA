"""Render the human report and descriptive figure from public scalar tables."""
import csv
import json
from pathlib import Path
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/results/r1_recovery_target_subspace_20260912'


def render():
    read = lambda name: json.loads((OUT/name).read_text())
    table = lambda name: list(csv.DictReader((OUT/name).open()))
    a, audit, m = read('public_aggregate.json'), read('EXECUTION_AUDIT.json'), read('MECHANISM_AND_COST.json')
    primary, orders, domains, paired = map(table, ['PRIMARY.csv', 'ORDER_SUBSET.csv', 'DOMAIN_SCORES.csv', 'PAIRED_DISTRIBUTIONS.csv'])
    arms = [r['arm'] for r in primary]
    domain_names = ['REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    subsets = ['remaining_dev', 'legacy_dev', 'p1_extension_dev', 'all_dev']
    f = lambda value: f'{float(value):.3f}'
    lines = []
    def add(text=''): lines.append(text)
    def md(headers, rows):
        add('| '+' | '.join(headers)+' |'); add('| '+' | '.join(['---']*len(headers))+' |')
        for row in rows: add('| '+' | '.join(map(str, row))+' |')
        add()
    add('# R1 Recovery / Target Subspace 实验报告\n')
    add('**R1_EXPERIMENT_COMPLETE：24/24 条轨迹、3/3 次设备 smoke 与独立 CPU 汇总全部完成，所有 27 个计算子进程和启动器退出码均为 0。**\n')
    add('主要结论：保留 C 作为当前基线。固定周期恢复在四个顺序上均下降；敏感性恢复未触发，评分与 C 完全相同；区域 PCA 的四序平均提升仅 **0.169 pp**，且只有 **2/4** 顺序改善。GLOBAL 和 SHUFFLED 均轻微下降。区域结构有小幅描述性信号，但本轮不足以确认稳定收益。\n')
    add('## 1. 执行与数据口径\n')
    add('- 运行时间：北京时间 2026-09-12 19:52:51 至 2026-09-13 00:07:22。')
    add('- 执行 SHA：[`54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b`](https://github.com/DLwbm123/DPA-CTTA/commit/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b)；报告提交另列于交付链接，不替代执行 SHA。')
    add('- GPU 4、6、7，均 RTX 3090；最多 3 worker，每卡一个，按冻结轮换分配。共享 GPU 的负载不受控。')
    add('- 一份给定 checkpoint；无源图像/mask/代理/原型、额外预训练或源训练。图像按当前到达顺序处理，固定预测后才读当前 mask。')
    add('- 每序完整 1,951 组；主要终点使用已暴露开发集 remaining_dev 的 1,695 组。先每图 OD/OC 平均，再各域平均，最后四域等权；四序再作描述性平均。')
    add('- pp 是 Dice 百分点。四个顺序重复使用相同内容，不能作为四组独立患者重复；未计算显著性或置信区间。\n')
    md(['order', '域序'], [[o, ' → '.join(a['target'][str(o)]['remaining_dev']['domains'])] for o in range(4)])
    add('## 2. 主要终点：remaining_dev\n')
    md(['方法', 'OD %', 'OC %', 'Macro %', 'ΔC pp', '改善顺序', '最差顺序 %', '顺序范围 pp'],
       [[r['arm'], f(r['OD']), f(r['OC']), f(r['macro']), f(r['delta_C_pp']), str(r['positive_orders'])+'/4', f(r['worst_order']), f(r['order_range'])] for r in primary])
    add('区域 PCA 相对 C 的 OD 为 −0.054 pp、OC 为 +0.392 pp；平均增益主要来自 OC。其顺序范围从 C 的 1.320 pp 扩大至 2.118 pp，最差观察顺序得分比 C 的最差顺序低 0.135 pp。\n')
    md(['方法', 'order0', 'order1', 'order2', 'order3'], [[arm]+[f(a['target'][str(o)]['remaining_dev']['task_domain_macro_dice_percent'][arm]) for o in range(4)] for arm in arms])
    pairs = [('C_SENS','C'), ('C_PER256','C'), ('C_SENS','C_PER256'), ('C_PCA_GLOBAL','C'), ('C_PCA_SHUFFLED','C'), ('C_PCA_REGION','C'), ('C_PCA_REGION','C_PCA_GLOBAL'), ('C_PCA_REGION','C_PCA_SHUFFLED')]
    def scores(arm, subset='remaining_dev'):
        return [a['target'][str(o)][subset]['task_domain_macro_dice_percent'][arm] for o in range(4)]
    md(['配对差值 pp', 'order0', 'order1', 'order2', 'order3', '四序均值'],
       [[x+' − '+y]+[f(v) for v in [sx-sy for sx,sy in zip(scores(x),scores(y))]]+[f(mean(scores(x))-mean(scores(y)))] for x,y in pairs])
    add('REGION 对 GLOBAL 和 SHUFFLED 分别为 +0.263、+0.265 pp，均在 3/4 顺序上占优，但 order1 仍为负。C_SENS 超过周期恢复只是因为周期恢复损害 C；不能据此归因为敏感性触发有效。\n')
    add('![Four-order matched gains](order_gains.png)\n')
    add('图中横线为四个观察顺序的最小至最大差值，圆点为各序，菱形为均值；横线不是置信区间。\n')
    add('## 3. 域差异、配对尾部与 ASSD\n')
    md(['配对 Macro Δ pp']+domain_names, [[x+' − '+y]+[f(mean(a['target'][str(o)]['remaining_dev']['domains'][d]['arms'][x]['macro']['dice_percent']['mean']-a['target'][str(o)]['remaining_dev']['domains'][d]['arms'][y]['macro']['dice_percent']['mean'] for o in range(4))) for d in domain_names] for x,y in pairs])
    add('周期恢复损害集中在 ORIGA（−5.971 pp）和 Drishti_GS（−3.137 pp），虽然 REFUGE_Valid 增加 3.840 pp，仍无法抵消其他域退化。区域 PCA 在 ORIGA、Drishti_GS 改善，在 REFUGE、REFUGE_Valid 下降。\n')
    add('remaining_dev 的四域分别为 336、586、736、37 组，主终点仍按冻结定义各占 25%。Drishti_GS 样本少而改善较大，需要结合这一权重理解区域 PCA 的小幅总增益。\n')
    tail_rows = []
    for left in ['C_PER256','C_PCA_REGION']:
        for domain in domain_names:
            rs = [r for r in paired if r['left']==left and r['right']=='C' and r['domain']==domain and r['subset']=='remaining_dev']
            od, oc = ([r for r in rs if r['channel']==c] for c in ('OD','OC'))
            tail_rows.append([left+' − C', domain, f(mean(float(r['dice_worst_decile_mean']) for r in od)), f(mean(float(r['dice_worst_decile_mean']) for r in oc)), f(mean(100*int(r['dice_negative'])/int(r['dice_n']) for r in oc)), f(mean(float(r['assd_delta_mean_px']) for r in oc)), '/'.join(r['assd_common_valid'] for r in sorted(oc,key=lambda r:r['order']))])
    md(['配对', '域', 'OD 最差10% Δpp', 'OC 最差10% Δpp', 'OC下降比例 %', 'OC共同有效 ASSD Δpx', 'OC有效配对 n（o0/o1/o2/o3）'], tail_rows)
    add('尾部列为各序逐图配对差值最差十分位均值，再对四序取平均；下降比例同样为四序平均。ASSD 仅使用两方法共同定义的图像，正值表示恶化；各序缺失/空预测、正负计数及全部 OD/OC/Macro 分布均保留在 [PAIRED_DISTRIBUTIONS.csv](PAIRED_DISTRIBUTIONS.csv) 和 [原聚合](public_aggregate.json)。不把无定义 ASSD 当作 0，也未重建未保存的边界几何。\n')
    add('## 4. 其他已暴露开发子集\n')
    md(['方法']+[s+' Macro %' for s in subsets], [[arm]+[f(mean(scores(arm,s))) for s in subsets] for arm in arms])
    add('legacy_dev 与 p1_extension_dev 各 128 组，all_dev 为完整 1,951 组；all_dev 与各子集重叠。各序的 OD/OC/Macro 完整表见 [ORDER_SUBSET.csv](ORDER_SUBSET.csv)，逐域同类表见 [DOMAIN_SCORES.csv](DOMAIN_SCORES.csv)。\n')
    add('## 5. 恢复与 PCA 机制\n')
    add('C_PER256 每条轨迹均恢复 7 次，位置固定为 257、513、769、1025、1281、1537、1793；四序合计 28 次。C_SENS 四序均 0 次恢复、没有缺失 sensitivity。达到 age≥50 后，EMA/best 的各序最大值如下，均未达到严格大于 7 的冻结条件。\n')
    md(['order', '最大 EMA/best', '恢复次数'], [[o, f(m[f'o{o}a2']['controller_summary']['max_eligible_ema_over_best']), m[f'o{o}a2']['resets']] for o in range(4)])
    add('因此本次敏感性探测没有改变 C 的恢复状态，却把每图前向数从 8 增加至 11（+37.5%）；不能把它与 C 的零分差称为成功恢复。\n')
    rows = []
    for order in range(4):
        for index in (3,4,5):
            v=m[f'o{order}a{index}']; rows.append([order, arms[index], str(v['pca_ready_visits'])+'/1951', '/'.join(str(b['ready_basis_visits']) for b in v['banks']), '/'.join(str(b['n']) for b in v['pca_last']), '/'.join(str(b['rank']) for b in v['pca_last']), f(v['subloss_mean'])])
    md(['order','方法','任一 basis 可用','各 bank 可用次数','最终各 bank 样本 n','最终 rank','未加权子空间损失均值'],rows)
    add('GLOBAL 为单 bank；REGION 的四 bank 顺序是 OD 背景、OD 前景、OC 背景、OC 前景；SHUFFLED 保留对应配额并打乱语义。任一 bank 可用比例均为 1935/1951（99.18%）；order2 的 OC 前景 bank 为 1929/1951（98.87%）。所有最终 rank 为 8。区域 PCA 的信号偏弱并非整个子空间分支未激活。缺少可靠 token、各 bank 的贡献图像数、状态字节、损失分布及触发位置见 [MECHANISM_AND_COST.json](MECHANISM_AND_COST.json)。\n')
    add('## 6. 实际预算、延迟与内存\n')
    md(['项目','实测'], [['正式记录 / Adam / backward','46,824 / 46,824 / 46,824'],['正式前向','398,004'],['3卡 smoke Adam / backward / 前向','42 / 42 / 354'],['合计 Adam / backward / 前向','46,866 / 46,866 / 398,358'],['总 wall',f(audit['total_wall_seconds']/3600)+' h'],['累计 active worker',f(audit['active_worker_seconds']/3600)+' h'],['最长轨迹',f(max(j['trajectory_seconds'] for j in audit['jobs'])/60)+' min'],['输出占用（沿用监督器计数口径）',f(audit['output_accounted_bytes']/1024**2)+' MiB']])
    cost_rows=[]
    for arm in arms:
        js=[j for j in audit['jobs'] if j['arm']==arm]
        cost_rows.append([arm,f(mean(j['latency']['host_seconds']['mean'] for j in js)*1000),f(mean(j['latency']['pipeline_seconds']['mean'] for j in js)*1000),f(sum(v['seconds'] for j in js for v in j['asset_io'].values())),f(max(j['peak_allocated_bytes'] for j in js)/1024**2)])
    md(['方法','host ms/图','pipeline ms/图','RGB+mask验证/解码总秒','峰值 allocated MiB'],cost_rows)
    add('这里是实际共享设备条件下的观测时间，不是受控方法加速比；每个方法跨四序的设备分配也不是严格等频。host 包含既定适应步骤，pipeline 另含当前资产读取与 evaluator 等处理。内存列是 PyTorch peak allocated，不是整卡占用。每条轨迹的延迟分布、checkpoint 和 RGB/mask 验证 IO、smoke backend、真实退出码见 [EXECUTION_AUDIT.json](EXECUTION_AUDIT.json)。\n')
    add('## 7. 已绑定的历史次要参照\n')
    hist=[]
    for name in ['A','C0']:
        deltas=[mean(a['secondary_A_C0'][str(o)][name]['comparisons']['C']['remaining_dev'][d]['macro']['dice_delta_pp']['mean'] for d in domain_names) for o in range(4)]
        hist.append(['C − '+name]+[f(v) for v in deltas]+[f(mean(deltas))])
    md(['配对 Macro Δpp','order0','order1','order2','order3','四序均值'],hist)
    add('A/C0 四序身份绑定均为 AVAILABLE；作为历史次要参照，未重跑旧方法。C0 沿用已核验的 stateless canonical 例外，再按各序配对。其余六臂与 A/C0 的逐域/子集/通道配对保留在原聚合。\n')
    add('## 8. 校验与交付边界\n')
    add('原执行器已完成冻结版本的独立 CPU 重算并原子发布有效结果。本次补表再次仅读取 scalar JSONL、完成记录与登记 metadata，复用冻结 `execution_evidence` / `validate`，检查全部 24 条的顺序/身份/控制器/PCA 标量、计数、GT metadata 和已发布均值；验证结果为 46,824 条一致、CUDA 未初始化。没有重新读取图像/mask/权重或运行模型。\n')
    add('补表脚本第一次把只含实际调用数的字典与含 trajectories/offline_updates 元数据的配置字典整体比较，造成报告检查 AssertionError；已改为对相同实际计数字段比较，CPU 补表通过。初次日志保留为 [CPU_REPORT_INITIAL_FAILURE_LOG.txt](CPU_REPORT_INITIAL_FAILURE_LOG.txt)，最终日志为 [CPU_CLOSEOUT_LOG.txt](CPU_CLOSEOUT_LOG.txt)。该问题不涉及原实验、原汇总或冻结运行代码；未重跑 GPU。\n')
    add('- science SHA256：`'+a['binding']['science_sha256']+'`。')
    add('- registration digest：`'+a['binding']['registration_digest']+'`。')
    add('- 公开：执行源码引用、CPU 报告生成脚本、去身份聚合、配对分布、预算证据、图表和本报告。')
    add('- 逐图身份/路径、registration、checkpoint、图像/mask 和原始逐图日志保持私有；公开审计省略设备 UUID/PID。未导出 PCA 特征/向量状态，公开 bank 数值仅为计数、rank、ready 等统计。')
    add('- 自动描述状态为 MIXED、candidate_routes 为空；它不代替外部研究选择，也不产生后续执行许可。\n')
    add('本批实验与结果交付结束，不追加组合、seed、域序、阈值调整或补跑。')
    (OUT/'EXPERIMENT_REPORT.md').write_text('\n'.join(lines)+'\n')
    fig, ax = plt.subplots(figsize=(9,4.2),layout='constrained')
    colors=['#d95f02','#666666','#7570b3','#6699aa','#1b9e77']
    for i,(arm,color) in enumerate(zip(arms[1:],colors)):
        delta=[x-y for x,y in zip(scores(arm),scores('C'))]
        ax.plot([min(delta),max(delta)],[i,i],color=color,lw=2)
        ax.scatter(delta,[i-.12,i-.04,i+.04,i+.12],color=color,s=24)
        ax.scatter([mean(delta)],[i],color=color,s=60,marker='D',edgecolor='white',zorder=3)
    ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(5),arms[1:]);ax.invert_yaxis()
    ax.set_xlabel('Macro Dice change vs C (percentage points)')
    ax.set_title('R1 remaining_dev: four observed orders\nDots: orders; diamond: mean; line: observed range')
    ax.grid(axis='x',alpha=.2)
    for extension in ['png','svg']:fig.savefig(OUT/('order_gains.'+extension),dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    render()
