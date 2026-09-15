"""Rebuild public CSV tables and the descriptive report from exported aggregates.

Run with Python 3; no external dependencies or private assets are needed.
"""
import csv
import gzip
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
A = json.loads(gzip.decompress((ROOT / 'aggregate.json.gz').read_bytes()))
E = json.loads((ROOT / 'execution.json').read_text())
P = A['primary_four_order_equal']['remaining_dev']['arms']
S = A['secondary_recurrence']['remaining_dev']['domain_equal_dice_percent']
CONTROLS = dict(T_LR=['T_ISO', 'T_DIAG'], U_PCA=['U_RAND', 'U_SCALE'],
                S_JOINT=['S_SHARED', 'S_NOPCA'], M_TRANSPORT=['M_IDPOST', 'M_SHUFFLE'],
                G_PCA=['G_ISO', 'G_ORDER'])


def write_csv(name, rows):
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (ROOT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, columns)
        writer.writeheader()
        writer.writerows(rows)


def md(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


if __name__ == '__main__':
    assert len(P) == len(S) == 17 and len(A['target']) == 5
    pairs, domains, means, arm_domains = [], [], [], []
    for order, subsets in A['target'].items():
        for subset, data in subsets.items():
            for arm, scores in data['domain_equal_dice_percent'].items():
                means.append(dict(order=order, subset=subset, arm=arm, **scores))
            if subset != 'remaining_dev':
                continue
            for domain, domain_data in data['domains'].items():
                for arm, channels in domain_data['arms'].items():
                    for channel, stats in channels.items():
                        arm_domains.append(dict(order=order, domain=domain, arm=arm, channel=channel,
                            **{'dice_' + k: v for k, v in stats['dice_percent'].items()},
                            **{k: v for k, v in stats.items() if k.startswith('assd_') and not isinstance(v, dict)},
                            **{'assd_conditional_' + k: v for k, v in stats.get('assd_conditional', {}).items()}))
                for pair, channels in domain_data['paired'].items():
                    for channel, stats in channels.items():
                        domains.append(dict(order=order, domain=domain, pair=pair, channel=channel,
                            **stats['dice_delta_pp'],
                            **{k: v for k, v in stats.items() if k != 'dice_delta_pp'}))
            for pair, channels in data['pooled_content_paired'].items():
                for channel, stats in channels.items():
                    pairs.append(dict(order=order, pair=pair, channel=channel, **stats['dice_delta_pp'],
                                      **{k: v for k, v in stats.items() if k != 'dice_delta_pp'}))
    write_csv('all_subset_domain_equal.csv', means)
    write_csv('remaining_domain_arms.csv', arm_domains)
    write_csv('remaining_domain_pairs.csv', domains)
    write_csv('remaining_pooled_pairs.csv', pairs)
    comparisons = []
    for candidate, controls in CONTROLS.items():
        for control in ['C', 'RP'] + controls:
            deltas = [A['target'][str(o)]['remaining_dev']['domain_equal_dice_percent'][candidate]['macro'] -
                      A['target'][str(o)]['remaining_dev']['domain_equal_dice_percent'][control]['macro'] for o in range(4)]
            row = dict(candidate=candidate, control=control, delta_pp=P[candidate]['macro']-P[control]['macro'],
                       positive_orders=sum(x > 0 for x in deltas), worst_same_order_delta_pp=min(deltas),
                       secondary_delta_pp=S[candidate]['macro']-S[control]['macro'],
                       **{'order%d_delta_pp' % i: v for i, v in enumerate(deltas)})
            assert abs(mean(deltas) - row['delta_pp']) < 1e-10
            comparisons.append(row)
    write_csv('candidate_comparisons.csv', comparisons)
    raw_costs = list(csv.DictReader((ROOT / 'costs_by_trajectory.csv').open()))
    costs = []
    for arm in P:
        rows = [c for c in raw_costs if c['arm'] == arm]
        assert len(rows) == 5
        costs.append(dict(arm=arm, mean_worker_minutes=mean(float(c['worker_seconds']) for c in rows)/60,
                         mean_host_ms_per_record=sum(float(c['host_seconds']) for c in rows)/9755*1000,
                         mean_pipeline_ms_per_record=sum(float(c['pipeline_seconds']) for c in rows)/9755*1000,
                         forward_per_record=sum(int(c['network_forwards']) for c in rows)/9755,
                         total_VJP=sum(int(c['jacobian_vjp_calls']) for c in rows),
                         peak_allocated_MiB=max(int(c['peak_allocated_bytes']) for c in rows)/2**20,
                         peak_active_bank_KiB=max(int(c['peak_bank_state_bytes']) for c in rows)/1024,
                         peak_context_KiB=max(int(c['peak_context_tensor_bytes']) for c in rows)/1024))
    write_csv('costs_by_arm.csv', costs)
    risk_rows = []
    for candidate in CONTROLS:
        pooled = [r for r in pairs if r['order'] != '4' and r['pair'] == candidate+'-C' and r['channel']=='macro']
        total = sum(r['n'] for r in pooled)
        assert total == 1695 * 4
        risk_rows.append([candidate, f"{sum(r['positive'] for r in pooled)/total*100:.2f}",
                          f"{sum(r['zero'] for r in pooled)/total*100:.2f}",
                          f"{sum(r['negative'] for r in pooled)/total*100:.2f}",
                          f"{mean(r['median'] for r in pooled):+.3f}",
                          f"{mean(r['worst_decile_mean'] for r in pooled):+.3f}",
                          f"{min(r['minimum'] for r in pooled):+.3f}"])
    assd_rows = []
    for candidate in CONTROLS:
        for channel in ['OD', 'OC']:
            rows = [r for r in pairs if r['order'] != '4' and r['pair']==candidate+'-C' and r['channel']==channel]
            valid = sum(r['assd_common_valid'] for r in rows)
            delta = sum(r['assd_delta_mean_px'] * r['assd_common_valid'] for r in rows)/valid
            assd_rows.append([candidate, channel, valid, sum(r['assd_not_jointly_defined'] for r in rows),
                              sum(r['assd_left_undefined'] for r in rows), sum(r['assd_right_undefined'] for r in rows),
                              f'{delta:+.4f}'])
    domain_rows = []
    for candidate in CONTROLS:
        for domain in A['target']['0']['remaining_dev']['domains']:
            values = []
            for channel in ['OD', 'OC', 'macro']:
                values.append(mean(r['mean'] for r in domains if r['order'] != '4' and
                                   r['pair']==candidate+'-C' and r['domain']==domain and r['channel']==channel))
            domain_rows.append([candidate, domain] + [f'{v:+.3f}' for v in values])
    method_rows = []
    for arm, prefix in [('T_LR','r3.teacher.'), ('U_PCA','r3.update.'), ('M_TRANSPORT','r3.transport.'), ('G_PCA','r3.graph.')]:
        index = list(P).index(arm)
        for scope, orders in [('primary',range(4)), ('secondary',[4])]:
            keys = set().union(*(A['mechanism'][f'o{o}a{index}'].keys() for o in orders))
            for key in sorted(keys):
                if not key.startswith(prefix) or '.evidence.' in key:
                    continue
                values = [A['mechanism'][f'o{o}a{index}'][key] for o in orders if key in A['mechanism'][f'o{o}a{index}']]
                n = sum(v['n'] for v in values)
                method_rows.append(dict(arm=arm, scope=scope, key=key, n=n,
                                        mean=sum(v['mean'] * v['n'] for v in values)/n))
    write_csv('mechanism_means.csv', method_rows)
    report = '''# R3 五框架冻结筛选：实验完成报告

**R3_EXPERIMENT_COMPLETE**。85/85 条完整轨迹、165,835 条正式评分记录；两个机械 smoke 和 85 个正式进程全部退出码 0。独立 CPU 标量汇总已完成并置 valid=true；公开导出再次核对逐轨迹记录与物理计数，全部匹配。

结论：本冻结配置下，没有主框架提供清晰、可归因于新增机制的增益。按预先给定的选择规则保留 C，归档五个主框架及全部匹配控制，不晋级候选、不追加组合、种子或调参实验。这是开发证据的描述性决定，不是外部代码 review 通过结论。

## 完成时间、身份及执行历史

- 北京时间 2026-09-14 13:52:46 开始，2026-09-15 07:56:02 结束；含前后处理约 18 小时 3 分 16 秒。监督器墙钟 18.033 小时，总 active-worker 34.887 小时。
- 物理 GPU 6、7，最多 2 个 worker，RTX 3090。没有因中间效果跳过任何臂。
- 实际运行 implementation SHA：`185fd440b16920f367c1f5e3096ab495bd85c0ec`。
- science 文件 SHA256：`73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e`。
- registration digest：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`。
- secondary stream digest：`cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`。
- 生命周期：阶段 I 实现 → 环境修复 d601496a 获用户提供的外部 review → 第一次 GPU 尝试在首个 forward 前进程审计失败、停止 → 授权工程修复 185fd440 → 用户明确免除新增 review 并授权直接执行 → 双 smoke → 85 条正式轨迹 → COMPUTE_COMPLETE → 独立 CPU 标量验证 → R3_EXPERIMENT_COMPLETE → 本次公开交付。
- 原外部 review 对应 d601496a；用户的执行授权不等于修复 SHA 获得了新的外部 review。历史失败保留在 [首次尝试报告](../../review/r3/stage2_attempt1/REPORT.md)，未混入效果表。该失败尝试额外 2 次模型加载、0 forward/Adam/target 记录，active-worker 约 6.615 秒。

## 终点与完整 17 臂结果

主终点 remaining_dev：1,695 内容/轨迹；每图 OD/OC 平均，域内平均，四域等权，再四主序等权。Dice 单位 %，差值为百分点 pp。回访流单列，不做五流合并。Drishti_GS 仅 37 内容却占每序 25% 域权重；四序共享内容、固定同一 seed=20260907，不能作独立患者重复或显著性检验。

'''
    report += md(['臂','OD','OC','主四序 macro','ΔC pp','独立回访 macro','回访 ΔC pp'],
                 [[arm,f"{x['OD']:.3f}",f"{x['OC']:.3f}",f"{x['macro']:.3f}",
                   f"{x['macro']-P['C']['macro']:+.3f}", f"{S[arm]['macro']:.3f}",
                   f"{S[arm]['macro']-S['C']['macro']:+.3f}"] for arm,x in P.items()])
    report += '\n\n## 五组候选与匹配控制\n\n'
    report += md(['候选','对照','主 Δpp','正向主序 /4','最差同序 Δpp','回访 Δpp'],
                 [[r['candidate'],r['control'],f"{r['delta_pp']:+.6f}",r['positive_orders'],
                   f"{r['worst_same_order_delta_pp']:+.3f}",f"{r['secondary_delta_pp']:+.3f}"] for r in comparisons])
    report += '''

预设 +0.5pp 且至少 3/4 主序正向是筛选参考，不是统计显著性或临床门槛；没有候选满足该增益参考线。最差同序 Δ 是 min(candidate_order − control_order)，不等于两个臂各自最差序之差。

- **T_LR：unsupported_configuration。** 比 C 低 1.401pp，比 RP 低 1.570pp；低于 T_ISO。虽高于 T_DIAG，不能支持低秩密度优于简单控制的统一收益。
- **U_PCA：prefer_simple_control（组内），不晋级。** 低于 C、RP、U_RAND 和 U_SCALE；单纯匹配步长控制 U_SCALE 接近 C，增加 VJP 没有换来收益。
- **S_JOINT：unsupported_configuration。** 比 C 低 1.071pp，明显低于 S_SHARED；比 S_NOPCA 仅高 0.045pp，回访流反而低于 S_NOPCA。实际有状态创建和切换，仍未带来性能提升。
- **M_TRANSPORT：prefer_simple_control（组内），不晋级。** 比 C 高 0.166pp，但比 RP 低 0.002929pp，比 M_IDPOST 低 0.003557pp。优于 M_SHUFFLE 说明打乱配对有害，不能证明正确迁移相对恒等 POST 有价值。M_IDPOST 的全表最高分比 RP 仅高 0.000628pp，不作实质优势主张。
- **G_PCA：prefer_simple_control（组内），不晋级。** 低于 C、RP、G_ISO、G_ORDER；图教师的机械变化不等于有效性。

## 各域与尾部风险

下表为候选相对 C 的四主序平均域内差值，REFUGE_Valid OC 单列在 OC 列。各候选相对 RP/两直接控制的逐序、逐域 OD/OC/宏指标及分母见 CSV；不据域标签拼接不同方法。

'''
    report += md(['候选','域','OD Δpp','OC Δpp','macro Δpp'],domain_rows)
    report += '\n\n主四序合计 6,780 个内容-顺序配对（不是独立患者）：正/零/负比例按内容计数汇总；中位数、最差 10% 均值是各序对应统计量的平均，最差单内容取所有序最小值。这些是样本权重的风险诊断，不能替代上面的四域等权主均值。\n\n'
    report += md(['候选−C','正 %','零 %','负 %','配对中位数均值 pp','最差10%均值 pp','最差内容 pp'],risk_rows)
    report += '\n\n## ASSD（共同有效配对）\n\n以下按四序共同有效内容配对汇总，距离单位为评估栅格像素，负值较好；undefined 不填 0。valid/缺失数计内容-顺序，不计独立内容。完整逐域、逐序的中位数、上侧最差 decile、条件 ASSD 和全部 20 组对照见 CSV/完整 JSON。通用 distribution 的 `worst_decile_mean` 总是下侧 decile；用于绝对 ASSD 时代表较小距离，不能误称最差距离。\n\n'
    report += md(['候选−C','通道','共同有效','非共同有效','候选未定义','C 未定义','均值 Δpx'],assd_rows)
    report += '\n\n## 实际资源开销\n\n正式：1,385,210 次 network forward、165,835 次 loss backward/Adam、231,784 次 Jacobian VJP、32,851 次参数替换；VJP 低于 234,120 上限。两个 smoke 共 632 forward、76 backward/Adam、36 VJP、10 次参数替换。正式加 smoke 共 1,385,842 forward、165,911 backward/Adam、231,820 VJP。\n\n下表包含五条流的平均单轨迹墙钟与主机端耗时；耗时受共享设备及不同时段负载影响，不能当作受控速度基准。peak allocated 为 PyTorch tensor 分配峰值，不是 nvidia-smi 总显存或 reserved；bank 仅记录 active bank 张量，context 为已记录上下文张量，不能当作完整模型/优化器/进程 RSS。C/RP bank=0 表示没有该 R3 计数字段，不表示没有状态。\n\n'
    report += md(['臂','轨迹 min','host ms/图','forward/图','VJP 总数','allocated MiB','active bank KiB','context KiB'],
                 [[c['arm'],f"{c['mean_worker_minutes']:.2f}",f"{c['mean_host_ms_per_record']:.1f}",
                   f"{c['forward_per_record']:.0f}",c['total_VJP'],f"{c['peak_allocated_MiB']:.1f}",
                   f"{c['peak_active_bank_KiB']:.2f}",f"{c['peak_context_KiB']:.2f}"] for c in costs])
    report += '''

完整 pipeline 时间、逐轨迹 host/wall/物理调用/状态载入开销见 costs CSV。机制细分计时（密度、Jacobian、SVD/transport、graph）保留在 aggregate 的 mechanism。没有独立的全进程 host RSS 峰值记录，不补造。有效性未胜 C 的候选无法凭额外开销获得晋级；M 的微小 C 增益由简单控制覆盖，不能建立新的性能-成本优势。

## 回访、机制与限制

回访流仍为 1,951 个不同内容，回访的是环境，不重复旧图。全 17 臂都已运行该流，五个主候选均低于 C。S_JOINT 在 stream4 实际建立 3 个 slot、切换 289 次、overflow 3 次，参数装载/替换 290 次（含初始装载）；累计记录载入约 36.459 秒，上下文张量峰值 999,128 bytes。S_SHARED 同样发生路由切换却不替换参数，效果优于 S_JOINT；不能把“存在复用”写成“复用有效”。完整逐域 slot 访问次数与连续域块统计在 contexts.json；slot 标签只用于事后汇总。

全部已保存机制分布在 aggregate.json.gz，主/次流加权标量均值在 mechanism_means.csv。举例，主序 0：T_LR 教师 ready 约 99.18%，平均绝对 logit 修正约 1.310；U_PCA 实际改变更新、平均 cosine 约 0.99593；M_TRANSPORT 的平均 ||Q−I|| 约 0.000959，拟合残差从 2.797e−6 降至 2.134e−6；G_PCA ready 时 1,984 条有效边，图能量均值 3.246→3.136，教师包含违反归零。这些证明路径有被执行，不能代替效果或建立因果归因。

未保存可支持真实教师 q/q* 对 GT 纠错/误纠正、完整 feature covariance、Jacobian、图几何、ASSD 几何重建的材料；本次 CPU 汇总不声称重建它们，也没有增加 GPU 诊断轮。未访问源数据、源代理或源原型，未重训源模型。没有独立新患者验证，四序共享内容不能支撑患者层面的统计显著性。结论限于该 checkpoint、冻结配置和开发流，不外推为五类思想普遍无效。

实际正式 backend：Python 3.10.6、Torch 2.2.1+cu121、CUDA 12.1、cuDNN 8902；TF32 off，cuDNN benchmark off / deterministic on；正式 deterministic_algorithms=false、warn_only=false、CUBLAS_WORKSPACE_CONFIG=null。smoke 的同设备比较实际进入严格确定性上下文，完整 before/after 记录见 execution.json；固定 seed 不是正式轨迹位级确定性的证明。

## 复现与证据索引

- [实际实现](https://github.com/DLwbm123/DPA-CTTA/tree/185fd440b16920f367c1f5e3096ab495bd85c0ec)、[冻结科学配置](../../../configs/r3_science_v1.json)、[实验方案](../../review/r3/03_EXPERIMENT_PLAN.md)、[方法规范](../../review/r3/02_METHOD_SPEC.md)、[方法来源](../../review/r3/METHOD_PROVENANCE.md)、[状态生命周期](../../review/r3/STATE_LIFECYCLE.md)、[预设选择标准](08_RESULT_SELECTION.md)。
- [执行/退出/85 完成凭据及两个 smoke 真实 traces](execution.json)、[CPU 导出真实日志](export_cpu.log)、[导出脚本](export_scalars.py)。原独立 CPU 分析器位于上述实现的 src/dpa_ctta/r3/analyze.py；完成标志由其核验后原子发布。
- [完整 17 臂主/次表](primary_and_secondary.csv)、[四个 subset、五序域等权结果](all_subset_domain_equal.csv)、[候选对照及顺序风险](candidate_comparisons.csv)。
- [remaining_dev 全臂各域分布/ASSD](remaining_domain_arms.csv)、[各域配对](remaining_domain_pairs.csv)、[逐序内容配对](remaining_pooled_pairs.csv)。
- [逐臂资源](costs_by_arm.csv)、[85 轨迹资源](costs_by_trajectory.csv)、[回访与 slot](contexts.json)、[机制均值](mechanism_means.csv)。
- [完整去身份聚合结果 gzip](aggregate.json.gz)：全部四个 subset、五流、逐域/逐序/配对的完整分布及标量机制；标准 gzip JSON，可直接用 Python gzip.open/json.load 读取。
- [本报告与 CSV 重建脚本](build_tables.py)。在此目录执行 `python3 build_tables.py`，只使用公开导出，无第三方依赖；内置 17 臂/5 流、配对加权和 85 轨迹覆盖断言。实际运行检查见 [本地构表日志](tables_cpu.log)。

发布范围为代码、冻结配置、去身份聚合指标、过程证据和脚本；不发布原始 RGB/mask、逐内容记录/文件名、checkpoint、设备 UUID、服务器路径和私有授权材料。既有科学配置、main、历史结果均保留。本批已结束，后续执行授权为 false；不自行追加实验或产生新的外部 review 通过结论。
'''
    (ROOT / 'REPORT.md').write_text(report)
    print('PASS 17 arms / 5 streams; 20 primary candidate comparisons; 85 cost rows.')
    print('PASS primary deltas equal the four per-order mean differences.')
    print('PASS risk denominators 6780 per candidate; no five-stream endpoint.')
