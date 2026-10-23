"""Aggregate-only receipts/report completion; no image or model access."""
import json, sys
from pathlib import Path

r = Path(sys.argv[1])
e = json.loads((r/'EXIT.json').read_text())
b = json.loads((r/'RUNTIME_RECEIPT.json').read_text())
if e['code'] != 0 or not (r/'TRAINING_FINISHED.json').exists():
    elapsed = e['ended_unix'] - b['origin_unix']
    d = {'status': 'INCOMPLETE_NO_SCIENTIFIC_VERDICT',
         'scope': 'Interrupted source-only observability diagnostic; not a failed scientific gate',
         'observability_gate': None, 'completed_views_verified': None,
         'actual_source_image_reads': None, 'actual_calibration_and_probe_solves': None,
         'neural_model_calls': 0, 'network_training_updates': 0, 'dev_or_target_image_reads': 0,
         'zero_neural_and_target_basis': 'Original implementation has no neural calls and only loads source-fit images',
         'known_synthetic_selfcheck': json.loads((r/'SELF_CHECK.json').read_text()),
         'exit': e, 'budget_audit': {'receipt_wall_seconds': elapsed,
             'recorded_seconds_before_exit_storage_scan': e['seconds'],
             'within_end_to_end_time_budget': elapsed < b['hard_seconds'],
             'worker_stopped_before_deadline': 'UNKNOWN',
             'exit_storage_below_soft_limit': e['allocated_bytes'] < b['soft_stop_bytes']},
         'operations': 'Original process identities retired; post-run result-reader processes observed in RPC wait; causal kill reason unknown',
         'decision': 'Close without replay, clock reset or renamed continuation; preserve observability hypothesis as untested'}
    (r/'REPORT.json').write_text(json.dumps(d,indent=2,allow_nan=False))
    (r/'REPORT_ZH.md').write_text('''# r05 未完成：没有科学门结论

退出码−9，没有完成回执或可用观测排名结果。合成6次求解通过不代表源图像上的观测可识别性；不能把本次未完成写成模型通过或被否证。源图读取数、校准/探针求解数及完成视图数没有可核实的持久计数，保持UNKNOWN；0神经网络调用/训练及0dev/目标图像读取依据原代码的数据入口与执行路径。

原进程均已退役。后续读取进程曾处于rpc_wait_bit_killable；这是当前远端RPC等待的直接证据，不能据此断言此前worker一定因NFS或OOM终止。退出回执reason=null，具体致死原因UNKNOWN。

原始15分钟时钟不重置。按起点到EXIT回执，端到端记录历时%.3f秒，超过900秒硬预算；EXIT自带seconds=%.3f也晚于硬限，且该字段早于退出存储扫描。worker实际停止时刻缺少本地独立回执，不能声称预算合规，也不把回执延迟全部算作已证明的模型计算。退出分配占用%d字节，低于50MB软限和100MB硬限；不声称已连续验证精确峰值。

本轮关闭，不重跑探针、不更换名称接着执行、不追加预算。尚未检验该二次照明+区域均值似然是否区分正确/错误边界，不作Dice/CTTA/论文性能主张。后续先处理预算回执与远端存储可用性，再设计实质独立路线；不能把修控制器当作新的科学结果。

公开源码、原冻结协议、匿名未完成报告与去标识回执。私有配置、图像、逐内容值、权重和第三方PDF未公开。没有启动后继实验。
''' % (elapsed,e['seconds'],e['allocated_bytes']))
    print(json.dumps({'status':d['status'],'budget':d['budget_audit']}))
    sys.exit(0)
d = json.loads((r/'REPORT.json').read_text())
f = json.loads((r/'TRAINING_FINISHED.json').read_text())
assert f['status'] == 'COMPLETE' and e['code'] == 0
c = f['counts']
assert c['completed_probe_views'] == 24 and c['calibration_solves'] == 67 and c['probe_solves'] == 144
assert all(c[k] == 0 for k in ['dev_images_read','target_images_read','neural_model_calls','network_training_updates'])
d['finished'] = f; d['exit'] = e
d['budget_audit'] = {'wall_seconds': e['ended_unix']-b['origin_unix'],
                     'within_time_budget': e['ended_unix']-b['origin_unix'] < b['hard_seconds'],
                     'exit_storage_below_soft_limit': e['allocated_bytes'] < b['soft_stop_bytes'],
                     'exit_storage_below_hard_limit': e['allocated_bytes'] < b['hard_storage_bytes']}
(r/'REPORT.json').write_text(json.dumps(d,indent=2,allow_nan=False))
lines = ['# r05：显式解剖观测的源数据有限可识别性检验','',
         '总体观测门：'+('通过，仅保留该观测假设' if d['observability_gate'] else '失败，拒绝该低容量观测模型')+'。', '',
         '这不是分割网络训练或适应评价。探针候选包含真值，Dice仅用于验证错误几何的扰动量；不能把似然排名当作Dice改善或CTTA支持。', '',
         f"完成12源内容×2视图={d['views']}项，合格内容{d['qualified_contents']}/12；余67源fit内容只校准噪声。0dev/target/神经模型调用/网络训练。", '',
         '|头|平均NLL margin|中位数|5%分位|内容胜/平/负|头门|', '|---|---:|---:|---:|---|---|']
for head,h in d['heads'].items():
    lines.append(f"|{head}|{h['mean_margin']:+.8f}|{h['median_margin']:+.8f}|{h['q05_margin']:+.8f}|{h['content_win_tie_loss']}|{h['ranking_gate']}|")
lines += ['', 'margin=NLL错误解剖−NLL真值，正数有利于真值；每内容平均两视图×该头两种错误。每头需≥9/12内容margin>1e-6、全均值及两视图均值均>1e-6。该探索门不是显著性检验；12内容/24视图不等于独立患者。', '',
          '|头|原图平均margin|光度视图平均margin|收缩候选margin|扩张候选margin|', '|---|---:|---:|---:|---:|']
for head,h in d['heads'].items():
    lines.append(f"|{head}|{h['view_mean_margin']['original']:+.8f}|{h['view_mean_margin']['photometric']:+.8f}|{h['direction_mean_margin'][head+'_SHRINK']:+.8f}|{h['direction_mean_margin'][head+'_EXPAND']:+.8f}|")
lines += ['', '真值在五候选中的名次频数：`'+json.dumps(d['truth_rank_aggregate'])+'`。小于1e-10的分差作排名数值并列，不改主margin门。', '',
          '每个候选都重新拟合同样24个病例系数，固定一次闭式岭解、棋盘偶数像素拟合、奇数像素评分；噪声校准后固定，没有给真值更多拟合预算。空间留出是同一图像内的诊断，不是独立病例泛化。无解剖门模型与候选无关是结构负对照，未重复求解5次来制造检验次数。合成控制只核查实现。', '',
          f"全部求解：合成{c['synthetic_solves']}、源校准{c['calibration_solves']}、探针{c['probe_solves']}，总{c['synthetic_solves']+c['calibration_solves']+c['probe_solves']}。源图像/掩膜各读{c['source_images_read']}，未复制图像或权重。", '',
          f"原时钟至EXIT回执{d['budget_audit']['wall_seconds']:.3f}秒；15分钟硬限、提前60秒停。退出分配占用{e['allocated_bytes']:,}字节，50MB软限/严格100MB硬限。exit0；预算布尔回执：`{d['budget_audit']}`。", '',
          '诊断只检验二次照明+区域均值+固定高斯残差模型，不证明或反证所有生成式潜变量方法。通过也需另立预测初始化与连续更新的分割协议；失败则不加阶数/换噪声/调门槛重试。重复源fit与患者UNKNOWN限制保留，没有外部目标域或论文对比证据。', '',
          '公开源码、冻结协议、匿名聚合与去标识回执；逐内容分数/几何/系数、私有配置路径、数据与第三方PDF不公开。']
(r/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'observability_gate':d['observability_gate'],'heads':{k:v['ranking_gate'] for k,v in d['heads'].items()},'budget':d['budget_audit']}))
