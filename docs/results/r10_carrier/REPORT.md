# R10_CARRIER_DECISION_3H_V1 — 未完成收束

本轮终态为 **PREFLIGHT_INCOMPLETE**，不是完整载体去留实验完成。初始 T0 为北京时间 2026-09-30 22:16:06.144414；30 分钟预检截止为 22:46:06.144414，未重置或延长。

## 实际执行与停止原因

1. 第一版源验证在第一幅 C0 访问后因 `visit` 日志字段重复而失败，完成 1 次前向、0 个完整 episode；该失败墙钟 16.643 秒。修复仅改日志装配，保留原模型、输入、状态递推和读出定义。第一版配置、日志、回执与报告已在私有版本目录保留。
2. 第二版重新验证通过，但监督器在取得运行锁之前退出：旧监督器已 inactive，持久化锁仍绑定第一版配置摘要，新配置被 `owner identity/host changed` 拒绝。此前写出的“RUNNING”启动说明只是发出了启动进程，未完成运行确认，现已纠正。
3. 对锁记录进行处理前，30 分钟预检截止已过；截止断言阻止新的启动，未移动旧锁或发起第三次启动。按输入计划 §4/§10 收束，不利用剩余总预算延长预检阶段。

本轮进程已全部退出。没有目标在线过程、目标评分、模型训练、优化器更新或 VJP/JVP；旧小时监测已按用户要求删除（MONITOR_CANCELLATION.json），没有新建或恢复小时监测，没有 GitHub push。

## 已取得的证据

7 项局部检查通过；两个版本各完成 32 个固定源验证访问的零调制检查，四种实际模式各 8 个。最终复验中，C0 同路径重复前向、B_ZERO 相对 C0 的 logits/概率最大差均为 0，硬掩码不一致为 0；参数和 BN buffers 不变；alpha=1 精确恢复原生 B；相同历史条件的不同读出 alpha 保持内部状态相同。

这些是源端预检结果。8 条新目标轨迹均 NOT_RUN；没有验证本轮 B_ZERO 目标概率摘要与旧 C0 的一致性。五组完整源验证均未完成，源/目标收益方向和尺度×历史交互均 unavailable。不能据此封存或保留 B，也不能把缺失指标当作零或负结果。

旧 N/G/C0/B 的 8 条短流完成记录经身份、内容、顺序、资格及评分封存核对后复用，作为已有证据；下附表明确标出 REUSED，不能视为本轮新增成果。旧模型、检查点、完整源/目标结果未改动。

## 资源和交付

截至收尾回执墙钟 2067.406 秒（包含读取、实现、复验、失败启动及匿名收集）；GPU-worker 107.994 秒；实际 961 次前向、0 backward、0 optimizer、0 VJP。961=两次预检各 480 次前向，加第一版源验证的 1 次前向。源端失败成本已计入，没有因修复清零。先前预算包封闭收费 24757.142 秒与本轮单独记录，GPU 时间不重复累加。

执行第二版 SHA `da5edaf9714a1c1ee569958c6e4f333189bc8e7a`；终态汇总代码另作本地提交，修复未完成矩阵被误判为负性能结论的报告逻辑。冻结配置、ZERO_PARITY、SOURCE_COMPARISON、TARGET_FACTORIAL、DOMAIN_CHANNEL、资源和真实回执一并匿名交付；未完成项保留空值及状态。私有输入、标识、预测、凭据和服务器路径未发布。

本轮只本地提交。没有自动续跑、新增种子/alpha/训练或下一阶段授权。

---

以下为终态统一生成的表及受限分析：

# R10_CARRIER_DECISION_3H_V1
Status: PREFLIGHT_INCOMPLETE; execution SHA da5edaf9714a1c1ee569958c6e4f333189bc8e7a.
New wall including preflight: 1900.26s; GPU-worker 107.99s; recovery False.
No training, optimizer update or VJP/JVP. Prior baselines are exact paired reuse. All new target scores unblinded only after terminal state.
One seed; two correlated orders, same contents. Unknown patient dependence; no independent-seed confidence interval. Source val is development evidence.

| condition | order | status/origin | OD % | OC % | domain macro % |
|---|---:|---|---:|---:|---:|
| B_ZERO | 0 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_ZERO | 1 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_RESET_1 | 0 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_RESET_1 | 1 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_FULL_READOUT_025 | 0 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_FULL_READOUT_025 | 1 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_RESET_READOUT_025 | 0 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| B_RESET_READOUT_025 | 1 | NOT_RUN_PREFLIGHT_INCOMPLETE/NEW | MISSING | MISSING | MISSING |
| C0 | 0 | REUSED_COMPLETE/REUSED | 83.347552 | 66.280674 | 74.814113 |
| C0 | 1 | REUSED_COMPLETE/REUSED | 83.347552 | 66.280674 | 74.814113 |
| B_FULL_1 | 0 | REUSED_COMPLETE/REUSED | 82.894093 | 62.260532 | 72.577313 |
| B_FULL_1 | 1 | REUSED_COMPLETE/REUSED | 82.892598 | 62.263494 | 72.578046 |
| N | 0 | REUSED_COMPLETE/REUSED | 78.833776 | 57.508616 | 68.171196 |
| N | 1 | REUSED_COMPLETE/REUSED | 78.833776 | 57.508616 | 68.171196 |
| G | 0 | REUSED_COMPLETE/REUSED | 85.032165 | 67.580112 | 76.306138 |
| G | 1 | REUSED_COMPLETE/REUSED | 84.952175 | 67.282234 | 76.117205 |

## Three bounded questions
1. Zero FiLM parity: source same-path repeat criterion, BN/parameter checks, compatible target probability seals and limits are in ZERO_PARITY.json.
2. Scale and history: fixed paired contrasts and the full-minus-reset scaling interaction are in TARGET_FACTORIAL.csv, including every domain and OD/OC. Norms describe actual injected increments; near-zero feature norms are flagged in private source/target traces. Scaling improvements below C0 are reduced modulation harm, not additional adaptation benefit.
3. Source/target direction: SOURCE_TARGET_DIRECTION.csv flags sign reversal without claiming which observer/basis/calibration/simulator component caused it. Source mode and target domain means are different populations.

```json
{
  "matrix_complete": false,
  "zero_parity_source": true,
  "zero_target_seals": [],
  "history_original_scale": [
    null,
    null
  ],
  "scale_FULL": [
    null,
    null
  ],
  "scale_RESET": [
    null,
    null
  ],
  "signals": {
    "B_RESET_1": {
      "delta_vs_C0": [
        null,
        null
      ],
      "meets_retest_scale": null
    },
    "B_FULL_READOUT_025": {
      "delta_vs_C0": [
        null,
        null
      ],
      "meets_retest_scale": null
    },
    "B_RESET_READOUT_025": {
      "delta_vs_C0": [
        null,
        null
      ],
      "meets_retest_scale": null
    }
  },
  "source_target_opposite_sign": [],
  "decision": "Unavailable: fixed source/target matrix incomplete; no carrier decision or negative performance finding.",
  "automatic_followon_authorized": false
}
```
The +0.5pp relative-C0 priority threshold is descriptive R&D, not statistical significance or clinical benefit. No extra alpha, training, seed or monitoring is authorized.
Old attribution/SUP/writer evidence is retained; its small future numerical effect did not establish positive target benefit. SUP_STATIC still uses the frozen B action/basis path; RET-versus-STATIC is not a writer-only experiment.
Last-quarter and switch-first32 records are composition-sensitive descriptions, not revisit/forgetting claims. Missing diagnostic quantities are NOT_RECORDED, never zero.
Anonymous summaries are intended for local commit only; private raw images/masks/probabilities, content IDs, credentials and host paths are not public artifacts.
