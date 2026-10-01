# R10_CARRIER_DECISION_3H_V1 — 完成报告

本轮于北京时间 **2026-10-01 11:36:39 完成**。五个固定源验证控制、16 个 episode、2560 次访问全部完成；8 条新增目标轨迹各 1024 次到达、888 条主评分全部在线封存并独立评分，总计 8192 次到达、7104 条新增主评分。8 份概率退休回执齐全，本轮进程已退出。旧 N/G/C0/B 的 8 条短流按原身份和封存结果复用。

续跑期间无新失败、无恢复。此前日志字段错误和配置锁拒绝保留在私有版本目录与公开 history/pre_renewal，既有失败计费未清零。运行代码 `1084fe1c9ffc030388fbe4c6ebecbd80b1bf64a5`；续跑冻结配置 `746c648a338cca94d48f57211068c43c4de3bc8bd825a9c2bdf653c51bf7d825`。本轮无模型训练、backward、optimizer 更新或 VJP/JVP。

## 三个回答

1. **关闭 FiLM 是否恢复 C0：是，在已登记的两条短流上。** 固定源端比较的 logits/概率差和硬掩码不一致均为零；两条目标 B_ZERO 的整条 little-endian float32 sigmoid 概率封存摘要及字节数分别与旧 C0 一致。旧概率已退休，因此这里依赖兼容封存摘要，没有额外声称逐字节重读比较。B_ZERO 与 C0 的全部已封存指标一致。
2. **当前强度、历史还是交互：两种固定干预都减轻损伤，并存在描述性交互。** 逐图 reset、alpha=1 相对 FULL、alpha=1 的两 order 域等权平均改善 1.008756pp；仅将 FULL 读出 alpha 降至 0.25 改善 1.922822pp；RESET 下同样降至 0.25 改善 1.046179pp，缩放效应之差约 0.876644pp。相同历史条件不同 alpha 的新目标内部状态在全部 1024 次访问逐次一致，且 FULL 0.25 在旧 B 仅有的 visit1000 快照处一致。源端各 512 次访问的配对状态也一致。可以解释为输出读出干预，不能把这些均值当作某单一机制的因果确认。
3. **源/目标方向是否相反：是。** 原 B_FULL_1 相对 C0，源端 hard Dice 四模式等权为 +1.144635pp，目标端四域等权平均为 -2.236434pp。三个固定非零控制相对 C0 也均为源端正、目标端负。reset/缩放相对原 B 的源端宏指标略降、目标端提高。这支持源到目标迁移不匹配的研发假设，不能据此确定 observer、basis、校准或模拟器哪一项是根因。

## 同口径主结果

目标指标为两个相关 order 各自四域/OD/OC 等权 hard Dice 的平均；pooled 指标只作次要结果，未用于换主指标或选赢家。

| 条件 | 来源 | 目标 Dice % | 相对 C0 pp | 源验证 Dice % |
|---|---|---:|---:|---:|
| N | 复用 | 68.171196 | -6.642917 | unavailable |
| G | 复用 | 76.211671 | +1.397558 | unavailable |
| C0 | 复用 | 74.814113 | +0.000000 | 86.694501 |
| B_ZERO | 新增 | 74.814113 | +0.000000 | C0 alias |
| B_FULL_1 | 复用 | 72.577679 | -2.236434 | 87.839136 |
| B_RESET_1 | 新增 | 73.586435 | -1.227678 | 87.742959 |
| B_FULL_READOUT_025 | 新增 | 74.500501 | -0.313612 | 87.084121 |
| B_RESET_READOUT_025 | 新增 | 74.632613 | -0.181500 | 86.994414 |

最佳固定非零条件 B_RESET_READOUT_025 仍低于 C0 0.181500pp；FULL 0.25 低 0.313612pp。三个非零条件均未显示相对 C0 的正效用，均不满足预声明的平均 ≥+0.5pp、两 order 均为正的复验优先尺度。改善原 B 是减少损伤，不能称为超出 C0 的适应创新收益。按计划封存当前固定 B＋actor 配方，本轮不再加训练/种子/alpha，也不自动启动下一方案。GraTa 的复用域等权平均为 76.211671%，仍是本开发集系统参考。

## 最差域/通道与限制

- B_FULL_1：相对 C0 最差为 order 0、Drishti_GS/OC，-13.212627pp，主评分 n=19。
- B_RESET_1：相对 C0 最差为 order 0、Drishti_GS/OC，-9.078425pp，主评分 n=19。
- B_FULL_READOUT_025：相对 C0 最差为 order 0、Drishti_GS/OC，-2.674928pp，主评分 n=19。
- B_RESET_READOUT_025：相对 C0 最差为 order 0、Drishti_GS/OC，-1.872784pp，主评分 n=19。

Drishti_GS 每 order 仅 19 个主评分样本，不能把这些退化当作独立患者统计。仅一个策略种子、两个相同内容的相关 order，不提供跨训练随机性的置信区间。source-val 已参与研发，不是独立验证集。无回访流，不制造回访/遗忘结论。切换后前32次和最后四分之一的描述受样本构成影响。全部域/OD/OC/order、配对分布、固定因子交互、源/目标方向及实际增量范数均保留于配套表。

## 资源、历史和本地交付

包含本次检查/修复/启动的续跑封闭墙钟 2887.634 秒（约 48 分 08 秒）；加上前一次已关闭尝试 2067.406 秒，累计已执行收费 4955.039 秒（1 小时 22 分 35 秒），在累计 3 小时额度内。旧包 prior charge 24757.142 秒，加本轮累计得到 29712.182 秒，仍低于原 12 小时包。GPU-worker 合计 2200.448 秒，包含旧失败，不重复加到账墙钟。物理计数 21953 forwards、0 backward/optimizer/VJP；输出占用 253698746 字节。

原 T0 及其截止完整保留；后续用户明确授权修复并开始，登记续跑窗口，保留已发生费用及失败。两次已关闭执行之间的空档、终态后等待本次收集的时间，与执行收费分开记录。自动报告在本次续跑截止内生成；目前仅收集、核对并本地提交，不把等待时间伪称 GPU 计算，也未再次启动进程。

按原计划只本地提交，不 push，不恢复小时监测。SOURCE_COMPARISON、TARGET_FACTORIAL、DOMAIN_CHANNEL、ZERO_PARITY、DIAGNOSTICS、SOURCE_TARGET_DIRECTION、DECISION、匿名配置、资源、封存/评分/退休/完成回执均已交付。未发布私有输入、患者标识、掩码、概率、凭据或服务器路径。原模型和 checkpoint 未改动，原始允许保存的标量/状态日志继续保留于私有执行目录。

---

以下保留自动终态报告及逐 order 表：

# R10_CARRIER_DECISION_3H_V1
Status: COMPLETE; execution SHA 1084fe1c9ffc030388fbe4c6ebecbd80b1bf64a5.
New wall including preflight: 4955.04s; GPU-worker 2200.45s; recovery False.
No training, optimizer update or VJP/JVP. Prior baselines are exact paired reuse. All new target scores unblinded only after terminal state.
One seed; two correlated orders, same contents. Unknown patient dependence; no independent-seed confidence interval. Source val is development evidence.

| condition | order | status/origin | OD % | OC % | domain macro % |
|---|---:|---|---:|---:|---:|
| B_ZERO | 0 | COMPLETE/NEW | 83.347552 | 66.280674 | 74.814113 |
| B_ZERO | 1 | COMPLETE/NEW | 83.347552 | 66.280674 | 74.814113 |
| B_RESET_1 | 0 | COMPLETE/NEW | 83.230975 | 63.941895 | 73.586435 |
| B_RESET_1 | 1 | COMPLETE/NEW | 83.230975 | 63.941895 | 73.586435 |
| B_FULL_READOUT_025 | 0 | COMPLETE/NEW | 83.267187 | 65.734073 | 74.500630 |
| B_FULL_READOUT_025 | 1 | COMPLETE/NEW | 83.267011 | 65.733735 | 74.500373 |
| B_RESET_READOUT_025 | 0 | COMPLETE/NEW | 83.328065 | 65.937161 | 74.632613 |
| B_RESET_READOUT_025 | 1 | COMPLETE/NEW | 83.328065 | 65.937161 | 74.632613 |
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
  "matrix_complete": true,
  "zero_parity_source": true,
  "zero_target_seals": [
    {
      "order": 0,
      "compatible_dtype": "little endian float32 sigmoid probabilities",
      "same_predictions": true,
      "same_bytes": true,
      "zero_prediction_sha256": "5211ec6e1846377b114ffdeb86a8ada69d2642f9e1b6209942df7ec03e434d05",
      "C0_prediction_sha256": "5211ec6e1846377b114ffdeb86a8ada69d2642f9e1b6209942df7ec03e434d05"
    },
    {
      "order": 1,
      "compatible_dtype": "little endian float32 sigmoid probabilities",
      "same_predictions": true,
      "same_bytes": true,
      "zero_prediction_sha256": "674ece50a5c1c6253821228acc0710ceae4cf060c510103c126097baa9fa919b",
      "C0_prediction_sha256": "674ece50a5c1c6253821228acc0710ceae4cf060c510103c126097baa9fa919b"
    }
  ],
  "history_original_scale": [
    0.010091223414099283,
    0.010083889719374482
  ],
  "scale_FULL": [
    0.019233173761899047,
    0.019223272187516162
  ],
  "scale_RESET": [
    0.010461785429574,
    0.010461785429574
  ],
  "signals": {
    "B_RESET_1": {
      "delta_vs_C0": [
        -0.012276784342611436,
        -0.012276784342611436
      ],
      "meets_retest_scale": false
    },
    "B_FULL_READOUT_025": {
      "delta_vs_C0": [
        -0.0031348339948116723,
        -0.0031374018744697563
      ],
      "meets_retest_scale": false
    },
    "B_RESET_READOUT_025": {
      "delta_vs_C0": [
        -0.0018149989130374355,
        -0.0018149989130374355
      ],
      "meets_retest_scale": false
    }
  },
  "source_target_opposite_sign": [
    {
      "left": "B_RESET_1",
      "right": "B_FULL_1",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": -0.002785059345981944,
      "target_four_domain_delta": 0.01681363011943232,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "B_FULL_1",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": -0.002785059345981944,
      "target_four_domain_delta": 0.016784015727711397,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "B_FULL_1",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": -0.0009617727648413288,
      "target_four_domain_delta": 0.010091223414099283,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "B_FULL_1",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": -0.0009617727648413288,
      "target_four_domain_delta": 0.010083889719374482,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "B_FULL_1",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": -0.01555772650266618,
      "target_four_domain_delta": 0.03473540996159371,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "B_FULL_1",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": -0.01555772650266618,
      "target_four_domain_delta": 0.034702414889302126,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "B_FULL_1",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": -0.007550147079564473,
      "target_four_domain_delta": 0.019233173761899047,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "B_FULL_1",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": -0.007550147079564473,
      "target_four_domain_delta": 0.019223272187516162,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "B_RESET_1",
      "channel": "OD",
      "order": 0,
      "source_four_mode_delta": -0.00033163741066410957,
      "target_four_domain_delta": 0.0009709069924116958,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "B_RESET_1",
      "channel": "OD",
      "order": 1,
      "source_four_mode_delta": -0.00033163741066410957,
      "target_four_domain_delta": 0.0009709069924116958,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "B_RESET_1",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": -0.01463924473899525,
      "target_four_domain_delta": 0.019952663866736307,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "B_RESET_1",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": -0.01463924473899525,
      "target_four_domain_delta": 0.019952663866736307,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "B_RESET_1",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": -0.007485441074829624,
      "target_four_domain_delta": 0.010461785429574,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "B_RESET_1",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": -0.007485441074829624,
      "target_four_domain_delta": 0.010461785429574,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_1",
      "right": "C0",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": 0.023219812578895382,
      "target_four_domain_delta": -0.040201422138777664,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_1",
      "right": "C0",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": 0.023219812578895382,
      "target_four_domain_delta": -0.04017180774705674,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_1",
      "right": "C0",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": 0.011446348699606479,
      "target_four_domain_delta": -0.02236800775671072,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_1",
      "right": "C0",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": 0.011446348699606479,
      "target_four_domain_delta": -0.022360674061985917,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "C0",
      "channel": "OD",
      "order": 0,
      "source_four_mode_delta": 0.0005343986366168618,
      "target_four_domain_delta": -0.0011657766658775374,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "C0",
      "channel": "OD",
      "order": 1,
      "source_four_mode_delta": 0.0005343986366168618,
      "target_four_domain_delta": -0.0011657766658775374,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "C0",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": 0.02043475323291344,
      "target_four_domain_delta": -0.02338779201934534,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "C0",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": 0.02043475323291344,
      "target_four_domain_delta": -0.02338779201934534,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "C0",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": 0.01048457593476515,
      "target_four_domain_delta": -0.012276784342611436,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_1",
      "right": "C0",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": 0.01048457593476515,
      "target_four_domain_delta": -0.012276784342611436,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "C0",
      "channel": "OD",
      "order": 0,
      "source_four_mode_delta": 0.0001303171638546985,
      "target_four_domain_delta": -0.0008036558124394028,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "C0",
      "channel": "OD",
      "order": 1,
      "source_four_mode_delta": 0.0001303171638546985,
      "target_four_domain_delta": -0.0008054108911849129,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "C0",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": 0.007662086076229202,
      "target_four_domain_delta": -0.0054660121771839525,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "C0",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": 0.007662086076229202,
      "target_four_domain_delta": -0.00546939285775461,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "C0",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": 0.0038962016200420058,
      "target_four_domain_delta": -0.0031348339948116723,
      "opposite_sign": true
    },
    {
      "left": "B_FULL_READOUT_025",
      "right": "C0",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": 0.0038962016200420058,
      "target_four_domain_delta": -0.0031374018744697563,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "C0",
      "channel": "OD",
      "order": 0,
      "source_four_mode_delta": 0.00020276122595275226,
      "target_four_domain_delta": -0.00019486967346584122,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "C0",
      "channel": "OD",
      "order": 1,
      "source_four_mode_delta": 0.00020276122595275226,
      "target_four_domain_delta": -0.00019486967346584122,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "C0",
      "channel": "OC",
      "order": 0,
      "source_four_mode_delta": 0.005795508493918189,
      "target_four_domain_delta": -0.0034351281526090326,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "C0",
      "channel": "OC",
      "order": 1,
      "source_four_mode_delta": 0.005795508493918189,
      "target_four_domain_delta": -0.0034351281526090326,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "C0",
      "channel": "macro",
      "order": 0,
      "source_four_mode_delta": 0.002999134859935526,
      "target_four_domain_delta": -0.0018149989130374355,
      "opposite_sign": true
    },
    {
      "left": "B_RESET_READOUT_025",
      "right": "C0",
      "channel": "macro",
      "order": 1,
      "source_four_mode_delta": 0.002999134859935526,
      "target_four_domain_delta": -0.0018149989130374355,
      "opposite_sign": true
    }
  ],
  "decision": "No fixed nonzero condition meets the predeclared +0.5pp vs C0 retest scale; retain zero-update C0 and investigate carrier/readout validity before adding actor complexity.",
  "automatic_followon_authorized": false
}
```
The +0.5pp relative-C0 priority threshold is descriptive R&D, not statistical significance or clinical benefit. No extra alpha, training, seed or monitoring is authorized.
Old attribution/SUP/writer evidence is retained; its small future numerical effect did not establish positive target benefit. SUP_STATIC still uses the frozen B action/basis path; RET-versus-STATIC is not a writer-only experiment.
Last-quarter and switch-first32 records are composition-sensitive descriptions, not revisit/forgetting claims. Missing diagnostic quantities are NOT_RECORDED, never zero.
Anonymous summaries are intended for local commit only; private raw images/masks/probabilities, content IDs, credentials and host paths are not public artifacts.
