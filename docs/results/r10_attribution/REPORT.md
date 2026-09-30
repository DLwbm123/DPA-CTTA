# R10_ATTRIBUTION_SUP_8H_V1
Status: COMPLETE; execution SHA fa32e7fc40629280d04ed1770979984193bace68.
Sealed comparisons were unblinded only after terminal run state. New training branches: ['SUP_RET', 'SUP_STATIC'].
Actual new wall: 11522.56s; GPU-worker: 9539.27s; recovery used: False.
One policy seed; two correlated orders with the same contents. Patient dependence is unknown; no independent-seed confidence interval or target-selected checkpoint.
SUP versus GR differs in objective and compute, so it is a training-recipe comparison, not a pure estimator causal experiment.
Last-quarter summaries use the fixed final 256 arrivals and may differ in domain composition. No revisits are invented.

| method | order | origin/status | OD % | OC % | domain macro % | pooled % |
|---|---:|---|---:|---:|---:|---:|
| N | 0 | REUSED/REUSED_COMPLETE | 78.83378 | 57.50862 | 68.17120 | 63.98124 |
| N | 1 | REUSED/REUSED_COMPLETE | 78.83378 | 57.50862 | 68.17120 | 63.98124 |
| G | 0 | REUSED/REUSED_COMPLETE | 85.03216 | 67.58011 | 76.30614 | 76.23514 |
| G | 1 | REUSED/REUSED_COMPLETE | 84.95217 | 67.28223 | 76.11720 | 75.94627 |
| GR_RET_EMA | 0 | REUSED/REUSED_COMPLETE | 82.91189 | 61.09620 | 72.00405 | 72.18133 |
| GR_RET_EMA | 1 | REUSED/REUSED_COMPLETE | 82.90616 | 61.07276 | 71.98946 | 72.18000 |
| GR_RET_EMA_CONST_HALF | 0 | REUSED/REUSED_COMPLETE | 82.91194 | 61.09654 | 72.00424 | 72.18139 |
| GR_RET_EMA_CONST_HALF | 1 | REUSED/REUSED_COMPLETE | 82.90620 | 61.07309 | 71.98965 | 72.18006 |
| C0_CURRENT_STATS | 0 | NEW/COMPLETE | 83.34755 | 66.28067 | 74.81411 | 74.27949 |
| C0_CURRENT_STATS | 1 | NEW/COMPLETE | 83.34755 | 66.28067 | 74.81411 | 74.27949 |
| B_PARENT_FULL | 0 | NEW/COMPLETE | 82.89409 | 62.26053 | 72.57731 | 74.25717 |
| B_PARENT_FULL | 1 | NEW/COMPLETE | 82.89260 | 62.26349 | 72.57805 | 74.25240 |
| WARM_CONST_HALF | 0 | NEW/COMPLETE | 83.27143 | 61.83396 | 72.55269 | 72.41173 |
| WARM_CONST_HALF | 1 | NEW/COMPLETE | 83.26487 | 61.79735 | 72.53111 | 72.41062 |
| R10_RESET_ALL | 0 | NEW/COMPLETE | 82.83448 | 61.47875 | 72.15662 | 72.23207 |
| R10_RESET_ALL | 1 | NEW/COMPLETE | 82.83448 | 61.47875 | 72.15662 | 72.23207 |
| R10_FORCE_WRITE | 0 | NEW/COMPLETE | 82.93337 | 61.16239 | 72.04788 | 72.19080 |
| R10_FORCE_WRITE | 1 | NEW/COMPLETE | 82.93280 | 61.15676 | 72.04478 | 72.19042 |
| SUP_RET | 0 | NEW/COMPLETE | 82.88366 | 61.70528 | 72.29447 | 72.36082 |
| SUP_RET | 1 | NEW/COMPLETE | 82.87899 | 61.68493 | 72.28196 | 72.36058 |
| SUP_RET_CONST_HALF | 0 | NEW/COMPLETE | 82.88456 | 61.71052 | 72.29754 | 72.36309 |
| SUP_RET_CONST_HALF | 1 | NEW/COMPLETE | 82.88046 | 61.69103 | 72.28574 | 72.36291 |
| SUP_STATIC | 0 | NEW/COMPLETE | 82.91914 | 61.85928 | 72.38921 | 72.29792 |
| SUP_STATIC | 1 | NEW/COMPLETE | 82.91914 | 61.85928 | 72.38921 | 72.29792 |

## Attribution (percentage points)
| comparison | order | macro delta pp |
|---|---:|---:|
| GR_RET_EMA - C0_CURRENT_STATS | 0 | -2.810067 |
| GR_RET_EMA - C0_CURRENT_STATS | 1 | -2.824651 |
| GR_RET_EMA - B_PARENT_FULL | 0 | -0.573267 |
| GR_RET_EMA - B_PARENT_FULL | 1 | -0.588584 |
| GR_RET_EMA_CONST_HALF - WARM_CONST_HALF | 0 | -0.548453 |
| GR_RET_EMA_CONST_HALF - WARM_CONST_HALF | 1 | -0.541460 |
| GR_RET_EMA - R10_RESET_ALL | 0 | -0.152570 |
| GR_RET_EMA - R10_RESET_ALL | 1 | -0.167154 |
| GR_RET_EMA - GR_RET_EMA_CONST_HALF | 0 | -0.000193 |
| GR_RET_EMA - GR_RET_EMA_CONST_HALF | 1 | -0.000185 |
| GR_RET_EMA - R10_FORCE_WRITE | 0 | -0.043831 |
| GR_RET_EMA - R10_FORCE_WRITE | 1 | -0.055320 |
| SUP_RET - SUP_RET_CONST_HALF | 0 | -0.003071 |
| SUP_RET - SUP_RET_CONST_HALF | 1 | -0.003779 |
| SUP_STATIC - GR_RET_EMA | 0 | +0.385164 |
| SUP_STATIC - GR_RET_EMA | 1 | +0.399748 |
| SUP_RET - GR_RET_EMA | 0 | +0.290423 |
| SUP_RET - GR_RET_EMA | 1 | +0.292502 |
| SUP_STATIC - G | 0 | -3.916928 |
| SUP_STATIC - G | 1 | -3.727994 |
| SUP_RET - G | 0 | -4.011670 |
| SUP_RET - G | 1 | -3.835241 |
| SUP_RET - SUP_STATIC | 0 | -0.094741 |
| SUP_RET - SUP_STATIC | 1 | -0.107247 |

## Three bounded answers
```json
{
  "where_improvement_comes_from": {
    "R10_minus_C0": [
      -0.028100672855053695,
      -0.028246512342781763
    ],
    "R10_minus_B_PARENT": [
      -0.005732665098342975,
      -0.005885838280795847
    ],
    "post_minus_WARM_fixed_writer": [
      -0.005484529823494101,
      -0.005414603780018579
    ],
    "interpretation": "Compare matched normalization, carrier and post-use controls; do not attribute R10-N to RL without these controls.",
    "answer": "R10-C0: -2.81007pp, -2.82465pp; R10-parent B: -0.57327pp, -0.58858pp; post-WARM at fixed writer: -0.54845pp, -0.54146pp. Matched current-statistics C0 already matches or exceeds R10, so R10-N does not establish a policy/RL gain."
  },
  "does_writer_change_future": {
    "source_probability_change_max": 0.0006586236850125715,
    "source_mask_flip_max": 0.0006551742553710938,
    "learned_SUP_writer_delta": [
      -3.070639725235007e-05,
      -3.779285672049701e-05
    ],
    "caveat": "Source intervention consequences do not establish deployable target benefit.",
    "answer": "Fixed-source writer interventions do change future probabilities (max recorded channel-mean absolute difference 0.00065862369; mask flip fraction 0.00065517426). Learned SUP writer target advantage is -0.00307pp, -0.00378pp; numerical change alone is not useful target performance."
  },
  "next_research_line": {
    "answer": "Prioritize current-image normalization/carrier validity; this run does not justify adding further policy complexity on frozen B.",
    "best_new_two_order_mean": "C0_CURRENT_STATS",
    "SUP_RET_research_signal": false,
    "SUP_STATIC_research_signal": false,
    "SUP_writer_retest_signal": false,
    "automatic_followon_authorized": false
  }
}
```
The +0.5pp recipe and +0.2pp learned-writer thresholds are descriptive R&D signals, not statistical significance or clinical standards. Decisions are post-run only; no additional run is authorized.
Raw permissible scalar records and checkpoints remain in the private output root. Four requested tables: ATTRIBUTION.csv, WRITER_DIAGNOSTICS.csv, SOURCE_COUNTERFACTUAL.csv, COST_AND_STATUS.csv.

## 本地完成交付

本轮于北京时间 2026-09-30 21:09:56 结束，原始自动报告于约 21:10 写出。包含预检的封闭运行墙钟为 11522.56 秒（3 小时 12 分 2.6 秒）；GPU-worker 累计 9539.27 秒。16 条新增轨迹全部完成在线封存及独立评分，共 16384 次到达、14208 条主评分；复用 8 条原轨迹。两条 SUP 各完成 1024 轮、2048 次 optimizer 更新及固定的 16 个源验证 episode；32 个源反事实上下文完成。无失败、无恢复，完成检查时本轮进程均已退出。

三个回答：

1. **改善归因：** 当前归一化 C0 的两 order 平均域等权 Dice 为 74.81411%，高于原 R10 的 71.99675%；R10 还低于父 B 约 0.58pp。共同固定 writer 下，post 比 WARM 低约 0.54pp。因此 R10 相对 Source-only 的改善不能归因于本轮 RL 训练收益。
2. **writer 是否改变未来：** 固定源端干预会改变完整提交状态和未来概率/硬掩码，但变化很小；记录中的最大通道平均概率绝对差为 0.000658624，最大掩码翻转比例为 0.000655174。SUP_RET writer 梯度非零，目标确定性均值约 0.4563，但 learned writer 相对同权重固定 0.5 在两条 order 分别低 0.003071/0.003779pp。未达到预先声明的 writer 复验尺度；不能把源端数值变化当作目标性能收益。
3. **当前研究方向：** 先检查当前图像归一化与父载体有效性。SUP_STATIC 与 SUP_RET 相对原 R10 分别平均提高约 0.39246/0.29146pp，均低于 +0.5pp 研发尺度，且低于 C0 和 GraTa。当前证据不支持继续盲目扩展冻结 B 上的策略复杂度。本轮是一种子、两条相关 order 的开发诊断，不建立跨种子或临床结论，也不自动启动后续实验。

DOMAIN_RESULTS 提供每方法/域/OD/OC/order 主评分，ATTRIBUTION 提供逐图配对分布及固定后 256 次到达的汇总。后段构成可能改变，不能直接解释为遗忘。Drishti_GS 每 order 仅 19 个主评分样本，最差域/通道值需结合样本量阅读。

RESOLVED_CONFIG.public.json 保留冻结代码、模型/流摘要、训练与评分配置及预算身份；仅省略私有路径和内容清单。COMPLETION_RECEIPTS、RESOURCE_SUMMARY 和四张请求表均已收集。原始允许保存的逐图标量、模型与检查点仍保存在原私有服务器目录，未复制患者影像到公开树。运行耗时与完成后等待本次收集的时间分开记录，未重置 T0。按本轮计划仅本地提交，不推送 GitHub。
