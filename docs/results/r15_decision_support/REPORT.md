# R15_DECISION_SUPPORT_V1 完成审阅

北京时间2026-10-02T03:17:19.032183+08:00完成第四且最后一轮后继。新增源16episode×32=512访问，源hard OD/OC/Dice与已封存quarter的16episode逐项差值均为0，容差1e-12。32图源资格和8197对有限logits CPU阈值边界检查通过，参数/BN不变；无失败、恢复或重放。没有新增目标图像访问、标签读取、概率保存或scorer，目标状态为NOT_RUN_SOURCE_ONLY。

规则在native与quarter硬决策不同时保留quarter logits，在决策相同时保留native logits。它没有可调区间，按构造保持quarter硬mask；这只是数学结论与源端执行资格，不是新的目标测量或封存轨迹。

| 源条件（同一16episode、域模式等权） | hard Dice % | soft Dice % |
|---|---:|---:|
| C0 | 86.694501 | 85.414565 |
| CV_H025 | 86.834606 | 76.933253 |
| U10_H025 | 86.797799 | 85.447182 |
| DS_H025 | 86.834606 | 85.429024 |

DS源soft相对C0+0.014460pp，相对quarter+8.495772pp，相对U10-0.018158pp。冻结机制条件“恢复至少一半quarter相对C0的源soft损失”判定为True。源support像素平均0.173182%；其余保留原概率。完整模式、OD/OC、各episode和不利结果均见附表。softDice不是概率校准充分证据，源恢复也不是目标soft收益或未见数据泛化证据。

历史目标完整quarter75.239183%，C075.079420%，GraTa77.229693%；quarter-C0 +0.159763pp，仍低于原+.5pp尺度，GraTa差距1.990510pp。DS不能通过保持相同硬决策再提高hardDice，因此没有重复目标计算或伪造DS目标表/soft分数。R14目标门控75.190854%、较C0+0.111434pp、较quarter-0.048329pp也完整保留。所有开发比较已暴露，未知患者依赖、确定性共享内容order，不能称独立确认、多seed、新CTTA方法或临床效果。

实际1184F=160资格+1024源，0BP/optimizer/VJP。GPU-worker116.730575秒，原T0准备及执行墙钟697.582701秒；原计费血缘累计38370.442624秒，campaign保守累计GPU4342.912945秒。执行2b279f37a0fe767f6320427da270507a29382cba，配置2d8abc060d67fc4d0a105a938ffb9de710c619b9929f0a36089a771f6d1b245d。未重置T0或账本，所有本轮attempt均计入。

四轮后继上限已用完，本轮交付核实后关闭campaign、暂停每小时自动化，不启动第五轮。水平翻转混合可获得小幅开发集hard收益，并可修复源概率损失，但没有达到预设目标收益尺度；当前证据不支持恢复RL或继续细扫权重/区间。

---

原自动终态报告：

# R15_DECISION_SUPPORT_V1
Status COMPLETE; execution 2b279f37a0fe767f6320427da270507a29382cba.
SOURCE_ONLY. Preserve native logits on native/quarter hard agreement, quarter logits on disagreement; no eligibility threshold search. No new target access/label/probability/scorer. Mathematical hard-mask preservation is not a new target measurement.
| source condition | hardDice % | softDice % |
|---|---:|---:|
| C0 | 86.694501 | 85.414565 |
| CV_H025 | 86.834606 | 76.933253 |
| U10_H025 | 86.797799 | 85.447182 |
| DS_H025 | 86.834606 | 85.429024 |
```json
{
  "source_complete": true,
  "targets": "NOT_RUN_SOURCE_ONLY",
  "new_target_visits": 0,
  "quarter_hard_equal_by_construction": true,
  "source_hard_reference_parity": {
    "episodes": 16,
    "max_hard_metric_delta": 0.0,
    "passed": true,
    "selection": false,
    "tolerance": 1e-12
  },
  "soft_half_gap_recovered": true,
  "source_soft_delta_vs_C0": 0.0001445952225676006,
  "source_soft_delta_vs_quarter": 0.08495771605788682,
  "source_soft_delta_vs_U10": -0.00018157832383136618,
  "original_target_priority": "Prior measured quarter+0.159763pp vsC0 below+0.5pp; no new target improvement possible from decision preservation",
  "campaign_successor_round": 4,
  "automatic_next_round": false,
  "stop_reason": "Four authorized successors exhausted; close and pause after delivery",
  "independent_confirmation": false
}
```
Full source modes/OD/OC, support fractions and all old adverse results retained. Source softDice improvement does not establish calibration, unseen generalization, a new CTTA method or clinical utility. Historical target quarter75.239183%,C075.079420%,G77.229693% are context only; new target soft is NOT_RUN. No meaningful fake seed rerun. Four successor budget reached; close report and pause heartbeat after publication verification.
