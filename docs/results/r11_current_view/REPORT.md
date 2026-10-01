# R11_CURRENT_VIEW_V1 完成审阅

北京时间 **2026-10-01T22:55:53.237371+08:00** 已完成，两个新增目标流各1024次在线访问、888条主评分，全部封存并独立评分。源验证16个固定episode、512访问完成；临时概率均已有退休回执，实验watchdog/supervisor/worker均已退出。

| 条件 | 两order域等权Dice % | 相对C0 pp | 来源 |
|---|---:|---:|---|
| CV_H2 | 74.134446 | -0.679668 | 新增 |
| C0 | 74.814113 | +0.000000 | 精确配对复用 |
| G | 76.211671 | +1.397558 | 精确配对复用 |

水平翻转等权概率平均较C0下降0.679668pp，两order结果相同，未达到原+0.5pp优先尺度；最差域/通道是REFUGE_Valid/OD，下降4.154192pp，n=386/order。四个域的macro差均为负，完整OD/OC与配对分布保留，不只报总体均值。源验证C0为86.694501%，CV_H2为69.208581%，下降-17.485920pp。该源差很大，后续需用同进程新C0/half配对重放验证基线一致性，不能据此武断归因。

固定增强思路未建立当前注册开发短流上的正收益，不进行无意义多seed复跑。所有数据属于开发证据，两个order共享内容、方法确定，未知患者依赖，不提供独立复验或临床泛化结论。四视图仅因目标访问前预算准入失败暂缓，无性能结论。

原90分钟预算包含准备与两次预检，原计费墙钟1977.004秒，GPU-worker853.928秒，5504 forwards，0 backward/optimizer/VJP。旧包加本轮计费共31689.185秒，未重置。终态等待本次监测和收集的空档不追加为实验GPU收费。

汇总字典漏写OPS迭代导致旧RESOURCE_LEDGER.operations只有soft_Dice:0；现由完整原始7份物理attempt回执修正为5504/0/0/0，原结果/概率封存/分母/实际时长不变，没有重跑。执行代码仍为b17fdd56e2be80f4df9f0c755c8a85675f53f551，报告修复版本eb72226a22284c4a01f65ef62297fb6513170998，修复记录另存REPORT_CORRECTION.json。

用户随后已授权独立后续阶段，故旧DECISION中的automatic_followon_authorized=false保留为R11冻结时的历史字段；新授权记录在CONTINUATION_20261001.md，不修改旧判定或旧科学锁。下一阶段只用于开发诊断翻转预测偏差和混合权重，不属于独立确认。

---

以下保留原自动终态报告：

# R11_CURRENT_VIEW_V1
Status: COMPLETE; execution SHA b17fdd56e2be80f4df9f0c755c8a85675f53f551.
Fixed current-image flip probability averages, zero parameter updates. All target scores unblinded after terminal matrix. No condition selection or tuning from source/target scores.
| condition | order | origin/status | domain Dice % |
|---|---:|---|---:|
| CV_H2 | 0 | NEW/COMPLETE | 74.134446 |
| CV_H2 | 1 | NEW/COMPLETE | 74.134446 |
| C0 | 0 | REUSED/REUSED_COMPLETE | 74.814113 |
| C0 | 1 | REUSED/REUSED_COMPLETE | 74.814113 |
| G | 0 | REUSED/REUSED_COMPLETE | 76.306138 |
| G | 1 | REUSED/REUSED_COMPLETE | 76.117205 |
```json
{
  "matrix_complete": true,
  "signals": {
    "CV_H2": {
      "delta_vs_C0": [
        -0.006796677144417792,
        -0.006796677144417792
      ],
      "worst_domain_channel_delta": -0.04154191620717625,
      "meets_priority_scale": false
    }
  },
  "automatic_followon_authorized": false,
  "meaning": "Fixed test-time augmentation pilot, not a new CTTA method. Two orders share contents; no independent replication or clinical/general RL conclusion."
}
```
Priority scale is +0.5 percentage point vs C0, both orders positive; descriptive development rule, not significance. Preserved domain/OD/OC degradations and paired distributions accompany all means. Unknown patient dependence; source simulator validation is development evidence only.
No RL, B, history, training, view search, seed expansion or automatic follow-on. Anonymous summaries exclude private identities, images, labels, predictions, host paths and credentials.
