# R12_HORIZONTAL_WEIGHT_V1 完成审阅

北京时间2026-10-02 00:41:16完成。四条新增目标流均完成1024在线访问、888主评分，合计4096访问/3552主评分；旧C0、50%翻转和GraTa按完全匹配的内容/order封存回执复用。所有新目标概率均封存后由独立CPU scorer读取标签，临时概率随后退休。整轮终态后才审阅新目标分数。旧watchdog、supervisor和worker均已退出，无恢复或GPU失败。

| 条件 | 两order域等权Dice % | 相对C0 pp |
|---|---:|---:|
| C0 | 74.814113 | +0.000000 |
| CV_H025 | 74.949636 | +0.135522 |
| CV_H2 | 74.134446 | -0.679668 |
| H_ONLY | 70.148508 | -4.665606 |
| G | 76.211671 | +1.397558 |

25%翻转条件两个order各+0.135522pp，低于冻结的+0.5pp优先尺度，因此不能称本轮成功确认。相对GraTa两order均值仍低1.262036pp。仅翻转条件下降4.665606pp、最差域/通道下降13.020236pp；50%翻转原负结果全部保留，未删除坏域或选择seed。

25%条件相对C0的域macro变化分别为Drishti -0.113843pp（n=19）、ORIGA +0.173686pp（n=307）、REFUGE +0.049362pp（n=176）、REFUGE_Valid +0.432885pp（n=386）。OD/OC域等权各+0.147654/+0.123391pp，最差Drishti/OC -0.278190pp。完整各条件、order、域、OD/OC及配对分布保留在CSV；n是图像而非独立患者。

源16×32访问、四控制共2048逻辑访问完成。新C0和50%翻转的32个控制episode与旧源记录hard/soft OD/OC逐项一致，最大差0、容差1e-12，确认巨大翻转退化属于冻结推理路径而非本次源重放不一致。源hard Dice：C0 86.694501%，25% 86.834606%，50% 69.208581%，仅翻转51.870965%。25%源hard仅提高0.140105pp，源soft Dice却从85.414565%下降到76.933253%（-8.481312pp）；小幅hard收益不能掩盖概率质量退化。源资格检查不用于调权，目标条件始终固定。

新增物理成本9440 forwards（预检224、源3072、目标6144），0 backward/optimizer/VJP；GPU-worker1640.772秒，原T0开始计费墙钟2533.203秒。旧包及后继计费血缘累计34222.388秒，24小时新campaign保守累计GPU2494.700秒，两账本分别保留。首次CPU资格测试因错误预期失败，修正测试扰动为0.9后通过；没有GPU消耗、目标访问或重置T0。

所有数据为已暴露开发证据，两个order共享内容，方法确定且模型无更新，不能当作独立复制、多seed验证或临床泛化证据。翻转预测的偏差机制得到支持，25%的微弱收益仍可能是短流构成效应。下一步冻结25%权重，扩展到原注册完整开发覆盖以检验收益是否保持；不细扫更多权重，不恢复RL。新实验依据用户的独立持续授权设计，原DECISION中automatic_followon_authorized=false保留为本阶段历史边界，不修改原判定。

---

以下保留原自动终态报告：

# R12_HORIZONTAL_WEIGHT_V1
Status: COMPLETE; execution SHA c3f1667db64a88831cda060c1e1ca5865c61a501.
Fixed horizontal flip weights .25/1, zero parameter updates; target weights 0/.5 and GraTa reused with matching seals. All target scores unblinded after terminal matrix. No condition selection or tuning from source/target scores.
| condition | order | origin/status | domain Dice % |
|---|---:|---|---:|
| CV_H025 | 0 | NEW/COMPLETE | 74.949636 |
| CV_H025 | 1 | NEW/COMPLETE | 74.949636 |
| H_ONLY | 0 | NEW/COMPLETE | 70.148508 |
| H_ONLY | 1 | NEW/COMPLETE | 70.148508 |
| C0 | 0 | REUSED/REUSED_COMPLETE | 74.814113 |
| C0 | 1 | REUSED/REUSED_COMPLETE | 74.814113 |
| G | 0 | REUSED/REUSED_COMPLETE | 76.306138 |
| G | 1 | REUSED/REUSED_COMPLETE | 76.117205 |
| CV_H2 | 0 | REUSED/REUSED_COMPLETE | 74.134446 |
| CV_H2 | 1 | REUSED/REUSED_COMPLETE | 74.134446 |
```json
{
  "matrix_complete": true,
  "signals": {
    "CV_H025": {
      "delta_vs_C0": [
        0.0013552237392671863,
        0.0013552237392671863
      ],
      "worst_domain_channel_delta": -0.00278190170848945,
      "meets_priority_scale": false
    },
    "H_ONLY": {
      "delta_vs_C0": [
        -0.04665605657874428,
        -0.04665605657874428
      ],
      "worst_domain_channel_delta": -0.13020236046450862,
      "meets_priority_scale": false
    }
  },
  "automatic_followon_authorized": false,
  "meaning": "Prespecified exploratory flip-bias/weight diagnostic after negative R11, not an independent confirmation or new CTTA method. Two orders share contents; no independent replication or clinical/general RL conclusion."
}
```
Priority scale is +0.5 percentage point vs C0, both orders positive; descriptive development rule, not significance. Preserved domain/OD/OC degradations and paired distributions accompany all means. Unknown patient dependence; source simulator validation is development evidence only.
No RL, B, history, training, view search, seed expansion or automatic follow-on. Anonymous summaries exclude private identities, images, labels, predictions, host paths and credentials.
