# R14_UNCERTAINTY_FLIP_V1 完成审阅

北京时间2026-10-02T02:28:53.432164+08:00COMPLETE，源新增16episode/512访问和两条目标1951在线/1695主评分均完整，共3902新目标访问/3390主评分。标签由独立CPU scorer在在线封存后读取，整轮终态后才审阅新目标分数。临时概率均已退休，所有父子实验进程退出，无恢复或计算失败。

| 完整流条件 | 两order域等权hard Dice % | 相对C0 pp |
|---|---:|---:|
| C0 | 75.079420 | +0.000000 |
| CV_H025 | 75.239183 | +0.159763 |
| U10_H025 | 75.190854 | +0.111434 |
| G | 77.229693 | +2.150273 |

U10较C0两order均+0.111434pp，较恒定quarter -0.048329pp，仍未达到原+.5pp优先尺度，也没有超过quarter。相对GraTa低2.038839pp。最差Drishti_GS/OC -0.029691pp(n=37)，未触发>2pp风险。所有条件、域、OD/OC、配对分布与尾部组成描述保留。

源U10 hard86.797799%，quarter86.834606%，C086.694501%；soft85.447182%，quarter76.933253%，C085.414565%。U10源soft恢复原大幅损失、略高C0约0.032617pp，hard相对quarter仅-0.036807pp，满足两项冻结源机制描述。源端正结果不等于足够的目标收益。目标U10相对quarter的域等权soft Dice提高OD1.059175pp/OC1.413335pp，macro1.236255pp；旧C0/G概率不存在，soft明确NA，不从mask伪造。门控只对主评价平均0.277453%像素生效，99.722547%保留原始logits。源平均eligible像素0.228987%。概率输出不被宣称校准；softDice也不是校准充分证据。

32真实源图C0/quarter/U10的公式与保护资格通过，受保护logits/概率精确、BN/参数不变。仅新增U1016×32源比较，旧C0/quarter已完成源记录复用。旧完整C0/GraTa历史轨迹和quarter静态分区标量合成按原seals复用，没有重复旧科学轨迹或伪造完整quarter online seal。所有原注册数据已在R7/R8/R10/R13暴露，两order共享内容且确定性，未知患者依赖，不能冒充独立确认、多seed或临床结果。

新增8988F(预检160+源1024+目标7804)，0BP/optimizer/VJP；GPU-worker958.338602秒，原T0含准备墙钟1870.943657秒。计费血缘累计37672.859923秒、新campaign保守累计GPU4226.182369秒，未重置或重复计费。执行e72aaf359561b25d22f9b3efe594c5f644338472，配置5c9cc22ae98be6767064e3d9c16412c9cc7072dfdc0fe16e02a8657c94e7309f。

下一独立且最后一轮将仅做源端机制检验：若quarter改变native硬决策，保留quarter logits，否则保留native logits。规则没有可调区间，按构造与quarter硬mask精确相同，旨在检验恢复soft概率表现能否同时保留原硬决策；不会声称额外目标hard收益，也不重复完整目标计算。原+.5pp判定不变，多次开发比较局限完整保留，RL仍退出主线。

---

原自动终态报告：

# R14_UNCERTAINTY_FLIP_V1
Status COMPLETE; execution e72aaf359561b25d22f9b3efe594c5f644338472.
Fixed native probability[.4,.6] eligibility with quarter flip mixing; confident original logits exact. Two new full1951/1695 gated trajectories only; complete prior C0/constantquarter/nativeGraTa baselines reused. New scores reviewed only after terminal matrix.
| condition | order | OD % | OC % | domain Dice % |
|---|---:|---:|---:|---:|
| C0 | 0 | 83.269851 | 66.888990 | 75.079420 |
| C0 | 1 | 83.269851 | 66.888990 | 75.079420 |
| CV_H025 | 0 | 83.410275 | 67.068091 | 75.239183 |
| CV_H025 | 1 | 83.410275 | 67.068091 | 75.239183 |
| G | 0 | 85.670816 | 69.073385 | 77.372100 |
| G | 1 | 85.774113 | 68.400457 | 77.087285 |
| U10_H025 | 0 | 83.362382 | 67.019326 | 75.190854 |
| U10_H025 | 1 | 83.362382 | 67.019326 | 75.190854 |
```json
{
  "matrix_complete": true,
  "delta_vs_C0": [
    0.0011143390381032627,
    0.0011143390381032627
  ],
  "delta_vs_constant_quarter": [
    -0.000483288575850183,
    -0.000483288575850183
  ],
  "meets_original_priority": false,
  "beats_quarter_both_orders": false,
  "worst_domain_channel_delta": -0.00029691386002419525,
  "risk_over_2pp": false,
  "source_mechanism": {
    "soft_half_gap_recovered": true,
    "hard_loss_vs_quarter_within_0_5pp": true,
    "selection": false
  },
  "independent_confirmation": false
}
```
Source and target signs retained, no score-based tuning. Original+.5pp priority and >2pp domain-risk threshold unchanged. Legacy probability/softNA; new gated-versus-constantquarter soft contrasts explicit. All data are already exposed development evidence, two correlated deterministic orders, unknown patients; no independent confirmation/new CTTA/clinical/RL claim. Physical cost and all failure/recovery expenses retained.
