# R13_FULL_COVERAGE_V1 完成审阅

北京时间2026-10-02T01:30:23.306820+08:00完成。两条新增补集流各927在线/807主评分，合计1854在线/1614主评分，独立CPU标签评分在各在线轨迹封存后执行，整轮终态后才审阅新目标分数。全部临时概率退休，父子实验进程均退出，无恢复、无计算失败。

| 完整原注册流条件 | 两order域等权Dice % | 相对C0 pp |
|---|---:|---:|
| C0 | 75.079420 | +0.000000 |
| CV_H025 | 75.239183 | +0.159763 |
| G | 77.229693 | +2.150273 |

25%条件完整开发覆盖的两order均+0.159763pp，仍未达到冻结+0.5pp优先尺度；相对GraTa均值低1.990510pp。补集quarter75.547926%、C075.362455%，+0.185472pp；旧短quarter74.949636%、C074.814113%，+0.135522pp。因此小幅hard收益不只来自旧短流，但规模仍不足。完整流最差Drishti_GS/OC -0.084311pp，n=37，未触发>2pp风险阈值。完整、补集、短流每域OD/OC与配对分布均保留；不只公布总体均值。

完整quarter1951/1695由旧短1024/888与不相交补集927/807的封存scalar按原注册内容身份合并重排；该模型无预测历史，源32图两条件正/逆序128访问逐图logits精确相等，参数/BN不变。没有伪造完整在线概率seal。完整C0和GraTa来自历史R8screen24各1951原轨迹，GraTa保持完整历史不进行拼接；旧C0与短流两order所有1024 hard指标/pixelcounts完全一致。历史bitmask仅支持hard/ASSD，旧soft指标标NA，不从二值mask伪造概率。quarter的新旧分区均保留各自online/scorer seal。

全部原完整开发数据包括补集在R7/R8已暴露，本轮不能称新独立确认。两个order共享图像且确定性，无患者独立性证据，不做假多seed。原16episode源hard/soft全部保留：quarter源hard86.834606%较C086.694501%仅+0.140105pp，soft76.933253%较C085.414565%下降8.481312pp。大幅源soft损失与仅小幅目标hard收益同时存在，后续机制检验需保护原图高置信预测，不能靠权重细扫或恢复RL。

新增3900F（192源资格+3708目标），0BP/optimizer/VJP；GPU-worker773.143510秒，含准备的原T0计费墙钟1579.527836秒。原血缘墙钟累计35801.916266秒；新campaign保守累计GPU3267.843768秒。初次部署模板替换错误在远端执行前修复，本机缺torch转用既有兼容远端CPU运行；两项0GPU/0目标，失败记录与T0均保留。结束到本次监测间空闲不追加GPU费用。执行80ffca90deeba8abcae14c4c6bc756a3b6671ec0，冻结配置331968d2a1ad6051331cb6cf5552eba593a4a0cd3093d60e5b51d426fe159903。

原判定保留为未达到优先尺度，不宣称CTTA新方法或临床效果。下一独立冻结后继只检验原图不确定区间上的25%混合这一因素，完整保留恒定25%与C0/GraTa对照；仍为开发机制试验。

---

原自动终态报告：

# R13_FULL_COVERAGE_V1
Status COMPLETE; execution 80ffca90deeba8abcae14c4c6bc756a3b6671ec0.
Only the fixed quarter-weight complementary927 arrivals/order were newly inferred; full1951 coverage is composed from disjoint sealed short1024 and complement927 partitions. Primary1695 =888+807. No invented full online seal or training seed replication. C0/GraTa complete1951 historical native trajectories reused with hard-metric compatibility qualification. Legacy probability/soft metrics are unavailable (NA).
| scope | condition | order | OD % | OC % | domain Dice % |
|---|---|---:|---:|---:|---:|
| COMPLEMENT | C0 | 0 | 83.183199 | 67.541710 | 75.362455 |
| COMPLEMENT | C0 | 1 | 83.183199 | 67.541710 | 75.362455 |
| COMPLEMENT | CV_H025 | 0 | 83.315888 | 67.779965 | 75.547926 |
| COMPLEMENT | CV_H025 | 1 | 83.315888 | 67.779965 | 75.547926 |
| FULL | C0 | 0 | 83.269851 | 66.888990 | 75.079420 |
| FULL | C0 | 1 | 83.269851 | 66.888990 | 75.079420 |
| FULL | CV_H025 | 0 | 83.410275 | 67.068091 | 75.239183 |
| FULL | CV_H025 | 1 | 83.410275 | 67.068091 | 75.239183 |
| FULL | G | 0 | 85.670816 | 69.073385 | 77.372100 |
| FULL | G | 1 | 85.774113 | 68.400457 | 77.087285 |
| SHORT | C0 | 0 | 83.347552 | 66.280674 | 74.814113 |
| SHORT | C0 | 1 | 83.347552 | 66.280674 | 74.814113 |
| SHORT | CV_H025 | 0 | 83.495206 | 66.404065 | 74.949636 |
| SHORT | CV_H025 | 1 | 83.495206 | 66.404065 | 74.949636 |
```json
{
  "matrix_complete": true,
  "full_delta_vs_C0": [
    0.0015976276139534457,
    0.0015976276139534457
  ],
  "meets_priority_scale": false,
  "worst_domain_channel_delta": -0.0008431133313785925,
  "risk_over_2pp": false,
  "independent_confirmation": false,
  "meaning": "Broader already exposed development coverage, correlated deterministic orders; original priority threshold unchanged"
}
```
Full old development cohort was exposed during R7/R8. Orders share content, unknown patient dependence. No independent confirmation, blind-test, new CTTA method, clinical or RL claim. Source quarter hard gain and large soft loss are retained without tuning. All target scores reviewed only after terminal matrix. See full domain/channel, paired distribution, source and cost tables.
