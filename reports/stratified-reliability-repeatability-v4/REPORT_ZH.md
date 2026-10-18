# V4：无标签重复性工程诊断结果

2026-10-19；冻结源码b7fb7eed0a21a12174e5c5520888d84bcb6295a0。**工程资格PASS，未估计分割效果。** 全部128arrival完成，256个PRE/POST读出构成128对；STRICT八组logits与末model.state_dict逐元素完全一致。GPU worker与CPU汇总均exit0/退役，controller已退出。

固定seed、禁benchmark/TF32的DEFAULT八组均出现logit/state差异；强制torch.use_deterministic_algorithms(True,warn_only=False)+cudnn.deterministic=True后，STRAT/RANDOM/CW1/CW2的RESET/RETAIN八组差异均为零。DEFAULT硬预测共11个位差异（STRAT RETAIN10、CW2 RETAIN1），其余虽无硬预测变化，浮点logits和状态仍不同。DEFAULT STRAT RETAIN最大logit差.0144863，说明连续更新可放大微小计算差异。

|设置|方法|状态|logit差异元素|预测位差|最大logit差|末模型状态差异元素|
|---|---|---|---:|---:|---:|---:|
|DEFAULT|STRAT|RESET|1355385|0|5.1498413e-05|2214|
|DEFAULT|STRAT|RETAIN|3491289|10|0.014486313|18999|
|DEFAULT|RANDOM|RESET|1510047|0|0.00085353851|3733|
|DEFAULT|RANDOM|RETAIN|2669525|0|9.2625618e-05|9260|
|DEFAULT|CW1|RESET|1386425|0|2.682209e-05|26|
|DEFAULT|CW1|RETAIN|2580494|0|4.1246414e-05|6868|
|DEFAULT|CW2|RESET|1519738|0|7.7366829e-05|2335|
|DEFAULT|CW2|RETAIN|3129282|1|0.00022816658|11371|
|STRICT|STRAT|RESET|0|0|0|0|
|STRICT|STRAT|RETAIN|0|0|0|0|
|STRICT|RANDOM|RESET|0|0|0|0|
|STRICT|RANDOM|RETAIN|0|0|0|0|
|STRICT|CW1|RESET|0|0|0|0|
|STRICT|CW1|RETAIN|0|0|0|0|
|STRICT|CW2|RESET|0|0|0|0|
|STRICT|CW2|RETAIN|0|0|0|0|

## 解释与边界

此干预支持非确定性计算是当前实现重复性差异的一个来源，但没有定位具体CUDA算子，也不能证明V3全部跨进程差异只有这一原因。这里是同一进程、两个已暴露内容、四次arrival；没有认证跨进程或完整40arrival的严格复现。model.state_dict包括参数和缓冲，**不包括Adam矩或独立native RNG状态**；logits/末模型一致不能扩写成全部内部状态逐位一致。

V3的冻结跨轮parity检查失败仍有效，NOT_EVALUABLE不变；不重评分、不放宽历史标准。V2首次到达无正收益、可靠性相对随机区间跨零等科学负证据也不改变。确定性是工程条件，不是TMI创新，更不是适应性能提升。

后续只值得另冻同进程、严格确定性的状态因果对照：同时计算保留参数和Adam、只保留参数及完全RESET，保留配额随机、源模型、原生CW强控制，以到达前PRE为主。新轮不使用V2/V3旧差值充当对照；若无首到达和源模型收益，不将优化器重置包装为CTTA贡献。

未读取标签、旧预测/评分；无Dice或患者效应估计。详细范围见[冻结协议](PROTOCOL_ZH.md)与[执行审计](EXECUTION_AUDIT.md)。
