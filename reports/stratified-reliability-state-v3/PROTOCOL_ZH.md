# V3：跨图像 Adam 状态的单因素拆解（冻结协议）

2026-10-19。用户持续授权下的独立新轮，依据V2完整结果：STRAT RESET_POST92.6026%、RETAIN_POST91.9442%、SOURCE_EVAL92.1572%；RETAIN_PRE−RESET_PRE −.9080pp，首块−.2600pp，参数距离和功能变化均明显累积。V2公开提交7d5c59f6b2c2d838616efa2c24379c7eb7bb0ee1。不能根据这些关联断言Adam矩导致伤害。

**唯一主要干预**：core八臂的RETAIN保持已适应仿射坐标，每个arrival开始前重新建立相同Adam，丢弃跨图像一阶/二阶矩及step计数；同一图像的两次更新仍共享Adam。RESET完全不变，CW1/CW2原生状态语义不变。输出表RETAIN在本轮core明确表示AFFINE_ONLY；V2core RETAIN表示AFFINE_AND_ADAM，不能混淆。模型buffers、损失、源锚、可靠性排序、视图、seed、.01实际半径和5e−7容差全部固定。非RDC/ENGRAM、无新源训练。

相同完整强控制矩阵3600读出/40轨迹/1600arrival/128旧参考parity，所有V2臂保留。CPU评分在worker退出后才读取V2配对私有Dice数组和预测；worker禁止整个V2目录访问，不能用旧标签或预测更新。配对按case/block/order/arm/policy/stage精确核同序，2320个未改变控制读出（400静态+1600RESET+320原生RETAIN）必须精确bit parity，否则停止为不可评价。

唯一主比较：STRAT_AFFINE_ONLY−STRAT_AFFINE_AND_ADAM:RETAIN_PRE，使用V2与V3配对，整体+≥.5pp且8内容组描述性bootstrap区间下界>0。不能用POST替代PRE主结论。预设拆解包括各core臂PRE/POST状态效应、STRAT状态变化减RANDOM状态变化、首块/全流/返回PRE对SOURCE_EVAL、POST对完全RESET，全部报告。仍报告原V2路由门槛到RESULTS.json（其primary仍是STRAT−RANDOM），**本轮主结果和推进判断以STATE_COMPARISON.json为准**，不覆盖历史冻结结果。

推进还需首次内容PRE对源模型区间下界>0、全流PRE对源模型区间下界>0、最差通道条件对源模型≥−1pp、同配额RANDOM POST差区间下界>0、实际范数匹配、无新增空或嵌套违规。完整结构与前景结果继续展示，未超过源模型不可称安全改进。2320控制parity是执行资格，不是科学成功。

预算新轮GPU进程≤1200秒、独立CPU评分≤900秒、输出≤2GiB。V1/V2累计GPU500.2589249611秒另行保留，此轮不改旧ledger或清零累计成本。运行前CPU新回归验证“参数逐值不变、Adam确实清空、下一次PRE等于上一POST、优化器年龄重置而总steps递增”，并保留其余8项回归；不得先运行再改主比较。预算、数值、访问、参数ownership或parity失败不重试、不放宽。

没有新增图像/seed/长流/模型。8个已暴露内容，患者依赖及预训练成员未知，区间仅描述开发组。新轮由已看V2结果提出，不是盲测，更不构成TMI创新；清空优化器状态也是已有工程策略。若因果拆解仍无首次到达收益，不将其包装成CTTA贡献，不继续盲调门控/步幅。
