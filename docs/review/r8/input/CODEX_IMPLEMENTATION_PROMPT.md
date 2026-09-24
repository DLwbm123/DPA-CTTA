# Codex任务：实现R8长程B/A性能边界实验，不自行启动真实计算

请阅读同目录R8_EXPERIMENT_PLAN.md、R8_EXPERIMENT_SPEC.json和CONFIG_GRID.csv。它们是新实验的设计，不是已经实现的runner或执行授权。不要只调用旧R7入口然后宣称R8已经跑完。

## 目标与范围

B主攻8配置、A辅助4配置，各配独立STATIC。先实现完整65个源训练任务、724条目标轨迹的有界任务图；使一次新范围授权之后可以按依赖连续执行，无须中途根据涨分情况追加计划。旧R7仅只读，不恢复旧中断，不覆盖旧结果，不修改旧论文/实现定义。

精确历史参照：DLwbm123/DPA-CTTA@f4adb85434e5ef36faecc1109f907d7a2f29d17f；旧执行c1763d00f4f29de1c168ccd75219199667478825。沿用CTTA/GraTa已有固定依赖，不自动升级环境。

## 优先实现的检查

1. 身份与隔离：source fit/cal/val分组、target完整1,951到达与1,695主要评分内容、five orders、源checkpoint和evaluator。source-val只作配置/检查点选择，不称盲测；所有目标内容保持已暴露开发身份。
2. FiLM零态与活梯度；a=.1/.3下C0一致；B32/64的H归一化与ISTA步长；A16/32完整P、Cholesky和固定方差下界；不能随机补rank。
3. 源端32visit episode、8visit TBPTT：chunk边界detach不reset FULL，STATIC逐visit重置；每step8query。梯度可达新模块、不更新主干。source query不提交episode state。
4. 五个checkpoint分别独立校准。每config/mode用前两个seed在source-val上选择公共t*，STATIC可与FULL的t*不同；家族配置选择只基于FULL source score。所有任务仍训练满16,000步。
5. VPTTA native memory/warm-up、C/G固定移植规则保持；同一源checkpoint与前后预处理，不把VPTTA的BN改成新路线的BN来伪造统一条件。
6. B_G1/G3仅优化标准化latent，soft target stop-grad、强视图每visit固定、Adam moments逐图清空、正确状态提交；COLD_G3每图zero-init。网络backward真实计数，不能标为zero-backward。
7. 全状态快照/恢复：新R8每job最多一次基础设施等价恢复；验证source/target RNG、优化器、state、VPTTA未注册可变属性、输出commit位置和计数。旧R7不允许恢复。不支持完整恢复者中断后PARTIAL，不伪造连续轨迹。
8. COST_LEDGER分别计source oracle/basis/query/train/calibration、模型F/B/VJP、Adam/AdamW、小矩阵、latent迭代、评分/IO和实际GPU占用。activation checkpoint重算不可漏计；确定性控制的seed引用不可算新增物理运行。

## 运行前交付

生成新代码SHA、可审阅科学配置digest、disabled authorization模板、source/target隔离证据、synthetic测试实际输出、静态全任务manifest与计数核验，以及需要真实profile后才能填的资源报告。不要填写虚假PASS，不把历史测试冒充新测试。没有获得新的真实数据/模型/profile授权时，只做代码和程序化检查。

可选MGIPT/PEOA-CTA/SPEGC只有主任务启动前完成固定、匹配的资格检查才加入，最多75条；否则记录缺失，不以原论文数字填表，也不临时开展漫长复现项目。

## 一次授权后任务图

SOURCE_PROFILE_AND_BIND -> SOURCE_ORACLES_AND_BASES -> ALL48_SOURCE_MODELS -> SOURCE_ONLY_CONFIG_AND_SNAPSHOT_SELECTION -> ADDITIONAL12_SOURCE_MODELS_AND5_MLP -> SOURCE_CAL_GRADIENT_LR_SELECTION -> LOCK_ALL_ARTIFACTS -> TARGET_GRID_AND_BASELINES_AND_ABLATIONS -> MIXED_AND_LONG10 -> READ_ONLY_AGGREGATION.

这里描述的是需要实现的任务图，不是现成函数名或已存在的命令。实际入口和参数在新代码里实现并审阅，保持execution_enabled=false直到新范围授权绑定。

禁止因早期/最终分数低提前停其他有效配置；禁止运行中换seed、LR、ridge、幅度、指标、患者划分或新加loss。数值错误停止受影响job并保留；共同身份/标签隔离出错或触及资源cap则全局停止。资源不足不得偷减矩阵。

## 验收输出

必须交出全部配置（含失败）、source-selected五seed主表、两seed网格、source observed envelope、所有同权重消融、B梯度成本/收益、每轮长流曲线、各域/OD/OC/配对尾部/ASSD有效数、实际资源成本与限制。STATIC赢就如实报告，不强行维护跨图历史有效的主张。
