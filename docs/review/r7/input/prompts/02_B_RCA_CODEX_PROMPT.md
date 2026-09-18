# B组 Codex独立任务：递归预测—稀疏修正

本Prompt必须与00_MASTER_CODEX_PROMPT、COMMON_PROTOCOL、TRACK_B_RCA_METHOD.md和specs/B_RCA.json一起使用。总Prompt的Stage I边界完整适用：只实现和程序化CPU验收，不读真实资产，不GPU，不训练source/target。

## 工作方式
基线00ccb401942e9f1d3fed48ccb7fb3777c87714b8；共享骨架由协调任务先提交。独立分支experiment/r7-b_rca-v1，独立worktree。只改本组目录和测试；共享接口变更通过协调任务，不覆盖其他组。论文来源见PROVENANCE；本组是借鉴后的设计，不是官方复现。

## 本组硬要求
实现v=U32z，appearance差分不能换成raw patient image差分。W/G有源端谱规范化，target冻结后校验所存eta。
固定5次ISTA，能量同时包含residual/kappa²、lambda L1和mu L2；不要只改loss漏改Lipschitz步长。
source时autograd穿过5次迭代，target时no_grad手写梯度，无网络backward、Adam或动态早停。
B_STATIC独立训练且第一访与每访diff=0；B_PRED_ONLY是同FULLcheckpoint的delta=0操作，不冒充重训参照。
额外测试H=I的闭式近端解、固定5步计数、kappa温度、第一张状态、只有d变化能进入delta-input、字典column规范化与source/query隔离。

## 完整交付
实现该组observer、state update、source fit/cal训练接口、FULL/STATIC、对应mechanism ablation、保存/载入、错误/状态清理。不得只给一份离线loss函数就称方法完成。
给出METHOD_CONFORMANCE、模块参数量、准确forward/latent-solve成本、程序化source微训练梯度、随机完整分割器接入、source/target/GT/未来隔离测试。
最终固定本组整合SHA后跑实际CPU套件，记录两个环境能实际完成的测试；不伪造未跑环境、不以共享math tests代替组实现。
保持spec原件不变，运行绑定独立。缺真实source元数据写PENDING而不偷读数据；真正算法缺项则INCOMPLETE。

完成后停在R7_B_IMPLEMENTATION_READY_FOR_REVIEW，external_review=NOT_RUN；SOURCE_PREP/TARGET_SCREEN/MECHANISM/EXTENSION均NOT_RUN。未通过则实际INCOMPLETE。禁止自签PASS或继续实验。
