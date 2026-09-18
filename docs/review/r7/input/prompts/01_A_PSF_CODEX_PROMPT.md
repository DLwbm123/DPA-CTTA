# A组 Codex独立任务：概率子空间过滤

本Prompt必须与00_MASTER_CODEX_PROMPT、COMMON_PROTOCOL、TRACK_A_PSF_METHOD.md和specs/A_PSF.json一起使用。总Prompt的Stage I边界完整适用：只实现和程序化CPU验收，不读真实资产，不GPU，不训练source/target。

## 工作方式
基线00ccb401942e9f1d3fed48ccb7fb3777c87714b8；共享骨架由协调任务先提交。独立分支experiment/r7-a_psf-v1，独立worktree。只改本组目录和测试；共享接口变更通过协调任务，不覆盖其他组。论文来源见PROVENANCE；本组是借鉴后的设计，不是官方复现。

## 本组硬要求
实现full16×16 F与完整P。所有矩阵更新用float64 Cholesky solves；F不是diag，否则P0/Q/R全diag会让完整协方差退化。
B16依据source pooled-logit GGN surrogate和probe预测协方差截断，不能偷偷用随机/PCA代替；stageI只合成J及随机网络少量J验证，真实1024VJP在未来SOURCE_PREP。
双代码本只参与观测，不替换患者特征。保持source-patient tokens为瞬时变量。NLL相对source proxy latent，不声称真实domain latent识别。
A_STATIC要单独source训练且每visit reset，A_ISO_OBS只是future同checkpoint部署消融。
额外测试source基与独立SVD、fullF产生非零offdiagP、高R极限、tau校准对R²因子、因果性、target不调用autograd。

## 完整交付
实现该组observer、state update、source fit/cal训练接口、FULL/STATIC、对应mechanism ablation、保存/载入、错误/状态清理。不得只给一份离线loss函数就称方法完成。
给出METHOD_CONFORMANCE、模块参数量、准确forward/latent-solve成本、程序化source微训练梯度、随机完整分割器接入、source/target/GT/未来隔离测试。
最终固定本组整合SHA后跑实际CPU套件，记录两个环境能实际完成的测试；不伪造未跑环境、不以共享math tests代替组实现。
保持spec原件不变，运行绑定独立。缺真实source元数据写PENDING而不偷读数据；真正算法缺项则INCOMPLETE。

完成后停在R7_A_IMPLEMENTATION_READY_FOR_REVIEW，external_review=NOT_RUN；SOURCE_PREP/TARGET_SCREEN/MECHANISM/EXTENSION均NOT_RUN。未通过则实际INCOMPLETE。禁止自签PASS或继续实验。
