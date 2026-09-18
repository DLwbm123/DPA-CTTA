# C组 Codex独立任务：可靠性加权基证据

本Prompt必须与00_MASTER_CODEX_PROMPT、COMMON_PROTOCOL、TRACK_C_RBE_METHOD.md和specs/C_RBE.json一起使用。总Prompt的Stage I边界完整适用：只实现和程序化CPU验收，不读真实资产，不GPU，不训练source/target。

## 工作方式
基线00ccb401942e9f1d3fed48ccb7fb3777c87714b8；共享骨架由协调任务先提交。独立分支experiment/r7-c_rbe-v1，独立worktree。只改本组目录和测试；共享接口变更通过协调任务，不覆盖其他组。论文来源见PROVENANCE；本组是借鉴后的设计，不是官方复现。

## 本组硬要求
query-conditioned H_i与o_i来自当前token/appearance，不给在线接口传入source style参数或GT。64×8=M512，mean归约不能漏掉512而改变ridge强度。
主fit时R=I，仅任务/basis/observer训练；随后source-cal噪声课程只训练variance meta-model。不得联合更新冻结task模块来降低cal loss。
固定3次Huber IRLS，lambda=.1、delta=1；用Cholesky，零residual权重1，异常hard fail，不动态jitter。
C_STATIC独立训练；C_CONST_R的8维常量是source-cal geometric mean，绝不能来自完整目标流。
额外测试IRLS与独立ridge/稳健参考、离群权重、curriculumseed、cal梯度隔离、variance边界、latent更新零target网络反向、无前门因果/Dirichlet伪声明。

## 完整交付
实现该组observer、state update、source fit/cal训练接口、FULL/STATIC、对应mechanism ablation、保存/载入、错误/状态清理。不得只给一份离线loss函数就称方法完成。
给出METHOD_CONFORMANCE、模块参数量、准确forward/latent-solve成本、程序化source微训练梯度、随机完整分割器接入、source/target/GT/未来隔离测试。
最终固定本组整合SHA后跑实际CPU套件，记录两个环境能实际完成的测试；不伪造未跑环境、不以共享math tests代替组实现。
保持spec原件不变，运行绑定独立。缺真实source元数据写PENDING而不偷读数据；真正算法缺项则INCOMPLETE。

完成后停在R7_C_IMPLEMENTATION_READY_FOR_REVIEW，external_review=NOT_RUN；SOURCE_PREP/TARGET_SCREEN/MECHANISM/EXTENSION均NOT_RUN。未通过则实际INCOMPLETE。禁止自签PASS或继续实验。
