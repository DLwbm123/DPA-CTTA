# 论文来源与本提案的明确差别

## 材料等级

用户提供：

- `CTTA_2024_2026_Medical_Image_Adaptation_Review.md`：自述为摘要／题录级方法地图，证据A也不等于全文、证明和实验独立复核。
- `ICLR_2026_CL_论文总结.md`：明确没有逐篇精读方法、实验与附录，部分venue存在待核对状态。

本包以上述总结为路线来源，补核一手摘要与现有仓库接口；**没有声称本次对所有候选论文完成全文复现审查**。所需算法已在METHOD_SPEC中完全给出，缺少某篇PDF不会迫使Codex猜公式，也不构成源数据或预训练资产的获取许可。

## 逐框架映射

| 框架 | 论文确有的方向 | 本项目自己的设计，不可归为原文公式 |
|---|---|---|
| T | BayesTTA：在线类别条件分布与GDA用于预测/监督；SicTTA总结：类内紧致密度与单图可靠性 | 固定给定分割checkpoint坐标、二值区域低秩密度、逐维log证据归一化、0.5修正系数与回退、ISO/DIAG控制 |
| U | KeepLoRA：预训练/历史方向保护；SplitLoRA总结：稳定-可塑子空间划分 | BN affine而非LoRA，区域feature-Jacobian、实际Adam位移近端解、同范数控制；没有原论文理论的自动继承 |
| S | MoIE：增量专家与知识复用；ReservoirTTA：风格路由、域相关模型状态 | 容量3的小BN/Adam/区域统计库、64维只读descriptor、固定无标签建槽规则；不实施MoIE的KTI、不复制整模型、不使用KFF额外源统计 |
| M | LCA：classifier与更新后backbone的兼容性问题 | 当前图前后位置Procrustes及mean/M2/basis坐标迁移；这不是LCA原始损失或模型合并 |
| G | SPEGC：语义提示、图聚类与结构监督 | 用既有区域PCA定义局部度量、4邻边、固定mirror/prox、Bernoulli包含投影；没有其prompt池、OT边稀疏化或原端到端图solver |

## 一手入口

- BayesTTA: https://arxiv.org/abs/2507.08607
- KeepLoRA: https://arxiv.org/abs/2601.19659
- LCA: https://arxiv.org/abs/2603.09888
- MoIE: https://papers.miccai.org/miccai-2025/0826-Paper0652.html
- ReservoirTTA: https://arxiv.org/abs/2505.14511
- SPEGC: https://arxiv.org/abs/2603.11492
- SicTTA、SplitLoRA：当前以用户摘要为方向补充；本批不使用未核对的原损失公式或默认参数。

不要根据文件所属文件夹声称论文已录用；本包不依赖venue判断。没有把classification的softmax替换名词后直接用于单/双Bernoulli分割。

## 现有实现锚点

- 基础代码： https://github.com/DLwbm123/DPA-CTTA/tree/f52f132e4be576ea871467432a6f4b12337209a5
- C/PCA host: `src/dpa_ctta/r1/host.py`
- 区域采样与grid: `src/dpa_ctta/r1/region_memory.py`
- C来源和固定GraTa依赖: `src/dpa_ctta/b1_host.py`
- R2最终IO修补: `docs/review/r2_io_continuation/LIVE_DIRECTORY_REPAIR.md`

不复制未知许可证的论文代码。优先在本项目写明确公式、复用现有MIT等已登记依赖；确实借用第三方代码时保留许可证及固定commit，记录差异。不把作者摘要报告的性能作为本包方法性能。

## 资源约束不变

禁止恢复旧源DD／源query流程；禁止DO-ALL锚点、GOLD源原型、额外生成模型或VLM文字编码器进入本次配置。允许给定checkpoint的只读副本，这是现有权重的复制，不是访问源数据或重新训练。

没有保证五个候选都有效；其共同作用仅在于把区域memory作为监督、位移、状态、坐标或图几何的一部分，逐条验证其必要性。
