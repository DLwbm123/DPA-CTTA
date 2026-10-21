# P-DORM与持续状态恢复：全文、源码及跨数据新机制方案

**交付类型：文献与源码静态审阅＋原创研究方案；没有启动任何实验。** 用户最新要求“先别启动实验”优先于旧自主运行授权，包含合成CPU、资格检查、模型调用和GPU。唯一半小时监测保留为只读研究与交付跟进。

结论：两篇不应简单拼接。“分开重置Adam一/二阶矩”“按层恢复”“固定profile跨数据迁移”均已有明确覆盖。当前保留一个可证伪候选：**参数恢复后，以同批输入的成对梯度迁移优化器矩，检验恢复参数与保留历史之间的兼容性。** 这是尚未实现和验证的方案，不是已成立的新算法或收益；梯度迁移本身也有IGT/MARS先例，必须限制创新声明。

## 1. 证据获取与阅读范围

原文均为ICLR2027在审投稿，不能写成已录用。人工完成OpenReview验证后，沿公开页面实际下载PDF和作者补充zip，未绕验证。P-DORM PDF28页，补充129个文件；PSR PDF12页，补充15个文件。原文与补充材料仅私有阅读，不公开重分发，不执行作者代码。

- [P-DORM论文页](https://openreview.net/forum?id=efsBsWqXBY)及[作者补充材料](https://openreview.net/attachment?id=efsBsWqXBY&name=supplementary_material)：重点读正文§1–4、Appendix A的阈值构造、D的优化器语义、F的实现设置、H/I的解释与限制；不是逐行复核全部证明和实验表。目视核PDF第4页公式、20页Adam扩展。
- [Persistent-State Management论文页](https://openreview.net/forum?id=hBGrz9SrlF)及[作者补充材料](https://openreview.net/attachment?id=hBGrz9SrlF&name=supplementary_material)：读§1–7与Appendix A–G的方法、实验和限制，目视核第11页完整漂移归一化与profile。
- 源码重点为P-DORM的Adam完整step、SGD重置分支、TENT接入、base、conf和代表yaml；PSR的core、三个profile、README和manifest。没有声称审完129个文件。

可定位的逐项证据见[证据矩阵](EVIDENCE.json)。后文的源码路径都是作者补充zip内成员名，不是本机私有目录。

## 2. P-DORM实际做什么

它把当前无监督梯度与对应参数张量的更新前方向历史作余弦比较；用各张量自己的历史中心/尺度标准化，再用FIFO背景分布和模块大小分配阈值。触发后保持模型参数，只处理该张量优化器状态，然后当前梯度进入原更新一次。检测量是优化历史不相容的操作信号，不是任务错误标签或因果域变化定位器，作者Appendix H也明确此边界。

默认参数来源：论文Appendix F给k0=2.5、阈值调整±0.5、统计衰减0.9、FIFO100/最少15、数值量1e−4/1e−6。模块大小是先验，不保证更小模块一定更该恢复。同尺寸LayerNorm时该大小分配可能退为统一系数。

**Adam并非全量fresh reset。** Appendix D.1已经规定一阶矩归零、二阶矩乘保留系数、第一矩偏置校正局部重启而第二矩计数保留。相应实现存在于`methods/Optimizer_AdaTTA/adat_adam.py`。原论文默认保留系数0.5，不可把这个分离设计再当原创。

作者在四个分类数据集、多个骨干、六个宿主与不同流上报告了普遍但非逐格必然的改善。它没有由这些实验自动取得当前分割二头、停止的BN尺度路线或新增数据模型的权限。

### 论文—源码需要先解决的契约差异

| 项目 | 实际核查 | 后果 |
|---|---|---|
| Z分母 | PDF第4页为sqrt(V+eps_var)，Adam源码295–296行为sqrt(V)+eps_var_z | 两者不代数等价；复现必须明示paper式与released-code式，不能混用默认 |
| 零梯度 | 正文Algorithm1要求两范数门；Appendix D称零梯度不进入评分/窗口。Adam实际门主要核sensor范数和冷启动，没有同等current-gradient范数门 | 可能出现零梯度仍贡献检测统计的路径；仅静态发现，未执行触发实例，不推断其影响所有结果 |
| 构造器与实际默认 | 构造器保留独立sensor/滞回选项，但`conf.py`实际设optimizer sensor、V612=true、k_dyn=small | 不能仅看构造器误报作者主方法用了独立sensor或滞回 |
| 配置优先级 | 论文与conf为tau_v=0.5；ImageNet-C的`cfgs/imagenet_c/Ours_compare/tent_ours.yaml`写0.4 | 当前公开材料存在配方歧义，不能宣称唯一完整主表复现契约已过 |

这些是需要澄清和对照的版本/实现问题，不是本轮证明作者结果无效。

## 3. 持续状态恢复（PSR）实际做什么

状态含可更新参数、优化器和方法历史；统一monitor→trigger→restore→native update→resume。用一个有标签开发流选三个候选profile之一，冻结漂移中心/尺度、阈值、持续次数、冷却和恢复组；新流部署不读标签、不重新调profile。它不是“设计阶段完全无标签”。

G1–G4按可更新结构的深度或语义stage映射，非任意等参数量四分。MLMP用G1→G1+G2，SAR用G4→G1+G4，TENT用FULL；正常部署只恢复源锚定参数，保留优化器、方法历史、RNG。完整state capture用于反事实重演，不应混作部署每次都清所有状态。

作者报告四个系统提升、一个系统下降；三次跨数据profile迁移为正。但两验证seed只支持作者给出的均值/方向描述，不冒正式显著性；ACDC这里是驾驶恶劣天气数据，不是同名心脏MRI。局部动作排序跨数据甚至可反向，固定profile转移为正不等于逐事件最优选择可转移。

### 补充源码的可复现边界

`anonymous_code/README.md`明确：完整数据运行器和checkpoint不在此包。`profiling_framework/core.py`提供profile/action/metric与adapter协议，但基础`measure_state_drift()`返回空映射；`select_profile_action()`只实现severity映射，不是完整patience/cooldown在线循环。optimizer-only接口也委托给名为`restore_source_active_and_clean_optimizer()`的外部canonical方法，不能从缺失的外部实现核实其只改优化器。

因此已经核到了**实际作者接口/配置源码**，没有取得可直接完整重演的全部GPU运行源码；不能把占位接口当可运行实现。三个profile的标签开发来源及部署preserve语义可核，完整运行契约仍有缺口。

## 4. 旧覆盖与创新边界

| 已有方法/证据 | 已覆盖 | 不能再声称为新贡献 |
|---|---|---|
| [本项目V6](../stratified-reliability-paired-state-v6/REPORT_ZH.md)与`src/dpa_ctta/srg_ctta/method.py:161` | 保留仿射、重建Adam；有限开发数据主差+1.10195pp，但总推进门失败 | “清Adam可改善”不是新发现；BN尺度路线仍停止 |
| P-DORM | 张量级方向异常、动态重置、Adam方向/尺度/偏置校正分离 | 一/二阶矩分开、按组清历史、梯度与动量冲突 |
| PSR | 方法条件化恢复组、冻结profile转移、参数/优化器2×2分析 | 状态分组、按层恢复、简单factorial或恢复后继续 |
| [ASR](https://arxiv.org/html/2603.03796v1) §3 | 自适应何时/何处恢复、Fisher知识恢复 | 通用“选择性重置” |
| [IGT](https://proceedings.neurips.cc/paper_files/paper/2019/file/1dba5eed8838571e1c80af145184e515-Paper.pdf)与[MARS](https://openreview.net/forum?id=NrcKQ3ASLZ) | 梯度历史迁移/相邻参数同批梯度差与方差缩减 | 首次梯度迁移、首次利用同批梯度差 |

**尚未在已核两篇的部署实现中看到的具体增量**：把模型参数恢复视为一次已知外生跳变，在相同输入/目标下估计跳变的梯度位移，并按Adam不同衰减权重迁移一、二阶历史；识别未恢复组受耦合影响的必要性。它仍可能与更广泛优化文献重叠，不能宣称全球首次或已具论文创新资格。

## 5. 方案与验证状态

完整数学、反例、操作顺序和边界在[方案](METHOD_PLAN_ZH.md)，跨数据与强控在[验证设计](VALIDATION_PLAN_ZH.md)。关键判断不是“新分数相关性高”，而是：固定同一参数恢复事件后，历史迁移是否比保留、清空及已有方向修复，改善下一段**先预测后更新**任务性能，并在未调参数据集上成立。

当前状态：`research_plan_only=true`；`experiments_authorized=false`；`implementation_complete=false`；`launch_eligible=false`。没有合成资格、模型/梯度试跑、数据下载、SSH/GPU/真实数据读取。本次仅本地PDF/源码文本、文档和发布检查；下载/阅读/编排未独立计时，不称零成本，既有科学账本不变。

用户要求的全文与源码审阅已进行到上述明确范围；PSR完整runner缺失、P-DORM论文与配置差异仍未解决。交付是一套有推导和停止标准的待验证方案，不能把这些缺口包装成复现成功或新机制有效。
