# R7 三组独立方法：实现—审阅—实验

## 执行决定
同时准备三组方法，但不拼接成一个大模型：
- A_PSF：双代码本观测 + 预测相关子空间 + 高斯持续过滤；
- B_RCA：权重/adapter坐标递归预测 + 稀疏势能修正；
- C_RBE：query条件化低秩证据 + 噪声课程方差 + Huber稳健状态估计。

本包定义的是受2026论文启发的研究版本，并非原论文代码的完整复现。额外源端训练合法但要计费；目标阶段不访问源图像/标签、目标标签或未来样本。C仍是对照，不再限制所有方法只能围绕其BCE做微调。

## 当前任务
只执行Stage I。交付三组实现、源训练接口、在线接口、数值和泄漏测试、随机完整网络接入、真实元数据dry-run、窄patch、原件摘要及审阅索引。尚无代码审阅PASS，更无源训练或GPU授权。

每组可独立审阅；共享骨架先合并再并行worktree。不得把一个未完成组填成READY来等实验救场，也不让一组bug迫使另外两组改方法。

## 方法分工
| 组 | 主状态 | 当前证据 | 核心更新 | 主要否证 |
|---|---|---|---|---|
| A | 16维均值+16×16协方差 | 代码本条件化Gaussian观测 | 因果信息过滤 | FULL不能胜过同预算STATIC，方差无校准价值 |
| B | 32维适配码+前一appearance码 | 源学习dictionary residual | 稳定线性预测+5步ISTA | 能量降但任务无改善，修正不如predict-only |
| C | 32维适配码 | 64patch×8维异方差证据 | 3步Huber IRLS | 常方差已解释收益，不确定性不识别代理误差 |

所有新方法只调制up1/up3，保持患者原特征通路；每图2backboneforward、0在线网络backward。它们不把患者形态做时间平滑。是否学到真正的成像/结构分离需要实验，不能从命名推断。

## 源端准备（未来审阅后）
共享：预登记source fit/cal/val、光度模拟器、192个source-proxy干预码、B/C的rank32基、A的rank16预测相关基。仅fit部分能拟合字典/任务模块；cal只校准，val只评价。
每组FULL与STATIC分别训练1000个4时刻episode steps，然后256校准steps。Source STATIC在训练时也清历史，用于隔离额外源端准备和持续机制。所有参数/调用/存储单独计费。

这一步是准备新方法所需能力，不冒充原C信息预算内的免费改进。没有合法source数据或新模块划分，source stage阻塞；不准使用target伪source。

## 目标实验
| scope | arms×streams | 轨迹 | 名义backboneF | 网络backward/Adam |
|---|---|---:|---:|---:|
| SCREEN | C_BASE/C0+三组FULL/STATIC ×0/1/4 |24|122913|5853|
| MECHANISM，上限 |A_ISO_OBS/B_PRED_ONLY/C_CONST_R ×0/1/4|9|35118|0|
| EXTENSION，至多2组 |C_BASE/C0+两组FULL/STATIC ×2/3|12|66334|3902|

全部scope独立授权，不自动衔接。每条1951到达内容，1695评分；主序域等权、回访单列。SOURCE费用、Jacobian、calibration、smoke、MLP/matrix成本未包含在目标表。

预测预算仅用于预注册，不等于已实测。未来目标smoke每设备可包含OLD_C4、C_BASE4、C0_4以及六个新FULL/STATIC各4访问：116F、8backward/Adam，latent更新另记；实际入口须CPU计数核验后绑定，不照此数伪造运行结果。

## Gate政策
实现/标签隔离/身份/计算完整性是硬要求；不复用R6的+.5pp全AND筛选。SCREEN结束生成多维研究评审材料：FULL−C说明总体实用增量，FULL−本组STATIC说明持续机制增量。允许均值接近C但机制证据明确的组进入一次有限诊断；不自动宣布有效。临床风险、ASSD、子域、顺序、计算代价并列。

一次SCREEN最多提名2组扩展；也可不提名。提名理由必须能描述新实验会改变什么判断，不按小数挑最有利gate。部署消融不等价于最优重训对照，后续论文必要时补独立重训。独立患者确认、多seed和其他任务不在此次自动预算内。

## 审阅顺序
1. Codex交付Stage I实现，停在R7_IMPLEMENTATION_READY_FOR_REVIEW（含各组独立状态）。
2. 外部审阅固定SHA；仅对通过的共享/组实现记录PASS。
3. 用户单独授权SOURCE_PREP，冻结source资产/split/模拟器/设备/caps，跑源准备。
4. 检查准备产物、heldout source报告与无target访问证据，绑定权重后再授权TARGET_SCREEN。
5. 实验结束交付，不自动跑机制/扩展、不自签reviewPASS。

不存在要求用户现在批准GPU。资源位置由已有项目元数据解析，不复用历史GPU号或receipt。

## 文档优先级与缺失处理
精确方法以specs/*.json和三个METHOD文档共同约束；出现真正冲突先报告，不能自己选择使测试好通过的版本。数学参考只检小矩阵公式，不能替代完整host或证明原论文复现。原提案字节归档，运行绑定独立存放；状态变更不重写设计原件。

当前已交齐所有依赖：COMMON_PROTOCOL、三组METHOD、四个prompts、四个spec、数学参考和测试、来源、目标矩阵与manifest。不需要不存在的math_history；本包自有测试历史另存。全部SHA以MANIFEST为准。
