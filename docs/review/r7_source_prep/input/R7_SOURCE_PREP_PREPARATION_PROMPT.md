# Codex后续任务：SOURCE_PREP启动准备（不执行真实训练）

R7-F1外部复审为R7_REVIEW_PASS_STAGE_I，绑定实现`8f179ce0ca848791793a17dab0606f0bde2b3e07`、证据发布`c53ad20a6c5e5e0caefe019441f10e6f612960ec`。先归档随附复审记录；不重写原NEEDS_FIX/失败记录。

本任务的范围是把已审CPU tensor方法接到可审阅的真实源准备执行层。不是算法重设计，也不是自动启动训练。提交本Prompt只要求准备与程序化验收，不授予GPU查询/使用或真实模型训练权限。

## 1. 保持
A/B/C公式、原四份science字节、FULL/STATIC独立同预算、fit/cal/val用途、查询不更新状态、24/9/至多12矩阵保持。不要重跑旧166项历史回归。F1已经关闭，除新增代码影响到其行为，不重新扩展绑定威胁模型。

## 2. 真实source登记
从已有权限下的登记与元数据查找source manifest、患者/眼别分组依据、checkpoint来源及已知预训练暴露。检查fit/cal/val无组交叉、最小数量、与target不重叠。先冻结组划分再解码像素；不能把文件名或每张图任意当成独立患者。
可以读取合法清单与记录并计算已确定源文件的只读SHA256；这与模型加载/像素解码分开计数。本任务不解码真实RGB/mask，不反序列化真实checkpoint，不将其送入模型。未知来源/摘要保持PENDING，不能用目标数据补齐，不能猜测私有目录或身份。
原checkpoint文件hash、有效主干tensor hash、训练产物hash分别记录。尚无法得到的有效tensor身份等真实加载后再填，不伪造。

## 3. 执行层实现
单独增加SOURCE_PREP runner与合法source读取器，默认disabled。保留已审tensor后端，明确模型/模块/潜状态的dtype和device路由；当前CPU实现不是GPU-ready。GPU支持代码可以实现，但未在实际GPU运行的检查必须为NOT_RUN，不能用CPU测试冒充。
真实入口先核对审阅scope/最终代码与science/source/资源绑定，再允许资产解码或设备使用。SOURCE_PREP与TARGET各scope隔离，禁止自动跨阶段。
不静默改变训练步数、截短预算、替换oracle支持组、改rank或遇错重试。将源端oracle、basis/VJP、scaler、六份模型fit/cal/val、恒定方差和oracle-query诊断分别计数。有限时限、输出上限、首失败、源只读、进程清理均需明确。
可信训练包保存完整context；fresh loader使用独立保存的expected，不能从candidate自签。真实source文件验证由执行层负责，BOUND本身不是授权。

## 4. 验收与交付
程序化CPU测试覆盖清单与split、错误scope/摘要先拒绝、真实IO接口的小型合成文件、源与目标隔离、设备边界的CPU可测部分、budget/首失败/产物绑定和旧C保持。对最终新增执行补丁运行适当的R7回归，量和成本如实记录，不要求为纯文档更新重跑方法。
交付SOURCE_BINDING.json（有证据才BOUND）、SOURCE_SPLIT审计、设备/存储/成本拟议配置、默认禁用的执行授权模板、窄patch、日志、REVIEW_INDEX与DELIVERY。设备ID、资源授权未知时null，不沿用旧R6授权。
若缺真实source字段，继续完成能做的执行层与合成验收，但列明阻塞。不要以缺源数据为由重设计三个方法。

## 5. 停止条件
通过准备与程序化验收后报告：
R7_SOURCE_PREP_IMPLEMENTATION_READY_FOR_REVIEW
stage_I_review=PASS
source_prep_execution_layer_review=NOT_RUN
source_binding_status=<按证据填写>
real_source_training_started=false
real_target_execution_started=false
SOURCE_PREP=NOT_RUN
TARGET_SCREEN=NOT_RUN
TARGET_MECHANISM=NOT_RUN
TARGET_EXTENSION=NOT_RUN
execution_authorized=false

未通过则报告具体准备阻塞。不得自行签发新增执行层PASS；不得执行真实源训练、GPU smoke或目标实验。交付后停止。
