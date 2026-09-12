# R1 Stage I Implementation Report

状态：**R1_IMPLEMENTATION_READY_FOR_REVIEW**。本次交付完成代码、CPU程序化测试、科学配置冻结、dry-run清单和去身份审阅材料；没有外部review结论或实验执行授权。

- 完整implementation commit：[f67a4d91c4337fe53e350cc2cf621ab703b9d2c1](https://github.com/DLwbm123/DPA-CTTA/commit/f67a4d91c4337fe53e350cc2cf621ab703b9d2c1)。
- 基线：`0ef9d593a196e18f38295edc5b919372e297671b`；分支：`experiment/r1-recovery-target-subspace-v1`。
- 实现提交仅新增18个文件、1575行，没有改写已有C路径。审阅日志、索引、patch、源码zip和本报告属于后续交付提交；实验代码仍唯一绑定上述implementation SHA。
- 固定外部代码：CTTA Suite `dbff0d985c6c95345d9fb78f5b1daef57b392564`，GraTa `33ae20d664f305af34739ec54a5bec7da53ffa0b`。

## 实现范围

| 实验臂 | 已实现行为 | 每次访问的前向 / backward / Adam |
|---|---|---|
| C | 继承原C的normalized_step，保持弱增强、强增强、teacher、更新及预测顺序 | 8 / 1 / 1 |
| C_PER256 | 在第257、513、769、1025、1281、1537、1793张的teacher前恢复初始BN affine，清空Adam状态 | 8 / 1 / 1 |
| C_SENS | 当前图像三个独立no_grad探测；共享未擦除区域上的Bernoulli敏感度、EMA与历史最佳值触发恢复 | 11 / 1 / 1 |
| C_PCA_GLOBAL | 相同可靠token池汇入一个全局统计bank；旧基底约束当前强分支 | 8 / 1 / 1 |
| C_PCA_SHUFFLED | 四个bank；随机分配保留有效token配额，memory与loss使用不同局部随机盐 | 8 / 1 / 1 |
| C_PCA_REGION | OD/OC各前景与背景四个bank；区域分别采样、积累和计算投影残差 | 8 / 1 / 1 |

三种PCA臂复用原C已有前向的`seg_head`输入特征；不增加网络、模型权重、前向或更新步。统计位于CPU float64，只保留累计count、mean、M2及中心/基底快照；每16张贡献图更新一次，至少128个向量，rank不超过8。当前图像仅在更新和预测之后入库，新基底最早下一次访问可用。强分支特征保留梯度，中心/基底和归一化分母按协议detach，额外损失权重为0.05。

零范数向量在shuffle分配前剔除，保证REGION与SHUFFLED的有效bank配额可比较；同时分别记录采样数、零向量数和入库数。四个语义区域允许OD/OC位置重叠，单位是(channel, position) token，1024个空间位置对应2048个候选token。

恢复只改变初始BN affine、Adam状态、梯度和segment age；保留全局访问序号、累计调用数及随机序列。C_SENS触发图像继续完成原C更新，不跳图，不使用分位数筛选。特征hook只在当前步缓存，`finally`清理，不跨访问保留图像或原始特征。

未来执行入口、有限任务调度、资源预算及离线标量汇总均已实现。默认执行检查在权重加载和GPU访问之前拒绝启动；没有自动等待review、重试或追加实验的后台流程。方法选择阈值只作为汇总解释标记，不是效果准入gate。具体函数、行号和测试见[REVIEW_INDEX](REVIEW_INDEX.md)。

## 科学冻结与资产边界

[science config](../../../configs/r1_science_v1.json)保留任务包`r1_science_v1.proposed.json`的原始字节，包括`proposed_for_code_review`状态。SHA-256为：

`e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2`

六臂、四种顺序、seed 20260907、1951组、remaining_dev 1695组，以及原C的41个BN层/19136个可训练affine标量、Adam设置、恢复参数和PCA参数均未更改。完整摘要见[SCIENCE_DIGEST](SCIENCE_DIGEST.json)。

服务器既有B3登记文件仅用于解析目标身份/域/子集/路径metadata和唯一源checkpoint位置。新私有登记剔除了旧proxy、source查询及无关history字段；该登记摘要为`8620ed1b0d225ad921d33998123dd322a30c9ed67c2d3de65521100f429c5d4e`。1951组包含1695 remaining_dev、128 legacy、128 p1_extension，已用metadata核对。A/C0历史标量仅解析了路径，没有读取结果。

服务器CPU测试只读取登记的源checkpoint权重，并使用程序化输入。源图像、源mask、源代理、源原型、额外预训练组件、真实目标图像及真实目标mask访问数均为0。没有查询、初始化或运行GPU，没有正式记录或真实效果分数。没有读取真实资产内容来检查可用性。详情见[ASSET_RESOLUTION](ASSET_RESOLUTION.json)。

两个用户提供的论文PDF已读取相关方法段落，文件摘要及“原方法 / 项目实现 / 未采用 / 未明确”对照见[METHOD_PROVENANCE](METHOD_PROVENANCE.md)。论文没有公开复制；可选背景论文未独立全文核验，不作为新增方法事实。生命周期及旧/新基底时序见[STATE_LIFECYCLE](STATE_LIFECYCLE.md)。

## CPU实测结果

| 环境 | 权重与输入 | 结果 |
|---|---|---|
| 本地现有Python 3.12.9 / Torch 2.6.0 | 固定完整网络、程序化随机权重和程序化输入 | 17通过，工具输出约15.9秒；未另存原始日志 |
| 服务器原项目Python 3.10.6 / Torch 2.2.1+cu121 | 登记的真实源checkpoint权重、程序化输入；CPU两线程 | **17通过、0失败、0错误、0跳过、exit 0** |

服务器真实[原始日志](CPU_TEST_LOG.txt)和[机器可读结果](CPU_TEST_RESULTS.json)均已保留。unittest计时69.482秒，外层测试包装器计时69.691秒；`cuda_initialized=false`。Torch版本含cu121仅说明安装包构建版本，不代表使用过GPU。CPU入口将`CUDA_VISIBLE_DEVICES`置空，并令CUDA初始化调用直接报错。

17个测试覆盖：

- 嵌套擦除面积、共享像素集合、Bernoulli熵、空区域、局部随机种子不扰动全局RNG。
- EMA严格阈值、50步horizon、floor、控制器清空与周期恢复的准确边界。
- 累计协方差等价性、基底快照延迟、正交投影、rank上限、零rank和eigh异常显式传播。
- 网格位置、OD/OC重叠、可靠性条件、shuffle配额、零向量处理、基底无梯度和强特征有梯度。
- 完整固定网络上旧C与新C连续两步的logits、梯度、affine/Adam状态和RNG等价；实际32维特征接口上ready PCA损失及强分支梯度有效。
- 六臂toy host的实际调用数、独立状态、恢复后参数对象/Adam/RNG归属，以及额外损失权重为0时与C一致。
- host拒收label，评价调用不改变host状态；被禁source reader一旦调用便使测试失败。
- 默认CLI在权重/GPU访问之前拒绝；错误摘要/提交/授权、重复任务、错误身份、错误计数和不完整轨迹拒绝。
- 完整24条程序化标量轨迹的CPU汇总及截断失败路径；这些标量不是实验结果。

复现使用已有固定依赖环境，设置`PYTHONPATH`含项目`src`、`tests`和所需已有依赖；`DPA_CTTA_BASE_ROOT`、`DPA_GRATA_ROOT`指向固定外部代码，`CHECKPOINT`仅指向登记的源权重，`CHECK_OUTPUT`指向私有结果文件。为保持进程参数中性，实际脚本路径通过环境变量`RUN_FILE`提供：

```sh
python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'
```

`RUN_FILE`的逻辑入口是`scripts/check_r1_cpu.py`，不调用实验启动脚本。公开命令省略私有绝对路径；真实测试结果来自上述服务器日志。

第一次metadata准备调用误用了缺少Torch的系统Python，在import阶段出现`ModuleNotFoundError: No module named 'torch'`。改为原项目现有解释器后完成；没有安装依赖或修改环境。该次没有开始测试、读取权重或运行GPU，不计为方法失败或正式重试。

## Dry-run与未运行项

[DRY_RUN_MATRIX](DRY_RUN_MATRIX.json)列出完整24条任务，全部为NOT_RUN，GPU分配为空。正式计划为46824条记录、398004次前向、46824次backward/Adam；未来每GPU smoke预算为118次前向、14次backward/Adam。以上均为静态预算，不是已发生的调用数。

| 未运行项 | 状态 / 原因 |
|---|---|
| GPU mechanical smoke、GPU确定性和峰值显存 | NOT_RUN；本阶段未授权GPU |
| 真实目标图像/mask可读性、预处理与mask延后读取的真实流验证 | NOT_RUN；本阶段禁止读取真实内容 |
| 24条1951组正式轨迹及真实分数 | NOT_RUN；本阶段未授权正式实验 |
| 真实数据上的恢复次数、PCA激活/ready比例、尾部风险和效果比较 | NOT_RUN；不能从toy测试推断 |
| A/C0历史标量身份核验及次要参照重算 | NOT_RUN；只完成路径解析及CPU程序化路径测试 |
| GPU吞吐、2小时单轨迹与24小时整体预算可达性 | NOT_MEASURED；静态预算不构成实测保证 |
| 外部review及阶段II执行许可 | PENDING_EXTERNAL_REVIEW；本次不生成通过结论或授权 |

没有新增实验、效果gate、后台任务或监测自动化。[execution defaults](../../../configs/r1_execution.defaults.json)保持`enabled=false`、审批字段为null、GPU列表为空、worker为0、后台禁止。下一步必须由外部review和新的用户指令决定。

## 交付范围

- [完整implementation patch](implementation.patch)是基线到`f67a4d91...`的全部差异；[diff stat](DIFF_STAT.txt)列出18个新增文件。
- [源码zip](r1_source.zip)只封装该implementation提交中的新增R1文件；需要指定基线仓库及固定外部依赖，不是含权重/数据的独立安装包。
- 本报告、[REVIEW_INDEX](REVIEW_INDEX.md)、方法溯源、状态生命周期、配置摘要、dry-run、真实CPU日志和[DELIVERY](DELIVERY.json)可公开。
- 私有绝对路径、目标身份清单、原始数据、源checkpoint、论文PDF、运行凭据及任何真实逐图像结果不进入本次公开材料。

最终状态：**R1_IMPLEMENTATION_READY_FOR_REVIEW**。本阶段停止，等待外部review。
