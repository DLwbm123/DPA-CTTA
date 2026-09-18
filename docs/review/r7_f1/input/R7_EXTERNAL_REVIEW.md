# R7 外部源码审阅

## 结论

**R7_REVIEW_NEEDS_FIX。** 三组算法核心在本次检查中没有发现阻断性公式或数值问题。整体尚不签发实施 PASS，原因是唯一一项共享部署身份缺口 **R7-F1**。不要求重写 A/B/C，不修改四份科学规格，不新增研究臂，不启动真实源训练或目标实验。

- 审阅实现：`82aca2080bb80c019b0d39b4c95d21a6ba2cff3d`。
- 证据发布：`7f129b484a5ab2b0af3efdf2ccea0eee1228969c`。
- source 资产绑定：PENDING；这是原 Stage I 允许保留的状态，并非本次拒绝的原因。
- 本次只提供源码审阅，不替代 SOURCE_PREP 或任何 TARGET scope 的执行授权。

## 1. 证据层级与边界

已读取原始本地设计 ZIP、共享协议、三组方法规范及公开生产源码、对应测试、两端日志和验收摘要。原 ZIP 的24项 MANIFEST 本地核对通过；四份规格摘要与公开 SCIENCE_BINDING 一致。这个核对不冒充对远端全部归档文件再次逐字节验证。

两端提交日志分别记录47项实现测试与22项独立参考测试通过。最终本地 Python3.12.9/Torch2.6.0，服务器 Python3.10.6/Torch2.2.1+cu121。两端每次完整实现套件的计数为464主干前向、36 Tensor.backward、2 autograd.grad、24 Adam、6 AdamW。79次主干前向使用随机完整ResUNet，其余为小型程序化fixture；不把464次都叫完整ResUNet。

我没有访问本地 `/Users/...` 私有交付ZIP，没有SSH服务器，没有重跑完整47项，也没有运行真实源/目标数据。独立探针使用Python 3.13.5 / Torch 2.10.0+cpu / NumPy 2.3.5。7个用于探针的源码文件逐字节重建后，其Git blob均与固定实现一致。

探针中用 identity stub 替代唯一的旧图像预处理导入；矩阵与latent模块不使用它。主干集成只使用标明TinyBackbone的三卷积小型网络，不是ResUNet。fresh inference loader执行原函数AST，未执行prepare_tensors或source数据管线。

最终主探针12个方法执行无运行错误，其中11个是正确性检查，另1个是记录当前不安全接受行为的观察探针；不能据此说“全部绑定检查通过”。另加的2个“应拒绝不兼容主干”回归均失败，退出1，正好复现F1。原参考22项在当前环境另行重跑通过。

最终主探针25次TinyBackbone前向、7次Tensor.backward、0次VJP、0优化器；反例拒绝回归另2次小主干前向。它们是合成CPU验证，不是新源/目标实验；这些计数是对应最后一次运行，不是把先前重复开发运行合并为独立覆盖。

## 2. 三组算法检查

| 对象 | 独立检查 | 结果 |
|---|---|---|
| A高斯过滤 | 96个稠密SPD/非对角F病例，维度2/4/8/16，对照独立NumPy解 | 均值最大绝对差约6.54e-14；协方差2.78e-14 |
| A预测子空间 | 12个Jacobian病例，对比预测空间截断协方差而非特征向量符号 | 最大绝对差约2.83e-11 |
| B稀疏修正 | 128病例，kappa=.25/.7/1/4，恰好5步，能量及状态对照 | code最大绝对差约1.33e-15 |
| C稳健融合 | 96个异方差/离群病例，含512条观测，恰好3步IRLS | code最大绝对差约2.21e-12 |
| 可微源序列 | 三组各四步latent unroll，程序化损失 | 关键新模块梯度有限且非零；不是完整源训练证明 |
| 校准隔离 | A/B只cal_raw；C只reliability四个参数tensor | 通过 |
| STATIC/因果/事务 | 清历史、state不就地修改、保存恢复、最终前向失败不提交 | 同主干的合法路径通过 |
| FiLM | 零输出精确一致、零点导数可达 | 通过 |

A的公式不是完整RP-GSSM；B是有限五步修正，不是精确argmin；C不是前门因果识别。源码保留了这些边界。数值符合公式不代表三组已有真实分割增益。

源码阅读还确认：fit/cal/val来源与oracle分开，query不参与状态更新；FULL/STATIC分别初始化和训练；校准冻结任务参数；目标接口只接收当前图像。公开完整网络测试检查了各组真实随机ResUNet的一次微训练、校准与两次在线访问以及原主干不变。这部分是提交方证据，不是我的完整网络重跑。

## 3. R7-F1：训练产物/在线状态没有绑定完整分割环境

**优先级P2；真实实验产物投入推断之前必须补齐。**

### 位置

- `r7_shared/host.py::Method.digest`：仅对group/static和method.state_dict摘要。
- `OnlineHost.save_state/load_state`：使用上述method摘要；C0为None。未包含segmenter或实际源主干的身份。
- `preparation.prepare_tensors`：源训练模型输出的binding仍是m.digest。
- `preparation.inference_from_tensors`：校验module摘要和B步长，却未验证被传入的segmenter是否正是生成该basis/observer/oracle的源主干。

### 实际反例

1. 创建一个带原Segmenter包装器的程序化主干和B模块，模块/基/scaler冻结。
2. 处理一张程序化图并保存状态。
3. 建立相同结构的另一主干，只把分割头bias增加1，其他输入及method字节相同。
4. 将旧状态加载到新主干：**当前代码接受**。
5. 原fresh inference loader同样接受不兼容主干和原method包。
6. 下一图最大logit差约 **1.00000119**，而method摘要仍相同。

这不是精度容差问题。basis、观察器、oracle和latent状态是相对于特定主干及预处理定义的；分割头也是A预测协方差基所依赖的预测映射的一部分。方法权重相同不代表推断环境相同。

反例是人工构造的，不是证明已经发生真实checkpoint混用。真实source尚未读取，所以当前CPU结果并不因此失效。它说明当前接口缺少拒绝这种混用的依据。

### 最小修复

新增一个版本化的完整推断context，至少包含：实际冻结主干参数/必要buffer的规范摘要、观察投影、BN/预处理/FiLM接口标识、method摘要（含basis/scaler/calibration）、组与FULL/STATIC、部署消融。源训练包保存context；fresh load和state restore都以可信包中的context核对当前segmenter，不能从当前segmenter重新自签一个摘要当作预期值。

实际源checkpoint文件哈希未来需要额外绑定；Stage I用随机完整模型tensor摘要即可测试，不读取真实checkpoint，也不伪造source绑定。C0同样绑定主干环境，不再以None作为完整身份。

摘要可在包装器创建/载入时计算并缓存，必要时做轻量版本检查；不要求每张图重哈希数千万参数。正例应接受不同对象但相同字节的合法副本，不能用Python对象id作为跨实例身份。

不能只在文档中加一个source_checkpoint字段：必须进入上述真实加载判定。旧不完整state包应明确拒绝或单独标不支持，不能静默补签。

## 4. 不列为本次算法阻断的问题

- SOURCE_BINDING_PENDING是计划允许的状态；不能拿目标数据补齐源数据。
- 目前Segmenter显式CPU初始化，输入及小矩阵也默认CPU；真实入口统一disabled。这是已实现CPU tensor backend，不是已经验收的GPU执行端。后续SOURCE_PREP应在另行授权前明确设备迁移、源IO及有限执行封装，不可仅把enabled改成True后声称GPU可运行。
- 本地最终日志含NumPy参考计算RuntimeWarning。本次Linux探针与提交服务器没有复现该警告；我没有证明警告来源，不将其写成已确认的预期异常。保留日志，不能声称无warning。现有生产finite校验及独立有限输出对照没有显示数值不一致。
- report.review是接受调用方标量的描述性汇总器，不是已完成的目标账本核验入口。后续目标执行层需显式绑定Dice单位、完整内容/四域/轨迹覆盖、计数可实现性。这里不因尚未授权的TARGET执行层未完成而要求现在重写分析器。

## 5. 修复后验证与下一步

只补共享绑定、相应测试和文档，不改变三组公式、训练预算、4份原science字节、24/9/至多12矩阵。最终修复SHA上两端重跑当前R7实现套件和22项参考，再加相应绑定正/负例；不重跑旧166项、不训练真实源模型。

复审通过后，下一阶段是SOURCE_PREP，而不是直接24条目标流。完成真实源资产/分组/源模型暴露核对并另行绑定资源；源训练后冻结六个FULL/STATIC产物和校准信息，再决定TARGET_SCREEN。审阅PASS不是执行授权。

## 6. 来源

- [公开审阅索引](https://github.com/DLwbm123/DPA-CTTA/blob/7f129b484a5ab2b0af3efdf2ccea0eee1228969c/docs/review/r7/REVIEW_INDEX.md)
- [源码：共享Host](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_shared/host.py)
- [源码：源准备与推断加载](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_shared/preparation.py)
- [源码：分割包装器](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_shared/network.py)
- [源码：源训练](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_shared/source.py)
- [源码：A](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_a_psf/__init__.py) / [B](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_b_rca/__init__.py) / [C](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/src/dpa_ctta/r7_c_rbe/__init__.py)
- [完整网络测试](https://github.com/DLwbm123/DPA-CTTA/blob/82aca2080bb80c019b0d39b4c95d21a6ba2cff3d/tests/r7/test_full_network.py)
- [两端结果汇总](https://github.com/DLwbm123/DPA-CTTA/blob/7f129b484a5ab2b0af3efdf2ccea0eee1228969c/docs/review/r7/CPU_RESULTS.json)

实际源码/日志来源及本地探针在本包中分别标注，不把私有或未执行部分伪装成已验证。
