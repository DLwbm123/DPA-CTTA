# Codex续办：修复R7共享推断身份绑定，保持三组算法不变

本次只修复外部审阅R7-F1。以`R7_EXTERNAL_REVIEW.md`与证据脚本为依据。当前状态为R7_REVIEW_NEEDS_FIX，并非真实实验已获准。

## 固定对象

- 被审implementation：`82aca2080bb80c019b0d39b4c95d21a6ba2cff3d`。
- 被审publication：`7f129b484a5ab2b0af3efdf2ccea0eee1228969c`。
- 建议独立分支：`review/r7-inference-context-fix-v1`。
- 原四份science摘要：
  - COMMON_PROTOCOL.json: a18f51317b8547067f8383ebd1125eca1697b24ce6245ca5e2733e2daa28f42d
  - A_PSF.json: 5e8f62ab8a39806ef5aef2dbb5b0206255d9e49c33fa3bcd68c82e78e88410bf
  - B_RCA.json: 9895dba31443881e50bb87d4df94f83598c0069931b35d6d000e6c2b1e61fc61
  - C_RBE.json: 4151c86ebec9073bca570e696c45bbecc4f10e0505698989a0be41cbe2032e86

保留用户工作区和所有既有报告/日志。不要把原INPUT中的DESIGN状态改成当前实现状态。

## 为什么需要修复

Method.digest只包括新模块；OnlineHost state packet与fresh inference_from_tensors都没有把实际源segmenter/观察投影纳入预期身份。小模型反例中仅改变seg_head bias，方法摘要不变，fresh loader与state restore都接受，下一图logit差约1。

这不能用“SOURCE_BINDING以后会填写”代替：现在就实现绑定机制，Stage I用随机模型验证。真实源manifest/checkpoint值继续PENDING，不读取真实资产。

## 修改范围

1. 在r7_shared新增或扩展版本化inference context。至少定义：
   - 原分割网络的有效冻结参数/必要buffer规范内容摘要；
   - 实际固定观察投影tensor摘要；
   - 预处理、BN统计政策、FiLM接口/站点与源码或规范标识；
   - 现有method摘要，保持包含基、scaler、训练权重与校准量；
   - group、FULL/STATIC、deployment ablation；
   - 来源包/训练规格关联。真实checkpoint文件摘要与effective-state摘要分开，不混用。

2. 规范tensor摘要需明确名称、dtype、shape和连续字节次序。不要依赖pickle时间戳/地址，不用对象id作跨实例身份。CPU/GPU上的同一tensor内容应定义一致的语义身份；不要因此改变原模型精度或科学数值。

3. prepare_tensors保存与训练所用segmenter相绑定的产物元数据；inference_from_tensors依据可信准备包核对当前segmenter。不能根据当前传入的segmenter同时生成expected与actual后自比较。

4. OnlineHost的state save/load纳入完整context；C0也绑定主干/预处理环境，不以None充当完整身份。不兼容检查应在恢复状态或第一次模型前向之前失败。

5. 兼容性正例：两个不同Python对象、完全相同网络/方法/投影字节的副本可以恢复并保持下一图输出。负例：主干头/encoder任一权重不同，投影不同，BN/介入政策标识不一致，组/static/消融/基/scaler/calibration不一致，应拒绝。

6. 当前旧格式包从未真实部署，可清楚拒绝或标注不支持，不要静默给旧不完整包补签新的可信context。

7. 不引入每图全模型SHA扫描。可在构建/保存/加载边界计算并缓存，并保留合理的轻量冻结检查。不要因此新增网络前向、在线反向、Adam或VJP。

## 必须保留

A过滤/完整协方差/子空间算法；B五步ISTA/kappa/步长；C三步IRLS/variance校准；所有损失、超参、四份science原字节；独立FULL/STATIC同预算训练；query与状态隔离；旧C与R1–R6结果；24/9/至多12目标矩阵。

不得重写算法以提高指标，不增加模块，不调整gate。

## 回归要求

- 先保留现版本fresh load及state restore漏拒绝的复现日志；可将附件预期拒绝测试移植到现有fixtures，不能把附件TinyBackbone叫完整ResUNet。
- 三组与C0覆盖合法跨实例恢复、不兼容主干拒绝、错误观察投影拒绝、旧不完整包拒绝。
- 真实加载检查必须被测试到，不只是比较两个辅助hash函数。
- 使用随机完整ResUNet做至少一次context来源/兼容性的窄检查；原完整网络梯度与C保持性仍由当前套件覆盖。
- 在最终candidate SHA上两端运行现有47项实现测试（数量按实际新增变化）及22项数学参考；保持原规范数值比较容差，记录实际模型/小矩阵/摘要CPU成本。
- 现有本地NumPy warning原样保留；若进一步诊断，明确观察范围，不直接消音后称无warning；无需为此改原数学公式。
- 不重跑旧166项，不运行1000步真实source，不伪造服务器结果或修复前测试等于修复后测试。

## 交付

新implementation SHA、窄patch、R7-F1解决说明、完整context schema及来源映射、四份science未改证据、实际回归与失败日志、更新REVIEW_INDEX和DELIVERY。报告原件SHA、实施SHA、训练资产SHA、推断context SHA各自含义。

本次只完成修复与复审交付。现有CPU tensor backend不能被写为GPU-ready；设备迁移、真实源IO/finite runner在后续SOURCE_PREP准备中另行明确。不得仅改enabled后启动。

若修复和回归通过，停止在：
```
R7_IMPLEMENTATION_READY_FOR_REVIEW
external_review=AWAITING_REREVIEW
source_binding_status=PENDING
real_source_training_started=false
real_target_execution_started=false
SOURCE_PREP=NOT_RUN
TARGET_SCREEN=NOT_RUN
TARGET_MECHANISM=NOT_RUN
TARGET_EXTENSION=NOT_RUN
```

不得自行签发PASS；不得查询GPU、读取真实RGB/mask/checkpoint，或启动任何真实实验。允许完成本修复必要的程序化CPU验证，并如实计数。
