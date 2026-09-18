# R7-F1 外部复审报告

## 结论

**R7_REVIEW_PASS_STAGE_I**。R7-F1关闭；本次限定复审未发现剩余Stage I阻断项。A/B/C原有算法审阅结合本次共享身份修复证据，可以结束Stage I。

- 实现：`8f179ce0ca848791793a17dab0606f0bde2b3e07`
- 证据发布：`c53ad20a6c5e5e0caefe019441f10e6f612960ec`
- 通过scope：`STAGE_I_IMPLEMENTATION`
- source绑定：`PENDING`
- 真实源训练、目标实验、GPU使用授权：本报告均不授予。

这不是SOURCE_PREP真实执行验收，不是GPU-ready，也不代表新方法有效。当前CPU tensor后端与后续执行层必须区分。

## 一、审阅依据与范围

读取公开修复说明、context schema、固定context/host/network/preparation源码、新增测试、保持性报告、CPU摘要及两端最终日志相关段落。比较实现和证据提交的根树：src/scripts/tests/configs完全相同，docs改变。八个本地待测源码副本的Git blob均与GitHub返回值一致。三组算法及numerics blob与上次审阅相同；原四份science从此前交付包恢复并核对SHA256。

本环境未登录用户服务器，未访问其本地Downloads ZIP或私有账本；未重跑提交方59项完整实现套件、22项数学参考或完整ResUNet。提交方的完整网络结果属于公开交付证据。

独立探针实际使用原context/host/network/A/B/C/numerics模块；preparation中的new_method、prepared_artifact、inference_from_tensors、prepare_tensors使用原AST函数体。只有旧图像预处理换成明确标注的恒等fixture。小主干是4个1×1卷积、2个BN及一个非持久buffer的程序化网络，不是ResUNet。源准备packaging检查明确模拟昂贵训练阶段，没有运行1000/256步骤。

容器直接HTTPS下载曾因DNS失败一次，无源文件因此下载成功；随后使用连接器文本及旧审阅包恢复副本，逐个核对Git blob。这是传输失败，不是实现测试失败。见TRANSPORT_NOTE。

## 二、R7-F1关闭依据

原问题：方法摘要不绑定实际分割主干，相同新模块可以在错误主干上fresh load或restore。

新实现：

1. `R7_INFERENCE_CONTEXT_V1`连接有效主干参数与buffer、固定投影、模块布局/BN设置、预处理与FiLM政策、method/basis/scaler/calibration、FULL/STATIC与消融、源端和原科学规格来源。
2. `prepare_tensors`先保存训练环境；每份`R7_PREPARED_TENSORS_V2`发布时对照该环境，并保存完整context。
3. fresh loader接受独立可信的expected_context，仅计算candidate实际身份；不按candidate重造expected。
4. `R7_ONLINE_STATE_V2`保存完整context；load在状态赋值和前向前核对。C0也绑定环境。
5. 消融context只能由可信FULL产物显式派生；不会自动覆盖expected的ablation。
6. 普通逐图冻结检查只读版本/对象/布局元数据；边界重算内容摘要，不新增每图完整主干hash。

## 三、独立探针结果

10/10 unittest方法通过，0失败、0错误、0跳过，实际耗时1.126秒。不可与提交方59项相加为一个完整套件。

| 检查 | 结果 |
| --- | --- |
| A/B/C的FULL、STATIC及C0，7个同字节跨实例案例 | 合法恢复；下一图输出与状态逐值一致 |
| A/B/C/C0 × head/encoder/projection/nonpersistent buffer，16个环境案例 | fresh和restore共32个拒绝检查全部通过；拒绝前新增前向0 |
| 政策、真实BN参数、train/eval与旧格式 | 13个拒绝检查通过 |
| basis/scaler/calibration/static/group | fresh与restore均拒绝 |
| 三个部署消融 | 只接受从可信FULL明确派生的context；拒绝FULL/消融错配 |
| tensor canonicalization | 与独立length-framed字节参考一致；非连续同值相同；名称/shape/dtype/值变化不同 |
| 逐图成本与冻结边界 | 两步更新不触发tensor内容hash；普通原位修改前向前拒绝；.data变化在save/load边界拒绝 |
| expected摘要、science、环境及BOUND缺失 | 拒绝不兼容或不完整身份 |
| 保存环境及拷贝独立性 | 环境漂移拒绝；训练产物不跟随原模块后续修改 |
| 六份源产物打包流程 | 原prepare函数体在明确mock的训练依赖下产出六个绑定包；没有真实训练 |

实测成本：49次小主干前向；网络反向、VJP、Adam、AdamW全部0；真实资产与GPU调用0。环境Python3.13.5、Torch2.10.0+cpu、NumPy2.3.5。完整记录见INDEPENDENT_PROBE_RESULTS和日志。

内容hash调用376次，累计处理73647733字节，包含重复扫描，不是模型大小。context时间包含hash时间，不相加。本次没有为未执行的完整GPU测试补填结果。

## 四、提交方证据核对

两端最终实现日志相关段落分别为59项、0失败/错误/跳过；CPU_RESULTS中的数学检查分别22/22。所报总前向495=479个R7计数+16个旧C；完整随机ResUNet85次，小fixture410次。反向36、VJP2、Adam24、AdamW6。没有把这些程序化调用写成真实源训练，也没有将mock的1000/256字段冒充实际训练。

本地NumPy警告原样保留，公开说明其来源尚未诊断；本次不声称警告无害，也没有为了消除警告而修改公式。F1闭合依赖独立边界检查和固定代码，不依赖对警告作未经证实的解释。

## 五、保留的边界，不作为追加Stage I修复任务

- context摘要是兼容性/完整性标识，不是签名；调用方提供的expected必须有可信来源。
- 源checkpoint文件摘要与有效内存状态摘要不同。当前随机模型状态不能填成真实source文件摘要。
- `.data`/raw storage的刻意绕过不在逐访次轻量版本检查承诺内；边界hash可以检测。不是针对任意同进程恶意代码的安全审计。
- 使用CPU/GPU同值的规范摘要定义，不等于GPU推断、设备混合或源端训练已验收。
- 本轮不扩展到新的抗遗忘、分割有效性、泛化或概率校准结论。

## 六、下一步

结束F1修复，三组公式、预算、24/9/至多12目标矩阵保持。下一项工作是SOURCE_PREP准备：确认合法source登记与fit/cal/val分组、源checkpoint来源及暴露历史；准备真实IO、设备放置和有限训练runner；做与新增执行层对应的程序化验收，然后独立绑定执行资源与授权。

不要只把原`real_entry`里的禁止改成允许。源模型和新模块的dtype/device、CPU光度增强、float64小矩阵、求导跨设备路径、计数和终止处理需要在新增执行层明确。

SOURCE_PREP训练完成后应冻结六份FULL/STATIC产物及calibration/basis/scaler/context，审阅源端结果，再决定目标SCREEN。不能在这次Stage I PASS后跳过源训练直接启动24条目标轨迹。

本报告不修改远端配置，不签发执行receipt。真实训练所有阶段仍NOT_RUN。完整源码和公开证据入口见SOURCE_REFERENCES.json。
