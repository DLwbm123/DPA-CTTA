# 2025–2026机制阅读与主线取舍

新增正文13篇，47,718,029字节；上限50篇，未为凑数量继续下载。五类会议均覆盖；2026条目来自已获取的会议主页/论文PDF，不据此推断尚未公布的NeurIPS2026录用。已有ICLR2027材料仍为投稿快照。

阅读等级：M=关键方法段核查，P=摘要及部分正文定位；均不是代码复现或逐页全文精读。

|论文|会议|证据|可复用机制与决定|
|---|---|---|---|
|[IT3](https://proceedings.mlr.press/v267/durasov25a.html)|ICML2025|M|源监督训练可接收预测/标签的固定点映射；测试时第二次映射用冻结/EMA模型避免共同塌缩。首版改为冻结源纠错算子+小状态适应，不声称原样IT3；真标签只在源训练。|
|[REM](https://proceedings.mlr.press/v267/han25e.html)|ICML2025|P|对信息退化程度施加熵排序并做预测一致性；依赖ViT注意力的遮挡选择。保留为后继对照，不声称任意RGB遮挡等价。|
|[RCP](https://proceedings.mlr.press/v267/hu25i.html)|ICML2025|P|邻域一致性作为置信替代；region主要是特征空间邻域，不直接等同医学空间区域。用于避免只按熵判断效用。|
|[SelfBootstrapping](https://proceedings.mlr.press/v267/niu25a.html)|ICML2025|M|几何保持的退化视图到原图自蒸馏，按置信选像素。作为无监督信号备选；原预测可能错，不与幂等损失盲目并加。|
|[MetaRegularization](https://proceedings.mlr.press/v258/iwata25b.html)|AISTATS2025|P|源任务meta学习任务相关正则强度，原任务为有标签少样本线性回归。借鉴外层学习适应约束，不把其监督设定当无标签部署。|
|[FAF](https://proceedings.mlr.press/v300/taga26a.html)|AISTATS2026|P|参考模型与当前模型的可约减损失差驱动样本选择，参考也适应。其预测任务有监督训练信号，不把RHO loss直接放到无标签医学目标；借鉴学习收益而非低损失的选择目标。|
|[DecisionTTT](https://proceedings.mlr.press/v306/wakayama26a.html)|ICML2026|P|高斯/核模型下更新方向、步数和信噪比的决策解释。作为低维、有限步更新的设计约束，不把理论保证外推到非线性分割。|
|[TTE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/c242f2b7f60d8c685b6481939330e241-Abstract-Conference.html)|ICLR2025|M|自适应参数EMA+dropout增加表征多样性，并防止蒸馏塌缩。因本项目历史状态未证实收益，首版不增加EMA主状态。|
|[IMSE](https://proceedings.iclr.cc/paper_files/paper/2026/hash/54f117f842d47c2007b501bddd04c0e7-Abstract-Conference.html)|ICLR2026|P|固定奇异向量、更新谱值并检索域谱码。保留低维适应思想；CNN谱改造暂不进入首版，避免重新堆叠旧谱/状态路线。|
|[Universe](https://openaccess.thecvf.com/content/CVPR2025/html/Lv_Test-Time_Domain_Generalization_via_Universe_Learning_A_Multi-Graph_Matching_Approach_CVPR_2025_paper.html)|CVPR2025|P|医学多源训练学习形态universe嵌入，测试图匹配。真实多源标签要求与当前单源首轮不同，后继第二任务/多源版本再考虑，不凭形态名字假设正确。|
|[ViT3](https://openaccess.thecvf.com/content/CVPR2026/html/Han_ViT3_Unlocking_Test-Time_Training_in_Vision_CVPR_2026_paper.html)|CVPR2026|M|内更新可微并随外层任务端到端训练。借鉴训练时展开部署更新，首版使用小状态非ViT3架构，不称复现ViT3。|
|[SNAP](https://proceedings.neurips.cc/paper_files/paper/2025/hash/820e42f39773c6cbbd875553db45658f-Abstract-Conference.html)|NeurIPS2025|P|代表性记忆与推理归一化降低更新频率。作为效率对照的思路，不默认减少更新就属于选择收益。|
|[RiskMonitor](https://proceedings.neurips.cc/paper_files/paper/2025/file/746960ad49ddb47248970a0e1404230c-Paper-Conference.pdf)|NeurIPS2025|M|顺序风险监测扩展到变化模型与无标签代理；校准/分布假设不可省略。首版记录实际任务指标与失败，不用置信上界伪造医学安全保证。|

## 已有全文如何接入

|已有论文|主线用途|
|---|---|
|Intervention-Dependent Selection Credit|同count/weight控制、独立状态轨迹、负交互与组合反转区分；是归因依据，不是部署算法|
|LSTA / AR-JEPA|学习更新子空间、展开部署更新后任务收益；源训练合理性已有，不自称原创|
|Identifiability|明确部署证据通道；增加源监督学习并不能保证任意目标移位可识别|
|CP-JEPA|源监督证据到任务修正、病例状态与持久状态分开；不冒用不存在的正交切面|
|IRP / ENGRAM|提醒保护/存储历史必须先证明方向有效；首版不加入域检索|
|SIFT|写入前功能验证思路；原文有答案，改用源训练收益预测器后不再享有有标签验证可靠性|
|DTS-Meta|快慢状态及在线元更新已有；首版不用同一伪标签的support/query冒充真实效用|
|LQN / SEER|病例局部计算、独立语义核验的可能；首版无额外VLM/SAM，减少资源与协议变化|
|RDC / Soft-PSCL / HRAC|关系约束、软目标与解剖比例的边界；不把关系一致/比例配额当语义真值|
|SphereTrust|必须与简单选择或随机匹配比较，不能归功于复杂评分|
|UnfoldCRF / REVIVE|已取得全文，但用户不选其为主线；本轮不追加结构传播/agent开销|

## 本轮决定

完整首版采用源监督纠错、固定点无标签更新、可学习区域支持/低维更新、跨病例收益写入四个相互连接的环节。与DIRECT同容量源监督模型比较是核心；先证明适应而不只是额外源训练有用。暂不把纯记忆、频谱正则、VLM、图匹配、多个熵损失同时堆进去。可根据源开发结果自主更换失败机制，但不依据封存目标结果继续调参。

失败依据详见METHOD_AND_PROTOCOL_ZH.md：A10未胜强B、MEMORY未显独立收益，R103–105闭窗无晋级。不同旧模型/分辨率/内容面板数字不与r01拼接成统一排行榜。
