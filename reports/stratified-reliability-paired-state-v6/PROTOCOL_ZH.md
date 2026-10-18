# V6：预算内的严格确定性、同进程状态对照

2026-10-19。用户持续实验授权下的独立新轮。V5已按冻结预算资格停止并公开交付c4055509fff2a54a93a28ae05a833f9c6ea5f351；前2arrival实测均时.6651904583秒、完整投影1287.60>1180，未评分。V5不续跑、不更改历史记录；V4仅短流重复性工程PASS、V3不可评价、V2科学负证据都保留。

**唯一设计缩减**：从V5移除RG、NOANCHOR、ENT_NORM的RESET三诊断臂，每臂80arrival；它们RETAIN保留。STRAT/RANDOM各RESET、RETAIN、FRESH三状态全部保留；真实CW1/CW2各RESET/RETAIN，及SOURCE_EVAL、CURRENT_BN、旧D/R/平移R五静态参考全部保留。主比较、数据、两顺序、五块、步幅、损失、视图、统计与安全门槛不变。原生CW仍是current-image BN，不能与源BN差别混为新方法贡献。

版本固定为V6：core9个方法/状态组合×两顺序×40=720科学arrival，native320，共1040/26轨迹；PRE/POST2080+400静态=2480读出，31绝对聚合行，每行80内容条件/160通道单元。1280次科学规范匹配更新+160次原始RG更新。实现按固定V5/V6版本核对策略与矩阵，旧V5配置仍识别为V5；禁止只改计数或任意删除控制。V5服务器源码与失败结果不可改。

同进程严格确定性：torch.use_deterministic_algorithms(True,warn_only=False)、cudnn.deterministic=True、禁benchmark/TF32、CUBLAS :4096:8；不支持算子必须停止，无warn_only或回退。STRAT/RANDOM的RESET每图重置仿射/Adam，RETAIN两者保留，FRESH保留仿射并每图重建Adam，同图两步共享。82个BN affine张量/19136参数，实际L2=.01±5e−7、方向cos≥.999999不变。running stats、卷积和源锚不可更新；没有RDC/ENGRAM、新源训练、标签或未来图像参与适应。

额外40arrival准入保持：首个STRAT RESET正序40流在同进程新host完整重演，80对logits及末model.state_dict必须精确一致。额外80更新计费，不计科学矩阵；仅此RESET完整流的资格，不认证其它长流/跨进程、Adam/RNG全状态，也不恢复V3。FULL_REPEATABILITY必须PASS。

预算仍GPU进程≤1200秒、CPU评分≤900秒、输出≤2GiB。头2arrival保守投影=已用时+(720+额外40−2)*实测均时*1.35+native320秒+worker内收尾60秒，须≤1180；60秒不替代独立CPU评分预算。不降低系数或放宽阈值；此前均时推算约1072秒只是规划，不保证现场准入。若再次资格停止，保留失败，不继续削控制或重复资格。历史累计GPU1074.9415838718秒、已计CPU540.5136370659秒保留，加本轮CPU/进程，旧未计时错误/只读诊断仍披露。

唯一科学主比较仍STRAT_FRESH−STRAT_RETAIN:PRE，至少+.5pp且八内容组描述性bootstrap95%下界>0。推进还需：首块及全流PRE对SOURCE_EVAL区间下界>0；全流PRE最差≥−1pp；返回PRE对源均值≥−.2pp；FRESH PRE对同配额RANDOM下界>0；FRESH POST均值超过源/RG/CW1/CW2；规范匹配无zero、无新增空/嵌套，ASSD/HD95/vCDR与覆盖不劣于源。全部状态差中差、首块/返回、PREPOST、强控制前沿、通道/顺序、实际功能变化及前景结构都报告，不用POST替代PRE。

bootstrap2000次按8内容组绑定重复条件、两顺序及通道。前景corrected/spoiled参照SOURCE_EVAL（与V2 CURRENT_BN参照区分）；门控审计在全图/真实3px边界，对应第二步更新前实际预测。CPU scorer只在正常worker退役后读取8评分标签，不读取历史预测/评分作对照；历史parity明确0，不改V3原标准。完整继承机制定义见[V5原协议](../stratified-reliability-paired-state-v5/PROTOCOL_ZH.md)。

先CPU回归（包含V5/V6固定策略兼容、只删三指定臂且主/native控制全保留、混版计数拒绝）、NAS实际挂载/空间/写读、显存≥8192MiB与中性入口准入，再只启动一次。完成或停止后中文报告、匿名结果和成本receipts公开GitHub，私有数据/逐例信息不发布。

只有8个反复开发暴露内容，患者与预训练成员未知；只有首块是本流首次暴露，没有独立患者/临床/TMI创新结论。清Adam/分层/KL本身都不新；科学门槛仍失败则放弃此机制的正收益叙事，不盲调步幅/损失/Adam，不扩大数据seed长流模型权限。
