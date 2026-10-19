# E2-A：零更新诊断完成，初评分接口失败后独立CPU重建

|静态输出|平均Dice|OC Dice|
|---|---:|---:|
|源模型|92.15716%|87.28154%|
|冻结反翻教师本身|50.03033%|40.74580%|
|固定概率等权融合|67.41532%|51.96243%|

主融合−源OC−35.31911pp，前景/边界净纠错均负，融合新增空杯读出。教师和融合worth_followup=false；无参数更新/反向传播、RDC/ENGRAM关闭，不称持续收益。

[完整分析](REPORT_ZH.md) · [执行审计](EXECUTION_AUDIT.md) · [原冻结协议](PROTOCOL_ZH.md) · [CPU接口修复协议](SCORING_RECONSTRUCTION_ZH.md)

32唯一输入/96预测、全encoder重演PASS。初score exit1与原ledger保留；修复scorer exit0、原与修复controller均退休，预测未重生成，所有成本合计。公开聚合不含私有数据。
