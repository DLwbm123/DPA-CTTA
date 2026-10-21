# R95：精简证据准入与确认合同准备

日期：2026-10-21。基准版本：8693e6e0b41b6ecb961dc816a2e6ae2c5a656046；既有R94终态b38d007。

**文档准备 COMPLETE；证据准入 NO_NEW_ADMISSIBLE_EVIDENCE；R96/R97/R98 均不准入运行。** 本次按用户明确授权完成精简R95，纠正NATIVE定义/预算，准备确认合同与协议草案。没有新cohort、身份或源成员材料，因此沿用R94的证据状态，不重复其统计或数据审计，不发起SSH，不读取私有数据，不启动CPU评分或GPU实验。

## 1. 必须纠正的基线定义

NATIVE 是冻结其**方法定义**的原适应基线，仍执行逐内容反向与Adam更新；不是冻结参数的模型。SOURCE_ONLY、P0、Q0才是不进行跨图参数学习的静态参照。既有代码的NATIVE分支调用原损失和原step，迁移运行亦计入physical_updates/训练反向。

证据：[NATIVE损失与keyed_step](../../src/dpa_ctta/r79_mechanism_control/method.py)、[迁移Host路由](../../src/dpa_ctta/r85_migration/method.py)、[旧迁移工作负载](../r85-r86-loss-aggregation/migration/PROTOCOL_AMENDMENT.md)。本轮只读源码，没有导入模型或运行代码验证。

## 2. 本次收到什么、没有收到什么

新增材料仅为Pro的条件式路线建议及用户的文档准备授权，未附新的目标内容清单、患者/左右眼/重复拍摄关联表、source/预训练成员清单、使用许可或新概率资产。路线建议不是这些条件已满足的证据。

|准入条件|当前状态|结果|
|---|---|---|
|未参与旧方法选择的授权目标内容|未提供，沿用R87 NOT_AVAILABLE|不进入独立确认|
|患者/眼别/重复拍摄关联|UNKNOWN|不能宣称患者独立|
|source/预训练成员或重叠记录|UNKNOWN|不能宣称已排除源重叠|
|新内容用于推理和离线评分的权限|未提交具体材料，NOT_ESTABLISHED|不扩展读取权限|
|新流顺序/独立样本规模|未确定|不冻结实际全流检验|
|新机制可辨识性证据|NOT_ESTABLISHED；R97只是假设|不开展selector/新采集|
|具体运行预算/空间/截止/guard|NOT_ADMITTED|不启动GPU|

没有全扫旧dataset、重新搜索旧1951内容、重算R94或新bootstrap。不证明所有外部位置都不存在新数据，只记录本控制会话没有收到合格新材料。

## 3. 交付物与状态

- [数据来源清单](DATA_PROVENANCE_CHECKLIST.md)：分别核新内容、患者关联、源重叠和权限。
- [基线合同](BASELINE_CONTRACT.md)：定义静态与动态臂，明确完整状态及输出时点。
- [确认协议草案](CONFIRMATION_PROTOCOL_DRAFT.md)：只有材料到位、未决项全部解决后才可冻结，不是可执行协议。
- [门槛草案](GATE_SPECIFICATION.md)：效应要求、局部保护、结构非劣和多重比较待审；不改旧门。
- [动作价值合同](ACTION_VALUE_CONTRACT.md)：只读动作示例、净收益、非信息性控制和误伤要求，尚未准入。
- [资源计划](RESOURCE_PLAN.md)：纠正NATIVE动态费用，本次仅本地小文档预留。
- [机器可读决定](DECISION.json)：明确不可自动GPU推进。

这次准备的完成标准是合同边界与阻塞明确，不是Dice提升，也不是R96获准运行。

## 4. 后继决定

当前保留VV为既有开发基线，维持科学终态。R90 M1/M2、固定Qt替换、R93边界隔离保持关闭；R94事后内容区间不改变其结论。

只有真实新确认材料到位后，才能补齐并冻结R96；新cohort不能同时承担R97方法选择和R98独立测试。只有新的在线机制信息和可证伪依据到位时，才进入R97协议审查，不直接采集概率或运行selector。两者皆无则NO_NEW_ADMISSIBLE_EVIDENCE，不重复审计或制造GPU任务。

旧campaign及R94时钟不重置，旧747updates/6300F不使用；用户本次授权仅文档准备。完整历史结果见[R90–R94综合报告](../r90-r94-synthesis/REPORT.md)。
