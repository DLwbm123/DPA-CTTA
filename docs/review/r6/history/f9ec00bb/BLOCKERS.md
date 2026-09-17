# R6 阶段 I 原始材料缺失

已收到原始 prompt 和实验计划。两份文件均列出下列配套输入，但附件路径及 Downloads 中未找到：

- R6_SCIENCE_PROPOSAL.json。期望原始字节 SHA256 为 `2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`；此值来自 prompt，不是本次对缺失文件计算出的摘要。
- R6_MATH_REFERENCE.py。
- math_history 的首失败、修正及最终日志。

因此未生成冒充原始 JSON 的替代文件，未声称核对了 prompt 与 JSON 的一致性，未声称验证了外部八项数学参考或恢复了其历史。`configs/r6_science_v1.json` 有意缺席；执行授权首先检查 disabled，开启时仍要求真实原始 JSON 的字节摘要符合注册值。

实现以自包含 prompt 与计划的共同定义为依据；新的独立数学、完整网络、故障及账本测试与缺失的外部参考区分。元数据 dry-run 使用既有真实 registration，输出 science_bytes_verified=false；它验证登记流与 12/8/20 矩阵，不构成完整 science 冻结。

需要提供上述原始材料或其已存在目录，再进行一致性核对和缺失证据归档。当前不能发布 R6_IMPLEMENTATION_READY_FOR_REVIEW。所有 GPU、实际 checkpoint/目标数据计算、R6-A/B 均 NOT_RUN。旧授权没有沿用。
