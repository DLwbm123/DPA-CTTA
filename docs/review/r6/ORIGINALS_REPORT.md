# R6 原件核对报告

本次补交的原始 MANIFEST 全部11项通过长度与SHA256核验，12个原件均已原样归档。科学提案与数学参考摘要分别等于用户指定值；原件prompt/plan与此前发布副本逐字节一致。详见 [机器核对记录](ORIGINALS_VERIFICATION_CURRENT.json) 和 [逐项对应表](CORRESPONDENCE.md)。

配置 configs/r6_science_v1.json 此前不存在；本次直接新增原提案的5462字节副本，没有覆盖或重排已有配置。其SHA256与原提案相同是因为实际字节比较通过。科学提案的历史status不代表当前实施状态；原件不改status，当前状态独立写DELIVERY。独立绑定摘要见 [SCIENCE_DIGEST.json](SCIENCE_DIGEST.json) 及其SHA256侧文件，不能与原提案摘要混用。

原件中的 R6_MATH_CHECK.log/json 以及 math_history/first_attempt.log/json、first_reference.py 保留原始字节。首次失败原因及最终8项通过仍是设计包的历史记录。补交 ORIGINALS_VERIFICATION.json 是提取记录，本轮没有重新取得其所称旧设计ZIP，不能独立验证该旧ZIP摘要。

本轮仅通过import加载数学参考，执行其ReferenceTests，不执行会覆盖原R6_MATH_CHECK.json的__main__。新的8项运行与240组共同输入、绑定/gate对照另存新日志；全部日志见 [ORIGINALS_ACCEPTANCE.json](ORIGINALS_ACCEPTANCE.json)。原参考和生产SCALE的求值顺序差别明列，不通过改公式/调容差/覆盖原件消除。

加入原始科学配置激活了原先缺文件的字节绑定路径，因此不按“仅文档变动”豁免回归。候选 c94fff7c05cec38d541c441b62a9381ee46ba076 在两端重新运行完整166项CPU套件；后续文档发布提交未重新运行这些测试。旧d875f20的166项日志及f9ec00bb因缺原件BLOCKED记录均保留，不能把先前状态追改成READY。

本轮GPU查询/初始化/运行均为0，真实目标RGB/mask、已登记checkpoint和源数据读取均为0。旧IO回归只使用现场生成的临时fixture。没有启动后台实验或等待任务，没有生成外部review PASS或A/B授权。
