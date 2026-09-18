# 后续任务：SOURCE_PREP 绑定与授权材料闭合（不执行）

外部执行层复审 PASS，绑定实现 f719c703087b38c07bdfbe7ce9dcfa62d88a12d9、证据发布 ecde8ba63940eb5c660f4de01bd7a9300b18c15b；SP1/SP2/SP3 CLOSED。仅归档新复审报告和 REVIEW_RECORD，保留旧 NEEDS_FIX 和所有开发失败；不重写历史审阅。

本任务只要求完成现有源登记、有限 CPU 资源和新的 SOURCE_PREP 授权材料，不启动训练、不授予任何新权限。不要改算法、science 字节、FULL/STATIC 预算、fit/cal/val 角色、24/9/至多12 矩阵或历史结果；不要新增 GPU 路线，也不要重跑旧166项。

从既有合法已知登记中核对 source/checkpoint/manifest/split/target/config 的路径和摘要、保持111/23/25。只有在现有明确权限允许时才对已登记文件做只读原始字节 hash；无权限或未知身份则列出阻塞，不猜私有目录，不解码真实 RGB/mask，不反序列化真实 checkpoint。

患者/眼别与预训练暴露继续按证据 UNKNOWN。分别说明内容分组限制、启动前文件绑定、运行后 effective tensor/产物身份；不得为通过检查而伪造 identity，也不得要求训练产物 hash 提前存在。风险接受不能自行替用户完成。

提出绑定实际 CPU/线程、有限 wall/output/memory/storage 的配置及默认禁用 receipt。拟议72小时和512MiB不是授权或ETA。新 receipt 需覆盖实际 code SHA、外部代码审阅、源及资源审查、独立元数据摘要、唯一 SOURCE_PREP scope 与用户授权；不得沿用旧授权或把本复审报告当作可执行 receipt。

只归档或编制文档时不得声称重跑实现测试；发生生产代码修改则另行记录新SHA和适当回归，再复审。

交付字段级绑定矩阵、未解决风险、有限 CPU 资源提案、默认禁用授权模板和材料摘要后停止。source_binding_status据证据填写；所有执行状态维持NOT_RUN、execution_authorized=false，等待用户针对准确材料的新执行决定。
