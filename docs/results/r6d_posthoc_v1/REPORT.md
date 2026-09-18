# R6-D delivery: preflight stopped

**POST_HOC_EXPLORATORY — R6D_INCOMPLETE.** 原 R6-A 保持 `R6A_COMPLETE_NO_ADVANCE`，R6-B 保持 `NOT_RUN`。本轮没有新模型计算。

独立分析工具已实现，最终本地和服务器均通过 **25/25** 项程序化 CPU 测试。真实固定分析在源输入预检退出 1，未进入快照、逐行校验或机制诊断。不能将工具测试通过表述为 R6-D 实际分析完成。

## 具体阻塞

原运行根的 `public_aggregate.json` 是符号链接，目标为 `current/public_aggregate.json`。工具按普通独立源文件要求拒绝该入口。其余白名单条目的类型检查未发现相同问题；新工作目录为空，快照 0 文件。该链接属于原结果发布布局，**不是旧账本损坏或旧实验失效的证据**。

本轮没有跟随该链接生成快照，也没有改动原链接、生产代码、science、gate 或历史结果。完整 before/after SHA256 与原指针逐字节核对尚未执行，明确记为 NOT_CAPTURED / NOT_PERFORMED，不能声称它们通过。早期发现阶段读取过原结果指针和登记 JSON；固定分析尝试本身在文件元数据预检停止。

任务 prompt 规定：“已有账本身份、完整性或只读保护失败属于 R6D_INCOMPLETE……不自动重跑。”因此保留本次失败，不自动改白名单并再次运行。后续若继续，应先依据原 current_result 的版本目录选择其普通聚合文件，单独审计原符号链接元数据；本轮未实施这一读取适配。

## 已完成与未完成

- 包内 MANIFEST 的全部 4 项原始字节、长度、SHA256 已核对。原文件原样归档，原规格 `PROPOSED_NOT_EXECUTED` 未被改写。
- 隔离工具已实现精确分解、域/通道/序权重、同内容跨流配对、注册 chunk/segment 与固定切换窗口、监督贡献和硬错误质量、C 锚定固定分层及完整空组导出。
- 复用原纯标量函数 AST，未导入生产模块；原模型/写结果入口被排除。原校验容差保留，输出容量校验使用原输出元数据总量。
- 25 项测试覆盖原标量 join、计数/GT/配对拒绝、分解与加权、段贡献、ties/null/稀疏组、快照/指针保护、错误路径、模型/GPU接口拒绝及完整去身份导出。
- 真实 12×1951 的逐行复核、原 gate 重新复算、所有机制分析及完整只读摘要**未执行**。所附七张 CSV 是显式 NOT_AVAILABLE 状态表，包含 0 条分析数据，绝无合成配对或虚构结果。
- 私有账本存在，本次为保护预检失败，因此不能改报 `R6D_PARTIAL_AGGREGATE_ONLY`。既有公开结果只作为历史背景，不冒充新分析。

## 成本与异常

开发阶段 24/24 测试通过；补充完整导出 fixture 后，最终本地 25/25 用时 15.95 秒，峰值 RSS 182796288 bytes；服务器 Python 3.10.6 的 25/25 用时 49.76 秒，峰值 RSS 204800 KiB。无测试失败、错误或跳过。测试中的拒绝样例是预期通过，不是真实 GPU 查询。没有重跑原 166 项模型测试。

首次代码传输脚本存在字符串替换造成的 SyntaxError，发生在远端脚本执行前；修正后部署成功，首次日志保留。之后只尝试了一次真实固定分析，并在普通文件预检退出 1。该退出早于运行遥测初始化，远端 CPU/RSS 与精确耗时未记录，不能补填估算。没有后台重试或轮询。

## 绑定与交付

分析实现 SHA：`ad75672e2931c5dd09ffb19f3249d6b65ec6aa0d`。

原执行 SHA：`c94fff7c05cec38d541c441b62a9381ee46ba076`。原结果发布 SHA：`aa732b42b03d378149028a3770879b72ebc55db3`。分析规格 SHA256：`ffd3c648931e27130c2c275da7ca83f7bac62906b9e9e3d51ec4b3fd62d07368`。文档发布 SHA 与分析实现 SHA 分开。

- [分析工具与调用链说明](../../../analysis/r6d_posthoc_v1/README.md)
- [输入 prompt](../../../analysis/r6d_posthoc_v1/input/R6D_CODEX_PROMPT.md)
- [分析绑定](ANALYSIS_BINDING.json)、[只读核对](READONLY_AUDIT.json)、[字段可用性](FIELD_AVAILABILITY.json)
- [真实测试与失败摘要](TEST_SUMMARY.json)、[本地测试日志](CPU_TEST_LOCAL_FINAL.log)、[服务器测试日志](CPU_TEST_SERVER_FINAL.log)、[去身份预检失败日志](PREFLIGHT_FAILURE.log)
- [支持/反证/不可识别与假设决定](HYPOTHESIS_DECISION.md)、[机器状态](AGGREGATE.json)、[交付清单](DELIVERY.json)

没有提出 R_HALF 或其他新干预。new_model_execution_started=false；forward/backward/Adam/VJP 全部为 0；next_gpu_execution_authorized=false。
