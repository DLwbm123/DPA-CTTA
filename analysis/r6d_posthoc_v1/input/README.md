# R6-D 交付包

先阅读 R6D_EXPERIMENT_PLAN.md，将 R6D_CODEX_PROMPT.md 连同整个包交给执行方。

本包仅提出使用现有 R6-A 标量账本的只读事后诊断。当前没有运行私有账本分析，没有创建或授权新 GPU 实验。R6-A 原NO_ADVANCE不变。

文件：
- R6D_EXPERIMENT_PLAN.md：研究问题、精确分解、固定事后分组、证据边界与条件性未来提案。
- R6D_CODEX_PROMPT.md：可直接交给Codex的实现及只读分析指令。
- R6D_ANALYSIS_SPEC.json：同一方案的机器可读规格（不是原R6 science，不替换它）。
- MANIFEST.json：前三个文件及README的原始字节SHA256/长度。

不依赖任何未交付的 R6-D math reference、math_history 或附加 science 文件。无需恢复旧 sandbox 文件才能理解本方案。原R6源码/结果绑定在计划与规格内完整列出。

原公式数值验证、实现回归及私有账本重放属于不同任务。本包的完整性核对不能称为通过了模型或分析实现测试。
