# R7 三组方法实施包

当前仅授权Codex完成Stage I实现和CPU验收。源端训练与目标实验必须在外部审阅后分别授权。

先阅读MASTER_PLAN.md与COMMON_PROTOCOL.md；将整个ZIP交给Codex，入口为prompts/00_MASTER_CODEX_PROMPT.md。已有并行执行能力时，再分派三份组Prompt；否则顺序实现。不要只发一份小Prompt而遗漏共同协议。

目录：
- 三个TRACK_*_METHOD.md：精确算法及迁移边界。
- specs/：共同协议和三组机器可读设计，原样归档。
- prompts/：总任务和三组独立任务。
- references/：独立小矩阵数学参考、原始论文来源。
- checks/：本包实际CPU数学测试日志/JSON，非完整网络结果。
- TARGET_MATRIX.json：目标scope矩阵，仅计划，全部disabled。
- MANIFEST.json：所有条目大小与SHA256（不含自身）；verify_manifest.py可以复算。

不需要之前未交付的math_history。旧R6结束状态不改。包内任何PASS只表示注明范围的本包数学检查，不是R7实现/源训练/目标实验/外部审阅PASS。

新执行数学检查请设置PYTHONDONTWRITEBYTECODE=1，将stdout日志重定向至新工作目录；仅在需要JSON输出时设置R7_MATH_RESULT为新工作文件。不要修改input归档里的历史测试结果。
