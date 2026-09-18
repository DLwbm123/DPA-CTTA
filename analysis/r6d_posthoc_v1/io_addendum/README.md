# R6-D 输入适配补充包

入口：R6D_IO_CONTINUATION_PROMPT.md。

本包仅补充已发布R6-D的I/O读取与异常记录规则。旧分析科学规格仍使用仓库已有input/原件；不需要未交付的新math_reference或math_history。

R6D_IO_ADDENDUM.json为机器可读规则，R6D_IO_DECISION.md说明根因与本次核对边界，SOURCE_REFERENCES.json列固定来源。

先核对MANIFEST，再将Prompt作为新的续办任务交给Codex。此前失败不删除。新任务限定为局部修复、标量CPU测试和一次前台只读诊断；不授权GPU、R6-B或R6-E。

该包不包含生产代码修复或已通过的修复测试，也不包含私有账本或真实机制分析。
