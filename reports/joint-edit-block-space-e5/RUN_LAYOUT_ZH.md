# 运行布局

本目录audit.py复用此前数学资格的formula_check.py；仓库用相对符号链接指向固定资格源码，不重复维护求解器。NAS执行包将相同helper置于audit.py旁，执行source见CPU_CHECK与审计。此链接为交付布局补充，不改变已运行算法或结果。

需既有NumPy、SciPy、Pillow环境，无torch或CUDA。独立新输出目录必须先有private/与public/，输入为已完成退役E3根；使用CPU-only环境运行audit.py，传入源根和新输出根两个位置参数。check参数只做合成核查。真实输入/标签/位图及逐块结果受隐私保护，不随公开仓库发布。
