# R6 设计交付包

本包是实验设计及阶段I任务规范，不是完成的CTTA实现或执行授权。

- R6_EXPERIMENT_PLAN.md：研究依据、公式、控制、12→8→20矩阵、gate与预算。
- R6_STAGE_I_CODEX_PROMPT.md：可独立交给Codex的实现任务。只到CPU/metadata验收。
- R6_SCIENCE_PROPOSAL.json：机器可读提案；字节SHA256为 2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff。
- R6_MATH_REFERENCE.py：无模型/optimizer的数学CPU参考。
- R6_MATH_CHECK.log/json：当前环境真实的8项数学检查，最终全部通过。
- math_history：首次失败与源文件。原因是torch.where的Python标量默认中间精度；改为直接float64赋值，未放宽均值断言。

数学检查使用Python3.13.5/Torch2.10.0+cpu，CUDA未初始化，真实图像/GT/checkpoint读取0。它不匹配用户完整软件环境，也不代表生产host、固定完整ResUNet、GraTa、GPU或任何真实数据效果已验收。

不得把当前数学成绩填写成R6实现通过。原R5保持NO_ADVANCE且B不运行。
