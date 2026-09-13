# R1 阶段 II 结果交付

状态：**R1_EXPERIMENT_COMPLETE**。24/24 轨迹、3/3 GPU smoke、独立 CPU 汇总完成。所有计算退出码为 0；本批停止，不自动追加实验。

- [完整实验报告](EXPERIMENT_REPORT.md)：主要结果、恢复/PCA 匹配比较、顺序、尾部、成本及限制。
- [主要终点表](PRIMARY.csv)、[四序 × 四子集 OD/OC/Macro](ORDER_SUBSET.csv)、[逐域分数](DOMAIN_SCORES.csv)。
- [全部八类配对分布](PAIRED_DISTRIBUTIONS.csv)：包含 SHUFFLED−C、OD/OC/Macro、尾部和共同有效 ASSD。
- [原运行器完整聚合](public_aggregate.json)、[原运行器自动报告](RUNTIME_REPORT.md)。
- [运行与预算审计](EXECUTION_AUDIT.json)、[恢复/PCA 机制统计](MECHANISM_AND_COST.json)。
- [最终 CPU 补表日志](CPU_CLOSEOUT_LOG.txt)、[补表初次失败日志](CPU_REPORT_INITIAL_FAILURE_LOG.txt)；失败原因及修复说明见报告，正式实验未失败或重跑。
- [结果图 PNG](order_gains.png)、[SVG](order_gains.svg)。
- [CPU 标量汇总脚本](../../../scripts/report_r1_results.py)、[报告/绘图脚本](../../../scripts/render_r1_report.py)。前者在私有环境读已有 scalar JSONL，后者仅需本公开目录和 matplotlib；没有模型运行。

执行提交：[54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b](https://github.com/DLwbm123/DPA-CTTA/commit/54c4ca4fc2672d3da37e1f70d3c8ab7ac6bad31b)。本目录所在提交是结果发布版本，未改变运行绑定。

复现补表时，`PYTHONPATH` 指向固定执行 checkout 的 src 及原有依赖；`RESULT_DIR` 是完成的私有运行目录，`CONTROL_DIR` 含其既有私有 assets/launcher receipt，`REPORT_DIR` 是全新补表输出目录；CUDA 关闭。脚本保留原结果目录和运行 SHA，不重新获取任何资产或运行实验。

science SHA256：`e23fb6de3e55f704ec2036d82777b29a78643ebc7e1ae1e89328f22f56de51e2`。
registration digest：`8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`。

开发数据已暴露，四序共享内容；结果不构成独立患者重复、盲测或受控速度比较。公开交付不包含身份、路径、权重、原始图像/mask 或逐图私有日志。
