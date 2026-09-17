# R6 保持性说明

从旧实现 d875f20c11cc7e617c9f39dba04ed46381aba450 到本次候选 c94fff7c05cec38d541c441b62a9381ee46ba076，src、现有 scripts/tests、execution defaults 均无变更；运行相关目录唯一新增为 configs/r6_science_v1.json。原提案、数学参考、历史日志、原MANIFEST均原字节存档，新增核对脚本独立于生产导入链。

从科学基线 e271098e… 起，本轮变更仍只涉及新增R6路径。未修改main、旧C、旧P2/R1–R5科学配置、结果、源模型、登记、固定依赖、IO/NFS容错或进程监督机制。生产指纹 a6ef161066d3560dc2dd540cb507bff2e4a7cadf94ab711545877c525dee2f2e 与d875f20相同。旧R5保持NO_ADVANCE、B关闭。

此前缺原件未达到READY的历史状态保存在history/f9ec00bb/，旧final-local/server CPU日志仍对应d875f20，新日志单列originals-redelivery。设计包中的首次失败和最终数学8项历史日志不覆盖。本次真实数学重跑不调用原参考会写历史文件的main。

本次CPU回归临时文件：本地专用tmp-local；服务器专用NAS jobs/r6_originals_redelivery/tmp。此前服务器/tmp偏差与被取消旧候选的未知成本继续保留在历史报告。没有为本次核验终止、重启或清理历史进程/目录。

两个完整CPU runner的主进程及当次存活子进程经ps核对为中性入口；不将这一个快照声称为所有短暂子进程的全程取证。元数据辅助命令曾直接在python -c中带模块名，未完全遵守中性参数约定；此为本轮辅助命令可见性偏差，没有GPU或后台执行，也未以重跑来掩盖。后续入口统一用环境RUN_FILE传递路径。
