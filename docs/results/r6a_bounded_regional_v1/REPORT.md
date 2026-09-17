# R6-A 完成报告

**R6A_COMPLETE_NO_ADVANCE**。12/12 完整轨迹，23412/23412 无标签与评价记录；CPU 标量核验有效，launcher exit=0，3 个 smoke 和 12 个正式子进程全部 exit=0，无失败、未启动任务或重试前缀。机械完整，但科学晋级门槛未通过。R6-B=NOT_RUN，next_execution_authorized=false。

北京时间 **2026-09-17 15:23:48** 完成。启动至 CPU 分析结束 14240.77 秒（约 3 小时 57 分钟）；监督器墙钟 14222.06 秒，累计 worker 活动 25342.34 秒。共享 GPU 5/6/7，每卡一个 worker、每 worker 两线程；按冻结静态分配完成，设备速度差异不证明方法加速。

## 来源与范围

执行 implementation SHA：`c94fff7c05cec38d541c441b62a9381ee46ba076`。Science SHA256：`2c9cb45d4e8c8022f5a61240da36bb05fa51db4bab5588f558e60976f54d79ff`。Run ID：`a70dc79f23db47289532792141b61f57`。其余 registration/stream/fingerprint 绑定保留在 EXECUTION_AUDIT.json。文档发布提交不等于执行提交。

授权范围为 R6-A 的 C/R_BAL/R_SCALE/R_SHUFFLE × orders 0/1/4；用户确认 GPU 5/6/7 与既有 NAS 存储。运行使用外部已审实现与冻结配置，不改 main、历史配置或结果，不训练源模型。此前缺原件的历史状态与后续原件核对/审阅材料继续保留。

## 主结果

主指标：remaining_dev=1695；每图 OD/OC 同权、域内平均、四域等权，再对 order0/order1 等权。每条全部 1951 内容参与更新。回访 order4 单列，不纳入两个主序平均。Dice 为百分制，差值为百分点（pp）。

| 方法 | OD Dice | OC Dice | macro Dice |
| --- | ---: | ---: | ---: |
| C | 86.915188 | 70.191131 | 78.553160 |
| R_BAL | 86.756720 | 70.313591 | 78.535155 |
| R_SCALE | 86.377948 | 69.816358 | 78.097153 |
| R_SHUFFLE | 86.303183 | 69.754927 | 78.029055 |

R_BAL−C = **−0.018004 pp**，未达到 +0.5 pp；R_BAL−SCALE = +0.438002 pp，R_BAL−SHUFFLE = +0.506100 pp。后两者平均改善不能弥补逐序和回访门槛失败。

| 流 | C | R_BAL | R_SCALE | R_SHUFFLE |
| --- | ---: | ---: | ---: | ---: |
| order0 | 79.107259 | 78.829690 | 79.004754 | 78.957893 |
| order1 | 77.999061 | 78.240621 | 77.189552 | 77.100217 |
| recurrence (order4) | 77.922143 | 77.087373 | 77.400577 | 77.373469 |

## 冻结 gate

| R_BAL 相对 | 主序均值 pp | order0 pp | order1 pp | recurrence pp | 均值 / 逐序 / 回访 |
| --- | ---: | ---: | ---: | ---: | --- |
| C | -0.018004 | -0.277569 | 0.241560 | -0.834770 | 未通过 / 未通过 / 未通过 |
| R_SCALE | 0.438002 | -0.175064 | 1.051068 | -0.313204 | 通过 / 未通过 / 未通过 |
| R_SHUFFLE | 0.506100 | -0.128203 | 1.140404 | -0.286096 | 通过 / 未通过 / 未通过 |

均值阈值分别为相对 C ≥+0.5、SCALE/SHUFFLE ≥+0.2 pp；逐主序都须 ≥0；回访每个差值须 ≥−0.1 pp。四域两主序 R_BAL−C 均须 ≥−2 pp，该项通过：

| 域 | remaining_dev 内容数 | R_BAL−C pp |
| --- | ---: | ---: |
| Drishti_GS | 37 | 1.738941 |
| ORIGA | 586 | 0.313433 |
| REFUGE | 336 | -0.261925 |
| REFUGE_Valid | 736 | -1.862467 |

不更改阈值、不追加 seed/顺序/轨迹、不启动 B。结论只限冻结开发集筛查；不作独立患者泛化、显著性或临床安全声明。

## 成本与完整性

正式成本：187296 forward、23412 backward、23412 Adam、0 VJP。三设备 smoke 共 480/60/60/0；合计 187776 forward、23472 backward/Adam、0 VJP。每设备 smoke 的 OLD/C/BAL/SCALE/SHUFFLE 各 4 访次；C 与旧路径 parity 均有效。

| 轨迹 | 方法 | 记录 / Adam | 主体秒数 | peak allocated bytes |
| --- | --- | ---: | ---: | ---: |
| o0a0 | C | 1951 / 1951 | 882.76 | 621460992 |
| o0a1 | R_BAL | 1951 / 1951 | 3575.12 | 621460992 |
| o0a2 | R_SCALE | 1951 / 1951 | 1841.63 | 621460992 |
| o0a3 | R_SHUFFLE | 1951 / 1951 | 853.38 | 621460992 |
| o1a0 | C | 1951 / 1951 | 3494.78 | 621460992 |
| o1a1 | R_BAL | 1951 / 1951 | 2014.76 | 621460992 |
| o1a2 | R_SCALE | 1951 / 1951 | 931.75 | 621299200 |
| o1a3 | R_SHUFFLE | 1951 / 1951 | 3509.10 | 621460992 |
| o4a0 | C | 1951 / 1951 | 1728.37 | 621460992 |
| o4a1 | R_BAL | 1951 / 1951 | 903.75 | 621460992 |
| o4a2 | R_SCALE | 1951 / 1951 | 3588.33 | 621460992 |
| o4a3 | R_SHUFFLE | 1951 / 1951 | 1895.93 | 621460992 |

成本上限：单轨迹 21600 秒、墙钟 86400 秒、累计活动 172800 秒、输出 8 GiB；原 launcher 的完成检查验证上限。最后 usage.json 的 active_workers 是采样值，完成判断采用最终 matrix、退出记录与有效结果指针。

CPU 标量重放由执行 SHA 内 `r6_regional_consistency.analyze.recompute` 完成：检查完整身份/顺序、因果性、计数、标量代数、metric 可实现性、跨轨迹 GT、gate 与成本。它不重建 tensor 空间位置、真实参数梯度或 ASSD 几何。归档阶段读取这些权威结果并核对绑定、计数、成本和 gate；没有重复模型推理，也未声称在文档发布 SHA 上重新运行历史 166 项测试。

## 聚合与复现边界

PUBLIC_AGGREGATE.json 保留全部四个子集（remaining_dev、legacy_dev、p1_extension_dev、all_dev）、所有主序与回访、OD/OC/macro、pre/q/post、逐域和内容池化结果、直接配对尾部、ASSD 共同有效分母及 undefined、无标签权重/梯度范数和更新位移摘要。内容池化均值不等于域等权主指标，不能混用。ASSD 单位为评估栅格像素，undefined 未填 0。回访仅报告注册流结果，未新增窗口或另行定义回访统计。

源代码、冻结配置和原始数学材料由本分支祖先提交提供。对有权访问私有执行账本的审阅者，可在执行 SHA 与原注册数据下调用既有 `analyze.recompute(output_root, registration)` 重放；该函数会重写指定输出根的结果指针，因此复核应使用独立账本副本。公开聚合支持主指标/gate/成本复算，但不能替代私有逐内容账本或重建 RGB/mask/ASSD 几何。

## 交付索引

- [完整去身份聚合](PUBLIC_AGGREGATE.json)
- [执行、退出、smoke、成本与授权摘要](EXECUTION_AUDIT.json)
- [交付与边界声明](DELIVERY.json)
- [已审准备材料](../../review/r6a_preparation/REVIEW_INDEX.md)

仅新增本结果目录，生产代码和科学配置均未改动。公开件不含逐内容身份/行、RGB、mask、checkpoint、optimizer、私有路径、PID 或设备 UUID；完整账本与运行日志保留在私有存储。既有外部 PASS 仅指阶段 I 的 A 实现审阅，本报告不签发新 review PASS 或执行授权。
