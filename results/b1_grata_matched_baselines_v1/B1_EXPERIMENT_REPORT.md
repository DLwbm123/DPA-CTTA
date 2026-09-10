# B1 matched GraTa continuous-stream baseline comparison

B1_MATCHED_BASELINE_COMPARISON_COMPLETE

Execution commit: 029a79340057b58b8074ccf86c7b5671b5fdbd1c

C/G are C-CTTA/G-CTTA on the P2 segmentation checkpoint and continuous protocol, not a complete original-paper reproduction.

## 完成后的结果解释

本轮得到一个均值更强的外部适配参考：**简单一致性 C-CTTA 的域等权 Dice 优于 A，且优于 G-CTTA；但存在明显逐域退化和顺序依赖，不能作为所有域统一替代 A 的依据。**

在主要子集 remaining_dev 上，C−A 为 +4.844977 / +3.463403 pp，G−A 为 +3.112821 / +2.544341 pp；G−C 为 −1.732156 / −0.919063 pp。C 在 legacy_dev、p1_extension_dev、all_dev 两序的任务均值也最高。G 的发布版扰动与动态 LR 组合没有带来整体增量，不能因其机制复杂就优先选择 G。

收益主要来自 ORIGA：C−A 为 +18.482492 / +17.949022 pp；但 REFUGE_Valid 为 −9.559154 / −5.169503 pp。该域的 OC Dice 分别下降14.585200 / 9.211466 pp，OC ASSD 在各自736个共同有效内容中增加8.619837 / 5.048136 px，说明退化并非只表现为细小的均值换位。C−A 的该域最差10%宏Dice配对差为 −22.184945 / −16.132152 pp。

G 在 REFUGE_Valid 相比 C 提高6.264363 / 2.560639 pp，缓和了这部分退化，但仍低于 A；同时在 ORIGA 相比 C 下降6.726974 / 6.655344 pp。G 是不同取舍，而非在所有域都无效。Drishti_GS 的 C−A 从顺序0的 +9.195009 翻转到顺序1的 −1.193137，不能忽视持续历史和域序。

按预登记规则，保留 C 为较简单的强参考，并保留 A 以展示其优势域；当前没有必要强推 G 的额外更新规则。成本是否适合实际用途仍取决于部署延迟预算。本轮结束，不自动开启 LR/norm/loss 搜索、逐域真值选择器或接回 DD。

C/G 对 A 的比较同时改变可更新参数、归一化规则、目标和状态机制，不能据此把收益因果归于“BN参数更多”或某个单独组件；G−C 也未拆分扰动与动态LR的独立作用。


### 主要子集：逐域宏 Dice 配对差（pp）

| 域序 | 域 | C−A | G−A | G−C | C−A负向数量/组数 | C−A最差10%均值 |
|---|---|---:|---:|---:|---:|---:|
| order0 | REFUGE | +1.261561 | +0.769483 | -0.492078 | 94/336 | -3.105783 |
| order0 | ORIGA | +18.482492 | +11.755518 | -6.726974 | 39/586 | -2.197270 |
| order0 | REFUGE_Valid | -9.559154 | -3.294790 | +6.264363 | 653/736 | -22.184945 |
| order0 | Drishti_GS | +9.195009 | +3.221074 | -5.973935 | 1/37 | +0.561517 |
| order1 | Drishti_GS | -1.193137 | -1.658582 | -0.465445 | 27/37 | -4.914477 |
| order1 | REFUGE_Valid | -5.169503 | -2.608863 | +2.560639 | 616/736 | -16.132152 |
| order1 | ORIGA | +17.949022 | +11.293678 | -6.655344 | 45/586 | -2.529298 |
| order1 | REFUGE | +2.267232 | +3.151130 | +0.883898 | 123/336 | -8.556490 |

最差10%按各自配对差排序，取ceil(10%×n)；不是同一个预设难例集合。小域在主指标中与大域等权，不能将域等权收益解释为每张图或每个患者的统一收益。

### 运行代价与学习率

| 方法/顺序 | 前向/图 | 反向/图 | Adam/图 | host秒/图 | 观测allocated峰值 MiB | 平均LR |
|---|---:|---:|---:|---:|---:|---:|
| C/0 | 8 | 1 | 1 | 0.706948 | 588.469 | 0.000100000 |
| C/1 | 8 | 1 | 1 | 0.719909 | 585.094 | 0.000100000 |
| G/0 | 9 | 2 | 1 | 0.928069 | 585.249 | 0.000024670 |
| G/1 | 9 | 2 | 1 | 1.065513 | 585.249 | 0.000024554 |

两序合计 C/G host 平均约0.713429 / 0.996791秒/图；G在各序比C耗时约31.3% / 48.0%。host时间包含更新和最终预测、排除evaluator；这是共存GPU上的本次观测，不是隔离硬件下的受控速度基准。前向8/9次、反向1/2次则是已核对的实际调用数。

两方法均更新41层BN的19136个affine标量（82个参数张量），源文件及其余网络参数不变。C固定LR=1e−4；G两序平均LR约2.46e−5，没有零LR访问。此现象不证明G劣势由LR单独造成，未运行LR扫描。

旧A的P2实测host时间约0.152313 / 0.142466秒/图，但来自不同运行和共享调度，只作历史成本参考，不给出受控加速比。旧P2的显存峰值是多父方法共享峰值，不能与本轮单轨迹峰值直接作方法显存优劣比较。旧O2/D4还需要凝缩图像与源mask；C/G没有源样本rehearsal。

### 完成及历史审计

正式记录7804条、base Adam7804次、网络前向66334次、反向11706次、G扰动与恢复各3902次，全部与计划一致。含已通过smoke共7820次base Adam；新source/DD/outer训练=0。GPU活跃阶段计时7395.097秒，约2.054小时（包含本次smoke与正式阶段的评分/等待，不是纯kernel时间）。后台run和独立CPU recompute退出码均为0，完成时间2026-09-09T14:17:18.097372+00:00。

首轮返回值错误发生于任何GPU Adam之前；零更新失败记录保留在ee4b434发布版本。用户明确批准后，修复版本029a79340057b58b8074ccf86c7b5671b5fdbd1c通过7项CPU回归及16次GPU配对smoke，随后完整执行本次四条轨迹。修复只包装模型返回值，没有修改发布更新规则或目标。完整结果的发布提交与执行提交分开。

独立CPU重读新记录和P2旧记录，验证内容身份、顺序、子集、计数、Adam步数、cosine/LR关系及由像素标量计数还原的Dice。ASSD只复核标量及有效cohort，不声称重新计算已丢弃的像素距离图。全部域/OD/OC/子集的分布、正零负、最差单图和ASSD共同有效cohort见public_aggregate.json。

所有内容均为开发用途，单seed、两个相同内容的域序不能作为患者独立重复或统计显著性。仅Fundus；没有Polyp移植、source遗忘测量、原创方法/SOTA/临床安全声明。公开代码、配置、许可证与去标识聚合；原始图像、mask、checkpoint、概率图、逐图身份及路径不公开。

## remaining_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 1695 | 68.200582 | 74.259279 | 73.426605 | 72.891721 | 72.686550 | 79.104256 | 77.372100 |
| order1 | 1695 | 68.200582 | 74.542945 | 73.813586 | 75.423504 | 74.305555 | 78.006348 | 77.087285 |

| Order | G-C | C-A | G-A | C-N | G-N | G-O2 | G-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -1.732156 | +4.844977 | +3.112821 | +10.903674 | +9.171519 | +4.480380 | +4.685551 |
| order1 | -0.919063 | +3.463403 | +2.544341 | +9.805766 | +8.886704 | +1.663781 | +2.781730 |

## legacy_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 128 | 69.092892 | 74.538998 | 73.801632 | 73.653989 | 73.009535 | 79.306413 | 77.579453 |
| order1 | 128 | 69.092892 | 75.140702 | 74.439300 | 75.885483 | 74.336631 | 77.919338 | 77.059848 |

| Order | G-C | C-A | G-A | C-N | G-N | G-O2 | G-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -1.726960 | +4.767415 | +3.040455 | +10.213521 | +8.486561 | +3.925464 | +4.569918 |
| order1 | -0.859490 | +2.778636 | +1.919146 | +8.826445 | +7.966955 | +1.174364 | +2.723217 |

## p1_extension_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 128 | 66.452389 | 73.796285 | 72.745511 | 72.436336 | 71.836462 | 79.497100 | 77.709793 |
| order1 | 128 | 66.452389 | 74.506257 | 73.700675 | 76.426326 | 74.433735 | 78.551758 | 77.384891 |

| Order | G-C | C-A | G-A | C-N | G-N | G-O2 | G-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -1.787308 | +5.700816 | +3.913508 | +13.044711 | +11.257403 | +5.273456 | +5.873331 |
| order1 | -1.166867 | +4.045501 | +2.878634 | +12.099369 | +10.932502 | +0.958565 | +2.951156 |

## all_dev

| Order | Groups | N | A | EA | O2 | D4 | C | G |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 1951 | 67.964224 | 74.328553 | 73.338857 | 72.983086 | 72.604889 | 79.230033 | 77.622550 |
| order1 | 1951 | 67.964224 | 74.691960 | 73.806501 | 75.548687 | 74.372899 | 78.241148 | 77.283082 |

| Order | G-C | C-A | G-A | C-N | G-N | G-O2 | G-D4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| order0 | -1.607483 | +4.901480 | +3.293998 | +11.265809 | +9.658326 | +4.639465 | +5.017661 |
| order1 | -0.958066 | +3.549188 | +2.591122 | +10.276923 | +9.318858 | +1.734395 | +2.910183 |

All domain/channel paired distributions, signs, worst decile/single and ASSD common-valid cohorts are in public_aggregate.json. No OD/OC macro ASSD. No patient independence, statistical significance, novelty, clinical safety or untouched-test claim.
C/G change the full adaptation scheme relative to A. G-C measures the combined published perturbation and dynamic-LR rule, not either component separately. Published entropy omits the Bernoulli complement term; preserved deliberately.
Old P2 controls bind the exact checkpoint, evaluator, content and order. Old proxy arms require condensed images/source masks at deployment; C/G do not rehearse source samples. No new proxy/source training. C/G do not use fixed source ensembling.
Cost includes 8 C / 9 G model forwards per image, 1 / 2 backwards and one base Adam; G perturb and restore are counted separately. Old P2 peaks are shared-schedule measurements, not isolated per-arm peaks.
CPU reconstruction rereads every scalar record and restores Dice from pixel counts; it cannot recompute discarded pixel ASSD. Source file unchanged and only registered BN affine parameters adapt.
Engineering completion does not establish usefulness. No automatic further run, parameter search or DD continuation.
