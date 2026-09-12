# B5 结果摘要

实验和独立 CPU 校验均已完成。**不将 H 保留为新的主方法候选；保留 H0 为零更新对照，本轮结构限定补救结束。**

F 只冻结三个指定头的六个 affine 标量；H 在 F 基础上保留三个头的源统计；H0 使用 H 的统计策略、全参数固定且每图仅一次前向。B4 的 C 保持原定义，其负结果不改写为实现错误。

主终点：remaining_dev，先域内均值、再三域等权，Dice 单位为百分比，差值为百分点。两个顺序复用同一批内容，不是独立队列。

| 方法 | order0 | order1 |
|---|---:|---:|
| N | 76.510373 | 76.510373 |
| A | 79.428436 | 79.105353 |
| EA | 79.689126 | 79.256312 |
| C | 74.946326 | 73.133208 |
| C0 | 74.307598 | 74.307598 |
| F | 74.910767 | 73.043217 |
| H | 75.115165 | 72.981078 |
| H0 | 74.328616 | 74.328616 |

| 主配对 | order0 | order1 |
|---|---:|---:|
| H-F | +0.204397 | -0.062139 |
| H-C | +0.168839 | -0.152130 |
| F-C | -0.035558 | -0.089991 |
| H0-C0 | +0.021019 | +0.021019 |
| H-H0 | +0.786549 | -1.347538 |
| H-A | -4.313272 | -6.124275 |
| H-EA | -4.573961 | -6.275234 |
| H-N | -1.395209 | -3.529295 |

H−F 随顺序改变符号；F−C 两序均略为负。H0−C0 在主子集只有 +0.021019 pp，在 all_dev 为 −0.002221 pp。这项指定干预没有显示三个头统计政策足以解释或修复 B4 退化；这也不是排除所有结构/历史因素的因果证明。H 对 C 两序描述性平均增量仅 +0.008355 pp。

H 在全部三个域、两个顺序的均值均低于 A/EA。下面保留每域 H−A 均值及最差 ceil(10%×n) 配对差均值；负值代表 H 更差。

| 域 | order0 H−A | order1 H−A | order0 最差10% | order1 最差10% |
|---|---:|---:|---:|---:|
| CVC-ClinicDB | -0.103537 | -8.170487 | -26.185665 | -55.446615 |
| ETIS-LaribPolypDB | -3.171674 | -4.839888 | -64.167303 | -66.171495 |
| Kvasir-SEG | -9.664603 | -5.362451 | -54.470098 | -39.905028 |

ETIS 的 H−A 最差单图为 −88.686606 pp，两序相同不构成独立重复。H−A 的共同有效 ASSD 平均差在所有域/序均为正（更差）；ETIS 为 +13.387852 / +17.550334 像素，对应各 115 对有效样本，11 对未共同定义。不将未定义 ASSD 记为零。

H−H0 在 order0 为 +0.786549 pp，order1 为 −1.347538 pp。持续优化没有稳定优于其零更新对照；域序历史仍影响结果。legacy_dev、p1_extension_dev 和 all_dev 的 H−A/H−EA 也全部为负，不通过更换子集改变结论。

正式：9,010 条真实评分、59,466 次前向、7,208 次反向/Adam。Smoke：100 次前向、12 次反向/Adam；合计 59,566 / 7,220 / 7,220。H0 独立预测 1,802 次、CPU 映射引用 3,604 个位置。source/DD/selector/outer/inner 新训练为零。

包含 smoke 的活跃阶段计时 9112.801 秒（约 151.9 分钟）；GPU 7，原环境。

| 新输出 | 平均 host 秒/图（不含 evaluator） | PyTorch allocated peak MiB |
|---|---:|---:|
| H0 | 0.066252 | 188.50 |
| F | 1.174704 | 475.47 |
| H | 1.217405 | 475.47 |

上述显存为 PyTorch allocated peak，非整卡显存占用；时间含当前实现的检查/标量采集，不据此声称受控加速比。网络调用数不是 FLOPs。8 项当前 CPU 测试、单次 GPU smoke 与正式标量重算均通过；ASSD 只复核保存标量和有效 cohort，没有重算丢弃的距离图。

执行提交：`3dc5decf75d627d75ec1afd77fe5abb9dcbdb81b`。此摘要仅补充解释与可读性，不改动任何实验配置、代码、逐图记录或既有结果。公开内容为代码、配置、聚合结果与审计；私有身份/路径/逐样本标量未公开。没有保存目标图像、mask、预测/激活图或适配后的模型/optimizer。

全部数据已用于开发。该结果支持结束本轮限定的三头补救，不支持新方法有效性、泛化或首次创新；不启动下一轮。

---
# B5 score-head preservation

B5_SCORE_HEAD_PRESERVATION_COMPLETE

Execution commit: 3dc5decf75d627d75ec1afd77fe5abb9dcbdb81b

At most partial or mixed recovery relative to C; consistent benefit over A/EA is not established. Do not promote H as a new effective main method or tune further automatically. Additional consistency optimization does not exceed H0 in both orders; retain the inexpensive zero-update control.

## remaining_dev / task_domain_macro_dice_percent

| Order | N | A | EA | O2 | D4 | C | C0 | H0 | F | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 76.510373 | 79.428436 | 79.689126 | 78.734012 | 79.686450 | 74.946326 | 74.307598 | 74.328616 | 74.910767 | 75.115165 |
| order1 | 76.510373 | 79.105353 | 79.256312 | 78.098019 | 77.616282 | 73.133208 | 74.307598 | 74.328616 | 73.043217 | 72.981078 |

## remaining_dev / task_comparisons_pp

| Order | H-F | H-C | F-C | H0-C0 | H-H0 | H-A | H-EA | H-N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 0.204397 | 0.168839 | -0.035558 | 0.021019 | 0.786549 | -4.313272 | -4.573961 | -1.395209 |
| order1 | -0.062139 | -0.152130 | -0.089991 | 0.021019 | -1.347538 | -6.124275 | -6.275234 | -3.529295 |

## legacy_dev / task_domain_macro_dice_percent

| Order | N | A | EA | O2 | D4 | C | C0 | H0 | F | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 78.461378 | 79.908197 | 81.034178 | 80.346747 | 80.412266 | 77.173263 | 77.159592 | 77.097146 | 77.218374 | 77.029666 |
| order1 | 78.461378 | 79.738532 | 80.913022 | 78.294175 | 81.321255 | 77.386514 | 77.159592 | 77.097146 | 77.278520 | 77.054302 |

## legacy_dev / task_comparisons_pp

| Order | H-F | H-C | F-C | H0-C0 | H-H0 | H-A | H-EA | H-N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | -0.188708 | -0.143597 | 0.045112 | -0.062446 | -0.067481 | -2.878531 | -4.004512 | -1.431712 |
| order1 | -0.224219 | -0.332212 | -0.107993 | -0.062446 | -0.042845 | -2.684230 | -3.858720 | -1.407076 |

## p1_extension_dev / task_domain_macro_dice_percent

| Order | N | A | EA | O2 | D4 | C | C0 | H0 | F | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 78.172439 | 81.054795 | 81.595596 | 79.369137 | 79.680475 | 79.158248 | 78.802491 | 78.746222 | 79.132988 | 79.210418 |
| order1 | 78.172439 | 80.304710 | 81.282401 | 75.633052 | 80.582167 | 78.049946 | 78.802491 | 78.746222 | 77.958909 | 77.800300 |

## p1_extension_dev / task_comparisons_pp

| Order | H-F | H-C | F-C | H0-C0 | H-H0 | H-A | H-EA | H-N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 0.077430 | 0.052170 | -0.025260 | -0.056269 | 0.464196 | -1.844377 | -2.385178 | 1.037979 |
| order1 | -0.158609 | -0.249646 | -0.091037 | -0.056269 | -0.945923 | -2.504410 | -3.482101 | -0.372139 |

## all_dev / task_domain_macro_dice_percent

| Order | N | A | EA | O2 | D4 | C | C0 | H0 | F | H |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 76.951272 | 79.858266 | 80.068498 | 79.058924 | 79.903984 | 75.868459 | 75.361448 | 75.359227 | 75.838704 | 75.997065 |
| order1 | 76.951272 | 79.463426 | 79.705695 | 78.085222 | 78.337377 | 74.211589 | 75.361448 | 75.359227 | 74.125162 | 74.033299 |

## all_dev / task_comparisons_pp

| Order | H-F | H-C | F-C | H0-C0 | H-H0 | H-A | H-EA | H-N |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| order0 | 0.158361 | 0.128607 | -0.029755 | -0.002221 | 0.637839 | -3.861201 | -4.071433 | -0.954207 |
| order1 | -0.091863 | -0.178289 | -0.086426 | -0.002221 | -1.325927 | -5.430126 | -5.672396 | -2.917973 |

## order0 remaining_dev domains and paired tails

CVC-ClinicDB: C=78.227144%, F=78.222718%, H=78.435726%, H0=77.246923%, A=78.539263%, EA=79.418113%

H-F: `{"assd_adverse_upper_decile_mean_px": 1.348126732820173, "assd_common_valid": 548, "assd_delta_mean_px": -0.11668470485311404, "assd_delta_median_px": 0.0035574937781263083, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 0.2130078996550843, "median": 0.028065220432205162, "minimum": -5.283075848542448, "n": 548, "negative": 224, "p10": -0.2485926647122849, "p90": 0.7143098983099475, "positive": 316, "tail_count": 55, "worst_decile_mean": -0.5923415815556858, "zero": 8}}`

H-H0: `{"assd_adverse_upper_decile_mean_px": 2.207348377684592, "assd_common_valid": 548, "assd_delta_mean_px": -1.5215902448542138, "assd_delta_median_px": -0.08082182695578488, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 1.18880280400916, "median": 0.14691123966408814, "minimum": -14.721913642086083, "n": 548, "negative": 199, "p10": -0.5262628660084601, "p90": 3.7816697493205846, "positive": 342, "tail_count": 55, "worst_decile_mean": -1.9270448033086318, "zero": 7}}`

H-A: `{"assd_adverse_upper_decile_mean_px": 24.059932679733446, "assd_common_valid": 529, "assd_delta_mean_px": 0.16400867621618878, "assd_delta_median_px": 0.20657026537817091, "assd_left_undefined": 0, "assd_not_jointly_defined": 19, "assd_right_undefined": 19, "dice_delta_pp": {"mean": -0.1035371655580701, "median": -0.32362915603838527, "minimum": -66.48493049651866, "n": 548, "negative": 322, "p10": -14.66137745464678, "p90": 6.09112023900329, "positive": 219, "tail_count": 55, "worst_decile_mean": -26.185664750284456, "zero": 7}}`

H-EA: `{"assd_adverse_upper_decile_mean_px": 30.799352816565587, "assd_common_valid": 528, "assd_delta_mean_px": 1.1313850809896429, "assd_delta_median_px": 0.15006660975607566, "assd_left_undefined": 0, "assd_not_jointly_defined": 20, "assd_right_undefined": 20, "dice_delta_pp": {"mean": -0.9823867607471968, "median": -0.2479565737597611, "minimum": -65.46577313293076, "n": 548, "negative": 296, "p10": -20.655772582619335, "p90": 10.700336148321592, "positive": 245, "tail_count": 55, "worst_decile_mean": -34.57496591707561, "zero": 7}}`

ETIS-LaribPolypDB: C=70.322813%, F=70.310624%, H=70.705872%, H0=67.076752%, A=73.877547%, EA=72.567182%

H-F: `{"assd_adverse_upper_decile_mean_px": 11.806962598965724, "assd_common_valid": 126, "assd_delta_mean_px": 0.6979607622157455, "assd_delta_median_px": 0.019897505759906897, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 0.39524796327316924, "median": 0.0, "minimum": -11.70913761801935, "n": 126, "negative": 62, "p10": -1.0451434119307823, "p90": 1.698542015324428, "positive": 58, "tail_count": 13, "worst_decile_mean": -3.57294235230633, "zero": 6}}`

H-H0: `{"assd_adverse_upper_decile_mean_px": 14.002951169720026, "assd_common_valid": 126, "assd_delta_mean_px": -4.573471726836648, "assd_delta_median_px": -0.39545073462628, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 3.6291201032931575, "median": 0.902454384565482, "minimum": -23.034090922951762, "n": 126, "negative": 34, "p10": -2.5534366406893243, "p90": 13.153129276897808, "positive": 87, "tail_count": 13, "worst_decile_mean": -8.880556191956625, "zero": 5}}`

H-A: `{"assd_adverse_upper_decile_mean_px": 94.32449702069404, "assd_common_valid": 115, "assd_delta_mean_px": 13.38785167799253, "assd_delta_median_px": 0.6607466620339046, "assd_left_undefined": 0, "assd_not_jointly_defined": 11, "assd_right_undefined": 11, "dice_delta_pp": {"mean": -3.171674216306568, "median": -1.067743976178992, "minimum": -88.68660598179454, "n": 126, "negative": 78, "p10": -40.27716381087414, "p90": 41.63386386324778, "positive": 43, "tail_count": 13, "worst_decile_mean": -64.1673033232455, "zero": 5}}`

H-EA: `{"assd_adverse_upper_decile_mean_px": 93.26220238412388, "assd_common_valid": 114, "assd_delta_mean_px": 13.095239237979525, "assd_delta_median_px": 0.8127279165321046, "assd_left_undefined": 0, "assd_not_jointly_defined": 12, "assd_right_undefined": 12, "dice_delta_pp": {"mean": -1.8613097390194828, "median": -1.5568897093256318, "minimum": -88.59060402684564, "n": 126, "negative": 78, "p10": -39.30041581190754, "p90": 43.04521790206739, "positive": 43, "tail_count": 13, "worst_decile_mean": -61.643193968374575, "zero": 5}}`

Kvasir-SEG: C=76.289020%, F=76.198960%, H=76.203896%, H0=78.662173%, A=85.868500%, EA=87.082083%

H-F: `{"assd_adverse_upper_decile_mean_px": 3.059257968352364, "assd_common_valid": 936, "assd_delta_mean_px": 0.0664097660416863, "assd_delta_median_px": -0.011311206700733045, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 0.00493620806448117, "median": 0.06489279129702208, "minimum": -8.02161199305006, "n": 936, "negative": 381, "p10": -1.181561121530439, "p90": 0.9932978545445781, "positive": 553, "tail_count": 94, "worst_decile_mean": -2.8760949589964926, "zero": 2}}`

H-H0: `{"assd_adverse_upper_decile_mean_px": 8.860151156518164, "assd_common_valid": 936, "assd_delta_mean_px": -0.9660568525513713, "assd_delta_median_px": 0.2713719605123851, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": -2.4582766975584094, "median": -1.0125769944785423, "minimum": -44.57977249353732, "n": 936, "negative": 699, "p10": -10.260430804712842, "p90": 2.6324651791258287, "positive": 236, "tail_count": 94, "worst_decile_mean": -19.701781334021693, "zero": 1}}`

H-A: `{"assd_adverse_upper_decile_mean_px": 33.38976333261135, "assd_common_valid": 919, "assd_delta_mean_px": 5.4047943707795785, "assd_delta_median_px": 1.1511424318142713, "assd_left_undefined": 0, "assd_not_jointly_defined": 17, "assd_right_undefined": 17, "dice_delta_pp": {"mean": -9.66460341552738, "median": -2.405101090228265, "minimum": -85.27630163995937, "n": 936, "negative": 762, "p10": -38.821002839912865, "p90": 0.9704363053510856, "positive": 172, "tail_count": 94, "worst_decile_mean": -54.4700977104409, "zero": 2}}`

H-EA: `{"assd_adverse_upper_decile_mean_px": 39.44177219543499, "assd_common_valid": 923, "assd_delta_mean_px": 6.374146415256707, "assd_delta_median_px": 1.2518277531370252, "assd_left_undefined": 0, "assd_not_jointly_defined": 13, "assd_right_undefined": 13, "dice_delta_pp": {"mean": -10.878186219169763, "median": -2.5972977211585038, "minimum": -91.66998795285099, "n": 936, "negative": 759, "p10": -42.731232144069295, "p90": 1.0587524713912277, "positive": 175, "tail_count": 94, "worst_decile_mean": -60.13087355759632, "zero": 2}}`

## order1 remaining_dev domains and paired tails

Kvasir-SEG: C=78.904902%, F=78.865620%, H=78.971315%, H0=78.662173%, A=84.333766%, EA=86.908199%

H-F: `{"assd_adverse_upper_decile_mean_px": 2.057843604265979, "assd_common_valid": 936, "assd_delta_mean_px": 0.0794903500938184, "assd_delta_median_px": 0.003788138602463964, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 0.10569546773584808, "median": 0.03190464978825025, "minimum": -3.467661464004046, "n": 936, "negative": 393, "p10": -0.35005871123872656, "p90": 0.5492920720151007, "positive": 541, "tail_count": 94, "worst_decile_mean": -0.7827520585766696, "zero": 2}}`

H-H0: `{"assd_adverse_upper_decile_mean_px": 4.387524895158936, "assd_common_valid": 936, "assd_delta_mean_px": -0.9324461149657628, "assd_delta_median_px": 0.0021463169798952575, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 0.3091420757943499, "median": -0.02355983345237178, "minimum": -34.94591235067649, "n": 936, "negative": 509, "p10": -1.899720820979789, "p90": 3.251280910541726, "positive": 425, "tail_count": 94, "worst_decile_mean": -5.719405058155315, "zero": 2}}`

H-A: `{"assd_adverse_upper_decile_mean_px": 30.377681403166534, "assd_common_valid": 921, "assd_delta_mean_px": 3.7186722237632317, "assd_delta_median_px": 0.4563683692546121, "assd_left_undefined": 0, "assd_not_jointly_defined": 15, "assd_right_undefined": 15, "dice_delta_pp": {"mean": -5.362450542590451, "median": -0.9022929221019382, "minimum": -79.0882090603776, "n": 936, "negative": 701, "p10": -23.10011990559949, "p90": 1.0759999861769036, "positive": 233, "tail_count": 94, "worst_decile_mean": -39.905027861691416, "zero": 2}}`

H-EA: `{"assd_adverse_upper_decile_mean_px": 42.60841779202324, "assd_common_valid": 925, "assd_delta_mean_px": 6.168487993056465, "assd_delta_median_px": 0.570770141886392, "assd_left_undefined": 0, "assd_not_jointly_defined": 11, "assd_right_undefined": 11, "dice_delta_pp": {"mean": -7.936884402040543, "median": -1.239102742226228, "minimum": -85.72129347880994, "n": 936, "negative": 701, "p10": -33.21437835739208, "p90": 1.000238957924593, "positive": 233, "tail_count": 94, "worst_decile_mean": -51.587117293117146, "zero": 2}}`

ETIS-LaribPolypDB: C=69.044412%, F=69.024939%, H=69.180082%, H0=67.076752%, A=74.019970%, EA=72.051873%

H-F: `{"assd_adverse_upper_decile_mean_px": 11.193594594486681, "assd_common_valid": 126, "assd_delta_mean_px": 0.23349899366664814, "assd_delta_median_px": 0.043235101298994105, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 0.15514303597565457, "median": 0.024269292443362955, "minimum": -14.101994845025878, "n": 126, "negative": 53, "p10": -1.9820983706405004, "p90": 1.7305855974444144, "positive": 67, "tail_count": 13, "worst_decile_mean": -4.499658601613089, "zero": 6}}`

H-H0: `{"assd_adverse_upper_decile_mean_px": 31.263240779536424, "assd_common_valid": 126, "assd_delta_mean_px": -1.9218102723164223, "assd_delta_median_px": -0.40129409317510933, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": 2.1033295966740884, "median": 0.5289300708195755, "minimum": -38.48461641436776, "n": 126, "negative": 48, "p10": -7.344639813640027, "p90": 12.731732780015717, "positive": 73, "tail_count": 13, "worst_decile_mean": -17.416147876315115, "zero": 5}}`

H-A: `{"assd_adverse_upper_decile_mean_px": 95.13540172880973, "assd_common_valid": 115, "assd_delta_mean_px": 17.550333969542795, "assd_delta_median_px": 2.1680884641378495, "assd_left_undefined": 0, "assd_not_jointly_defined": 11, "assd_right_undefined": 11, "dice_delta_pp": {"mean": -4.839887675637825, "median": -2.3243082723397146, "minimum": -88.68660598179454, "n": 126, "negative": 83, "p10": -53.50375549771005, "p90": 41.7000211108148, "positive": 38, "tail_count": 13, "worst_decile_mean": -66.17149527295022, "zero": 5}}`

H-EA: `{"assd_adverse_upper_decile_mean_px": 94.02482368460558, "assd_common_valid": 112, "assd_delta_mean_px": 17.330246821290217, "assd_delta_median_px": 4.123459865871499, "assd_left_undefined": 0, "assd_not_jointly_defined": 14, "assd_right_undefined": 14, "dice_delta_pp": {"mean": -2.871790612809325, "median": -1.6175631618262742, "minimum": -89.12751677852349, "n": 126, "negative": 77, "p10": -51.403236084670134, "p90": 55.216292976910175, "positive": 44, "tail_count": 13, "worst_decile_mean": -64.9374599508518, "zero": 5}}`

CVC-ClinicDB: C=71.450310%, F=71.239092%, H=70.791838%, H0=77.246923%, A=78.962325%, EA=78.808864%

H-F: `{"assd_adverse_upper_decile_mean_px": 3.415176765735136, "assd_common_valid": 548, "assd_delta_mean_px": 0.05282837616014107, "assd_delta_median_px": -0.003932391456297424, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": -0.4472546758263823, "median": 0.01808813602693915, "minimum": -11.019959781140642, "n": 548, "negative": 258, "p10": -2.725155956929794, "p90": 0.8732434535320033, "positive": 285, "tail_count": 55, "worst_decile_mean": -4.9942400652273635, "zero": 5}}`

H-H0: `{"assd_adverse_upper_decile_mean_px": 11.748148000185036, "assd_common_valid": 548, "assd_delta_mean_px": -1.3176089149248082, "assd_delta_median_px": 0.23990412958686458, "assd_left_undefined": 0, "assd_not_jointly_defined": 0, "assd_right_undefined": 0, "dice_delta_pp": {"mean": -6.455085176910081, "median": -2.1258141086667512, "minimum": -49.24028418894785, "n": 548, "negative": 412, "p10": -25.290665648569735, "p90": 2.4247444876195248, "positive": 133, "tail_count": 55, "worst_decile_mean": -34.558370459340196, "zero": 3}}`

H-A: `{"assd_adverse_upper_decile_mean_px": 28.483876730155128, "assd_common_valid": 520, "assd_delta_mean_px": 2.1510947387749053, "assd_delta_median_px": 1.2410823038621437, "assd_left_undefined": 0, "assd_not_jointly_defined": 28, "assd_right_undefined": 28, "dice_delta_pp": {"mean": -8.170487410218144, "median": -2.819611802441985, "minimum": -83.83448350949705, "n": 548, "negative": 389, "p10": -40.009931599076126, "p90": 6.305695043856496, "positive": 155, "tail_count": 55, "worst_decile_mean": -55.44661495553607, "zero": 4}}`

H-EA: `{"assd_adverse_upper_decile_mean_px": 30.132724818257362, "assd_common_valid": 522, "assd_delta_mean_px": 1.6810955794034375, "assd_delta_median_px": 1.137829539996067, "assd_left_undefined": 0, "assd_not_jointly_defined": 26, "assd_right_undefined": 26, "dice_delta_pp": {"mean": -8.0170261161474, "median": -2.6586846206794523, "minimum": -84.01041946852834, "n": 548, "negative": 369, "p10": -40.912891470283576, "p90": 7.560250520124631, "positive": 175, "tail_count": 55, "worst_decile_mean": -57.37093489095147, "zero": 4}}`

## Cost and scope

{"records": 9010, "forwards": 59466, "backwards": 7208, "base_adam": 7208, "perturb": 0, "restore": 0}; including smoke: {"forwards": 59566, "backwards": 7220, "base_adam": 7220, "perturb": 0, "restore": 0}
H0 independent predictions: 1802; CPU references: 3604.
Active-stage seconds: 9112.800878924085
{"0_F": {"host_seconds": 2106.42309596017, "peak_allocated_bytes": 498566656, "pipeline_seconds": 2168.034375381656, "records": 1802}, "0_H": {"host_seconds": 2203.9150984312873, "peak_allocated_bytes": 498569728, "pipeline_seconds": 2256.8113413148094, "records": 1802}, "1_F": {"host_seconds": 2127.208821193315, "peak_allocated_bytes": 498566656, "pipeline_seconds": 2180.578582611168, "records": 1802}, "1_H": {"host_seconds": 2183.6109211251605, "peak_allocated_bytes": 498569728, "pipeline_seconds": 2234.537201527506, "records": 1802}, "H0": {"host_seconds": 119.38622915605083, "peak_allocated_bytes": 197652480, "pipeline_seconds": 199.47082585818134, "records": 1802}}

F/H reuse B4 PolypC.step unchanged; only three exact head affine/statistics policies differ. Loss crosses frozen heads without no_grad. H0 has no optimizer. The scalar mechanism observations are from the existing final-original forward; they do not establish calibrated probability or unique causality.
All domains, four subsets, unrounded paired signs, ceil(10%) tails, common-valid ASSD, undefined/empty/full, FP/FN and pixel totals are in public_aggregate.json. CPU reconstructs Dice from scalar counts and ASSD cohorts from recorded values, not discarded maps.
Host timing excludes evaluator; pipeline timing includes it. Network calls are not FLOPs, old timings are historical, and repeated content/orders are not independent cohorts.
No new Fundus run or source/DD/selector/outer/inner training. DD, SA, G, U/S/I, fusion, radius/LR/layer search remain closed. B4 negative results are preserved. Completion is distinct from scientific utility; no automatic next experiment.
