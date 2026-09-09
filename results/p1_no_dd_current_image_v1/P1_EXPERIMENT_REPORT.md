# P1 no-DD current-image source anchor

P1_NO_DD_COMPARISON_COMPLETE

Execution commit: f65e119016f2d6a32cd0cffbee2bb2e94f570c2d

## 完成结论（2026-09-09 实时核验）

**P1 已完成，科研判断为“混合”：Polyp 的主要收益来自固定概率集成；新增 SA 约束未显示跨任务稳定的实质增量。** 正式 run 与独立 CPU 标量重算退出码均为 0；448 个内容组（224 legacy + 224 extension），7,168 条新评分、4,480 次正式 online Adam、896 次共享 teacher 前向全部覆盖。含 smoke 为 4,492 次 online；无新的 outer、可微 inner、代理或 source 训练。正式后台流程约 23.91 分钟，含 smoke 的活跃阶段约 24.13 分钟；GPU 7。

按预定主终点 extension_dev 解读，单位均为 Dice 百分点：

- Fundus：SA−A 为 +0.1643 / +0.1454，ENS_SA−ENS_A 为 +0.0276 / +0.0892。两序小幅同向，但 SA 仍低于 O2（−0.1684 / −0.1431）；不能将这点改善写为方法突破或达到资源优先级参考线。
- Polyp：SA−A 为 +0.0205 / −0.0276，ENS_SA−ENS_A 为 +0.0028 / −0.0320，方向不稳且接近零。ENS_A−A 为 +1.7670 / +1.2616，ENS_A−N 为 +3.6305 / +4.6961：观察到的任务均值收益属于固定概率平均，不应归功于 anchor loss。
- 融合存在明确代价。Fundus 的 ENS_A−A 均值为 −0.9728 / −0.3251，ORIGA 域下降 −5.3068 / −3.7336。Polyp 的 ETIS 域下降 −2.2839 / −2.3205：尽管 Polyp 均值改善，融合牺牲了本来有效的适配收益，不能称为所有域都更安全。
- 新增 Kvasir 组中，原顺序 ENS_A 仍低于 N 0.7106 pp，反转顺序则高于 N 1.4350 pp。对方法的判断需要同时保留域差异与顺序依赖，不能使用 GT 逐域挑选 N/A/ensemble 拼成可部署方法。

全部 legacy/combined、八方法、两顺序及各域/通道的配对分布、最差十分位、最差单图和 ASSD common-valid/恶化尾部均在 public_aggregate.json，成本和实际计算计数在 execution_audit.json。ASSD 未定义项未置零。两种顺序共享同一批内容，单 seed、已暴露开发域及 UNKNOWN patient/video linkage 限制不变；不作显著性或临床部署结论。本次未新增 source 保持性/遗忘评分。

本轮停止，不自动搜索 lambda、融合系数或启动 P2。保留无 DD 简单融合在 Polyp 上的描述性正结果，同时如实保留 Fundus 与 ETIS 等域的负结果，以及完整历史 DD 对照优于 SA 的情况。

准备阶段曾修复 M4 登记结构读取 KeyError，未消耗 GPU 更新；该失败记录和 CPU 夹具修正见[启动报告](https://github.com/DLwbm123/DPA-CTTA/blob/3e83a972ffb2d325cd03637a608e273aa4f453f8/results/p1_no_dd_current_image_v1/P1_EXPERIMENT_REPORT.md)。正式阶段无失败。实际执行提交与此次结果发布提交分开；公开内容不含私有数据、逐样本身份、路径、逐资产摘要、模型或代理。

No offline training; lambda=0.1, probability averaging=0.5 fixed. Extension development is primary; it is not an untouched or patient-independent test.

## extension_dev

| Task/order | Groups | N | A | ENS_A | SA | ENS_SA | O2 | D4 | L4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 128 | 66.452389 | 75.530124 | 74.557334 | 75.694456 | 74.584981 | 75.862848 | 75.855592 | 75.251903 |
| fundus/order1 | 128 | 66.452389 | 74.417744 | 74.092643 | 74.563188 | 74.181860 | 74.706328 | 74.578011 | 74.470338 |
| polyp/order0 | 96 | 78.172439 | 80.035920 | 81.802933 | 80.056420 | 81.805723 | 79.850722 | 80.400210 | 79.950987 |
| polyp/order1 | 96 | 78.172439 | 81.606976 | 82.868578 | 81.579378 | 82.836592 | 81.293925 | 81.380334 | 81.156179 |

| Task/order | SA-A | ENS_SA-ENS_A | ENS_A-A | ENS_A-N |
| --- | ---: | ---: | ---: | ---: |
| fundus/order0 | +0.164332 | +0.027647 | -0.972790 | +8.104945 |
| fundus/order1 | +0.145444 | +0.089218 | -0.325102 | +7.640253 |
| polyp/order0 | +0.020499 | +0.002790 | +1.767013 | +3.630494 |
| polyp/order1 | -0.027598 | -0.031986 | +1.261602 | +4.696139 |
## legacy_dev

| Task/order | Groups | N | A | ENS_A | SA | ENS_SA | O2 | D4 | L4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 128 | 69.092892 | 76.571807 | 75.623136 | 76.646396 | 75.584131 | 76.779898 | 77.047866 | 76.384693 |
| fundus/order1 | 128 | 69.092892 | 75.811600 | 75.059481 | 75.872907 | 75.036303 | 75.784095 | 75.767521 | 75.639887 |
| polyp/order0 | 96 | 78.461378 | 77.706947 | 82.206507 | 77.797865 | 82.198541 | 77.642892 | 78.216371 | 77.471006 |
| polyp/order1 | 96 | 78.461378 | 79.630010 | 82.184525 | 79.617003 | 82.140482 | 79.513938 | 79.370609 | 79.592992 |

| Task/order | SA-A | ENS_SA-ENS_A | ENS_A-A | ENS_A-N |
| --- | ---: | ---: | ---: | ---: |
| fundus/order0 | +0.074589 | -0.039006 | -0.948670 | +6.530244 |
| fundus/order1 | +0.061308 | -0.023178 | -0.752118 | +5.966589 |
| polyp/order0 | +0.090918 | -0.007965 | +4.499560 | +3.745129 |
| polyp/order1 | -0.013007 | -0.044043 | +2.554516 | +3.723147 |
## combined

| Task/order | Groups | N | A | ENS_A | SA | ENS_SA | O2 | D4 | L4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus/order0 | 256 | 67.772641 | 76.050966 | 75.090235 | 76.170426 | 75.084556 | 76.321373 | 76.451729 | 75.818298 |
| fundus/order1 | 256 | 67.772641 | 75.114672 | 74.576062 | 75.218048 | 74.609082 | 75.245212 | 75.172766 | 75.055113 |
| polyp/order0 | 192 | 78.316909 | 78.871434 | 82.004720 | 78.927142 | 82.002132 | 78.746807 | 79.308291 | 78.710997 |
| polyp/order1 | 192 | 78.316909 | 80.618493 | 82.526552 | 80.598190 | 82.488537 | 80.403931 | 80.375472 | 80.374585 |

| Task/order | SA-A | ENS_SA-ENS_A | ENS_A-A | ENS_A-N |
| --- | ---: | ---: | ---: | ---: |
| fundus/order0 | +0.119461 | -0.005679 | -0.960730 | +7.317595 |
| fundus/order1 | +0.103376 | +0.033020 | -0.538610 | +6.803421 |
| polyp/order0 | +0.055709 | -0.002588 | +3.133286 | +3.687811 |
| polyp/order1 | -0.020302 | -0.038015 | +1.908059 | +4.209643 |

## Audit and limits

{"records": 7168, "online": 4480, "outer": 0, "inner": 0, "teacher_forwards": 896}; smoke online=12; outer/inner/teacher training=0.
Task means are equal-domain (Fundus OD/OC macro). All domain/channel/subset paired distributions, signs, worst decile/single, ASSD common-valid and adverse tails are in public_aggregate.json; undefined ASSD is never zero-imputed.
Eight predictions use five adapting trajectories. Ensemble rows carry zero physical updates and bind to parent prediction/state identifiers. Source predictions are shared within a visit; standalone costs charge teacher computation to each applicable method.
GT is read only after N/A/SA/ENS_A/ENS_SA are fixed; historical proxy controls also predict first. No target probability/logit/image/mask files are stored. Independent CPU process verifies coverage, parents, counts, Dice from pixel counts and scalar aggregates; it cannot reconstruct discarded pixel maps or independently recompute pixel ASSD.
Legacy groups occur within new expanded histories. Different orders reuse the same contents. Single seed, previously exposed development and UNKNOWN patient/video links prohibit unsupported statistical/clinical claims. No new source-clean or forgetting measurement.
Completion is distinct from usefulness. Apply the frozen interpretation rules to extension results and all controls/tails; no performance gates, coefficient search or automatic P2. Source anchoring and ensembling are established ideas; no novelty/SOTA claim.
Only source/config/tests and deidentified aggregate evidence are public. Data, checkpoint/proxy files, paths, group identifiers and per-asset digests remain private.
