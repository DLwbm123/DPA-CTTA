# M4 trajectory distillation report

M4_TRAJECTORY_COMPARISON_COMPLETE

Execution commit: f88e99d212e914b74a7c521bac8c418fc73e5d6c

## 中文结论

**工程执行已完成，M4 的跨图像轨迹梯度收益假设未获本轮支持。** 六套训练各完成 600 次访问 / 150 次外层更新，2,172 条新评分及独立 CPU 重算全部完成，run/recompute 退出码均为 0。没有非有限梯度中止；所有训练块的零梯度比例均为 0。前次数值实现故障已解决，当前效果结论来自完成的实验。

核心匹配比较是 T4−L4（Dice 百分点，按域等权）：Fundus 原顺序 −0.1133、反转域顺序 −0.0770；Polyp 分别 −0.1317、−0.1482。四种设置均未改善。L4 与 T4 使用相同四步前向目标和预算，区别在跨图像状态导数；这比与历史 O2 的比较更直接检验本轮假设。

Fundus 的 D4 在两个顺序下分别为 76.8872 / 76.3161，均高于 L4、T4；Polyp 中 O2 在原顺序为 78.4944，L4 在反转顺序为 78.8045。T4 四种设置均低于 O2，不能解释为“只输给了某一个弱对照”。相对 N，T4 在 Fundus 提高约 7.08–7.56 个百分点，在 Polyp 为 −0.1061 / +0.1949；因此不能笼统总结为 N 在所有任务上最好。

退化并非只存在于整体均值。REFUGE_Valid 的 T4−L4 域均值为 −0.6120 / −0.3849；Polyp 的 ETIS 两个顺序也均为负，最差单图差值为 −7.5781 / −6.0655。这里的单图是尾部描述，不能当作患者级显著性。全部 ASSD common-valid 与缺失计数、恶化尾部以及 Dice 分位数保留在 aggregate 中，不用均值掩盖它们。

T4 相对 L4 的实测训练墙钟时间增加约 28.9%（Fundus）和 11.2%（Polyp）；这些是当次共享设备的耗时观察，不是受控 FLOPs 结论。源域 fresh-host 分数中 T4 也未超过 L4，且所有适应臂均低于 N；它不是 target 适应后遗忘测量。

研究判断：本轮应记录为**执行成功、未观察到轨迹导数净收益**，不据此推进为有效方法或 SOTA 结论。差值较小，只有单 seed；两种顺序共享样本，不是独立重复实验，不能声称统计显著或证明轨迹蒸馏普遍无效。没有自动增加实验或启动 M5。

修复及验收的历史依据见 [启动报告](https://github.com/DLwbm123/DPA-CTTA/blob/a1cf11625571f11b9afbca2754582c147b0af8e5/results/m4_trajectory_distillation_v1/M4_EXPERIMENT_REPORT.md) 与 [repair evidence](repair_parity_evidence.json)。标准差零点二阶导数延拓约定、原生一阶数值对齐和修复前缀预算保持公开。

Six source-only fits completed. Each fit used 600 image visits, 150 outer updates and 600 functional inner Adam updates. Four distinct images per window; K=4 proxy batch unchanged. D4/L4 detach each incoming image state; T4 retains within-window Adam and memory-value derivatives, including the host gradient under native AdaBN stop-gradient statistics. All use native identity-proxy online deployment.

## Target order0

| Task / domain | N | A | D2 | O2 | D4 | L4 | T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / domain-equal mean | 69.092892 | 76.606283 | 76.843102 | 76.725005 | 76.887194 | 76.762589 | 76.649272 |
| fundus / REFUGE | 82.646698 | 84.758880 | 84.915672 | 84.832865 | 84.954949 | 84.975844 | 84.975149 |
| fundus / ORIGA | 56.776000 | 72.100268 | 72.593176 | 72.490236 | 72.680596 | 72.257882 | 72.484088 |
| fundus / REFUGE_Valid | 62.337465 | 74.724864 | 73.153641 | 72.493807 | 73.069215 | 72.478218 | 71.866206 |
| fundus / Drishti_GS | 74.611408 | 74.841118 | 76.709921 | 77.083113 | 76.844016 | 77.338414 | 77.271645 |
| polyp / domain-equal mean | 78.461378 | 78.437953 | 78.321466 | 78.494365 | 78.361303 | 78.486956 | 78.355252 |
| polyp / CVC-ClinicDB | 78.481590 | 70.492085 | 70.500488 | 70.505629 | 70.440720 | 70.286719 | 70.344286 |
| polyp / ETIS-LaribPolypDB | 68.829560 | 81.532518 | 81.174384 | 81.619834 | 81.393582 | 82.015633 | 81.841965 |
| polyp / Kvasir-SEG | 88.072985 | 83.289255 | 83.289527 | 83.357631 | 83.249608 | 83.158516 | 82.879505 |

| Task / domain | T4-L4 | T4-D4 | L4-O2 | T4-A | T4-D2 | T4-O2 | T4-N | T4-O3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / mean | -0.113317 | -0.237922 | +0.037584 | +0.042989 | -0.193830 | -0.075733 | +7.556380 | -0.217322 |
| fundus / REFUGE | -0.000695 | +0.020200 | +0.142979 | +0.216268 | +0.059477 | +0.142283 | +2.328451 | +0.108919 |
| fundus / ORIGA | +0.226207 | -0.196508 | -0.232354 | +0.383820 | -0.109087 | -0.006148 | +15.708089 | -0.450346 |
| fundus / REFUGE_Valid | -0.612012 | -1.203008 | -0.015589 | -2.858658 | -1.287435 | -0.627601 | +9.528742 | -2.165284 |
| fundus / Drishti_GS | -0.066769 | +0.427629 | +0.255300 | +2.430527 | +0.561724 | +0.188532 | +2.660237 | +1.637421 |
| polyp / mean | -0.131704 | -0.006051 | -0.007409 | -0.082701 | +0.033786 | -0.139113 | -0.106126 | +0.204178 |
| polyp / CVC-ClinicDB | +0.057568 | -0.096434 | -0.218911 | -0.147799 | -0.156202 | -0.161343 | -8.137304 | +0.137356 |
| polyp / ETIS-LaribPolypDB | -0.173668 | +0.448383 | +0.395798 | +0.309447 | +0.667581 | +0.222130 | +13.012405 | +0.714535 |
| polyp / Kvasir-SEG | -0.279010 | -0.370103 | -0.199115 | -0.409750 | -0.410021 | -0.478125 | -5.193479 | -0.239358 |
## Target order1

| Task / domain | N | A | D2 | O2 | D4 | L4 | T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / domain-equal mean | 69.092892 | 76.230920 | 76.286910 | 76.250384 | 76.316133 | 76.248625 | 76.171602 |
| fundus / Drishti_GS | 74.611408 | 76.036129 | 76.583826 | 76.704255 | 76.610468 | 76.668746 | 76.639132 |
| fundus / REFUGE_Valid | 62.337465 | 73.878087 | 72.277910 | 71.729164 | 72.331561 | 72.039653 | 71.654709 |
| fundus / ORIGA | 56.776000 | 70.705645 | 71.364344 | 71.873221 | 71.406710 | 71.338756 | 71.502006 |
| fundus / REFUGE | 82.646698 | 84.303819 | 84.921559 | 84.694895 | 84.915792 | 84.947343 | 84.890561 |
| polyp / domain-equal mean | 78.461378 | 78.741572 | 78.540971 | 78.738218 | 78.561349 | 78.804480 | 78.656283 |
| polyp / Kvasir-SEG | 88.072985 | 82.078664 | 82.027753 | 82.182044 | 82.015180 | 81.993351 | 82.062982 |
| polyp / ETIS-LaribPolypDB | 68.829560 | 81.536636 | 81.191204 | 81.582235 | 81.385446 | 81.967195 | 81.730102 |
| polyp / CVC-ClinicDB | 78.481590 | 72.609417 | 72.403956 | 72.450376 | 72.283422 | 72.452894 | 72.175765 |

| Task / domain | T4-L4 | T4-D4 | L4-O2 | T4-A | T4-D2 | T4-O2 | T4-N | T4-O3 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / mean | -0.077023 | -0.144531 | -0.001759 | -0.059318 | -0.115308 | -0.078782 | +7.078709 | N/A |
| fundus / Drishti_GS | -0.029614 | +0.028664 | -0.035509 | +0.603002 | +0.055306 | -0.065123 | +2.027724 | N/A |
| fundus / REFUGE_Valid | -0.384944 | -0.676852 | +0.310489 | -2.223378 | -0.623201 | -0.074455 | +9.317244 | N/A |
| fundus / ORIGA | +0.163250 | +0.095295 | -0.534465 | +0.796361 | +0.137662 | -0.371216 | +14.726006 | N/A |
| fundus / REFUGE | -0.056783 | -0.025232 | +0.252449 | +0.586742 | -0.030999 | +0.195666 | +2.243862 | N/A |
| polyp / mean | -0.148197 | +0.094934 | +0.066262 | -0.085289 | +0.115312 | -0.081935 | +0.194905 | N/A |
| polyp / Kvasir-SEG | +0.069631 | +0.047802 | -0.188693 | -0.015682 | +0.035229 | -0.119062 | -6.010003 | N/A |
| polyp / ETIS-LaribPolypDB | -0.237092 | +0.344657 | +0.384960 | +0.193467 | +0.538898 | +0.147867 | +12.900542 | N/A |
| polyp / CVC-ClinicDB | -0.277129 | -0.107656 | +0.002518 | -0.433652 | -0.228191 | -0.274611 | -6.305824 | N/A |

## Source-only preservation (fresh host; not post-target forgetting)

| Task | N | A | D2 | O2 | O3 | D4 | L4 | T4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus | 92.442733 | 84.118845 | 84.211643 | 84.233354 | 84.203395 | 84.216906 | 84.222586 | 84.206549 |
| polyp | 97.221359 | 96.709932 | 96.715436 | 96.705380 | 96.706962 | 96.714293 | 96.704334 | 96.702599 |

## Training cost

| Task / arm | Image visits | Outer updates | Seconds | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| fundus / D4 | 600 | 150 | 952.610 | 11244212736 |
| fundus / L4 | 600 | 150 | 901.629 | 14302520832 |
| fundus / T4 | 600 | 150 | 1162.027 | 14300433408 |
| polyp / D4 | 600 | 150 | 370.346 | 7248154624 |
| polyp / L4 | 600 | 150 | 359.046 | 9578470912 |
| polyp / T4 | 600 | 150 | 399.340 | 9578471936 |

## Incremental scoring cost

| Stream | Visits | Host seconds/image | Pipeline seconds | Peak allocated bytes |
| --- | ---: | ---: | ---: | ---: |
| source0_fundus_D4 | 20 | 0.350275 | 11.886 | 4188245504 |
| source0_fundus_L4 | 20 | 0.321836 | 11.482 | 4188245504 |
| source0_fundus_T4 | 20 | 0.320644 | 10.941 | 4188245504 |
| target0_fundus_D4 | 128 | 0.329220 | 72.675 | 4191915520 |
| target0_fundus_L4 | 128 | 0.303585 | 62.175 | 4191915520 |
| target0_fundus_T4 | 128 | 0.344180 | 69.663 | 4191915520 |
| target1_fundus_D4 | 128 | 0.304025 | 65.829 | 4191915520 |
| target1_fundus_L4 | 128 | 0.339973 | 67.098 | 4191915520 |
| target1_fundus_T4 | 128 | 0.331596 | 65.274 | 4191915520 |
| target1_fundus_A | 128 | 0.161616 | 48.279 | 1130003456 |
| target1_fundus_D2 | 128 | 0.343941 | 66.773 | 4191915520 |
| target1_fundus_O2 | 128 | 0.312123 | 62.104 | 4191915520 |
| source0_polyp_D4 | 32 | 0.244350 | 9.899 | 2248487424 |
| source0_polyp_L4 | 32 | 0.244116 | 9.402 | 2248487424 |
| source0_polyp_T4 | 32 | 0.246665 | 9.595 | 2248487424 |
| target0_polyp_D4 | 96 | 0.253922 | 31.565 | 2251526656 |
| target0_polyp_L4 | 96 | 0.243515 | 28.979 | 2251526656 |
| target0_polyp_T4 | 96 | 0.255916 | 30.559 | 2251526656 |
| target1_polyp_D4 | 96 | 0.258953 | 30.981 | 2251526656 |
| target1_polyp_L4 | 96 | 0.250160 | 29.833 | 2251526656 |
| target1_polyp_T4 | 96 | 0.254724 | 30.409 | 2251526656 |
| target1_polyp_A | 96 | 0.173617 | 22.617 | 749360640 |
| target1_polyp_D2 | 96 | 0.249475 | 29.770 | 2251526656 |
| target1_polyp_O2 | 96 | 0.251589 | 30.124 | 2251526656 |

## Evidence and interpretation limits

Successful-attempt update counts including smoke: {"online": 2180, "outer": 906, "inner": 3624, "source_visits": 3600, "records": 2172}.
Cumulative updates including prior failure and repair diagnostics: {"online": 2197, "outer": 922, "inner": 3708}. Prior private bytes: 51406714. GPU-stage time includes those prior attempts when bound in the repair receipt.
GPU-stage wall seconds: 5378.757; private bytes measured at recompute: 559976181. Old records reused for display: 1880 (includes reverse-order stateless N reuse).
The [aggregate](public_aggregate.json) contains OD/OC, all per-domain paired means/medians/signs/worst-decile and worst-single Dice differences, ASSD common-valid/missing counts/adverse tails and empty/full outcomes. ASSD is in pixels; no macro ASSD or zero imputation. The [audit](execution_audit.json) separates native online, outer, functional inner, forwards, memory and incremental runtime.
T4-L4 is the matched derivative comparison. L4-O2 and T4-O2 also change collected history, sequence organization and aggregation frequency; they cannot isolate on-policy effects or cross-step derivatives. Own evolving histories use earlier S versions and four-step truncation, not a fully recomputed on-policy prefix or 120-step BPTT.
Tasks are not pooled. Orders contain the same target contents, not independent patients or independent replicate datasets. Targets were previously exposed; patient/video linkage remains UNKNOWN. Source scores use a fresh source host and do not measure forgetting after target adaptation. Historical timings are not matched hardware/FLOPs comparisons. No target checkpoint, seed, style or domainwise baseline selection.
Completion is separate from scientific merit. Interpret joint comparisons and tails; do not auto-label a positive mean NET_GAIN. No additional M5, style/norm/sampler search or automatic retry is authorized. Raw images/masks, sample identities/logs, synthetic proxies, checkpoints and state histories remain private.
Prior multi-step distillation: [Cazenavette et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Cazenavette_Dataset_Distillation_by_Matching_Training_Trajectories_CVPR_2022_paper.html). This experiment does not claim to invent trajectory distillation.
