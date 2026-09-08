# M1 method validation — final results

**M1_METHOD_VALIDATION_COMPLETE.** Both tasks completed GM-DD and Ours training (600 episodes each), all five-arm source/target evaluations, and independent CPU reconstruction. The scientific outcome is **mixed**, not a demonstrated general advantage of adaptation-oriented DD.

## Target five-arm main table

Dice values are percentages. Fundus averages OD/OC within each domain, then weights the four domains equally; Polyp weights its three domains equally. Tasks are not pooled.

| Task | No Adapt N | Base A | Real R | GM-DD D | Ours O |
| --- | ---: | ---: | ---: | ---: | ---: |
| fundus | 69.0929 | 76.6063 | 76.5617 | 76.8104 | 76.6601 |
| polyp | 78.4614 | 78.4380 | 78.4387 | 78.3169 | 78.4963 |

| Task | O−D | O−R | O−A | O−N | D−R | R−A | A−N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus | -0.1504 | +0.0984 | +0.0538 | +7.5672 | +0.2488 | -0.0446 | +7.5134 |
| polyp | +0.1794 | +0.0576 | +0.0583 | +0.0349 | -0.1218 | +0.0007 | -0.0234 |

Differences are Dice percentage points computed before rounding. On Fundus, GM-DD is best. Ours is 0.1504 pp below GM-DD, 0.0984 pp above Real and 0.0538 pp above Base. Its 7.5672 pp gain over N is almost entirely already present in Base (+7.5134 pp). On Polyp, Ours ranks first but exceeds N by only 0.0349 pp, Real by 0.0576 pp and GM-DD by 0.1794 pp. The automatic `NET_GAIN` label for Polyp means only positive numerical differences in this run; it is not evidence of a material, significant or stable improvement.

## Per-domain and channel target results

| Task / domain / channel | N | A | R | D | O |
| --- | ---: | ---: | ---: | ---: | ---: |
| fundus / REFUGE / OD | 86.4751 | 87.2511 | 87.2531 | 87.4941 | 87.5175 |
| fundus / REFUGE / OC | 78.8183 | 82.2667 | 82.3528 | 82.4479 | 82.3955 |
| fundus / REFUGE / macro | 82.6467 | 84.7589 | 84.8029 | 84.9710 | 84.9565 |
| fundus / ORIGA / OD | 70.7265 | 79.9100 | 79.7713 | 80.2415 | 79.8158 |
| fundus / ORIGA / OC | 42.8255 | 64.2906 | 64.2337 | 65.2089 | 64.5087 |
| fundus / ORIGA / macro | 56.7760 | 72.1003 | 72.0025 | 72.7252 | 72.1623 |
| fundus / REFUGE_Valid / OD | 73.3552 | 83.4085 | 82.8510 | 82.4921 | 82.4408 |
| fundus / REFUGE_Valid / OC | 51.3198 | 66.0412 | 64.7776 | 61.7615 | 61.3954 |
| fundus / REFUGE_Valid / macro | 62.3375 | 74.7249 | 73.8143 | 72.1268 | 71.9181 |
| fundus / Drishti_GS / OD | 83.4781 | 85.2666 | 85.5890 | 86.1688 | 86.1783 |
| fundus / Drishti_GS / OC | 65.7447 | 64.4156 | 65.6648 | 68.6687 | 69.0285 |
| fundus / Drishti_GS / macro | 74.6114 | 74.8411 | 75.6269 | 77.4187 | 77.6034 |
| polyp / CVC-ClinicDB / polyp | 78.4816 | 70.4921 | 70.2819 | 70.3216 | 70.3735 |
| polyp / ETIS-LaribPolypDB / polyp | 68.8296 | 81.5325 | 81.7458 | 81.5770 | 81.6458 |
| polyp / Kvasir-SEG / polyp | 88.0730 | 83.2893 | 83.2884 | 83.0521 | 83.4696 |

The domain pattern is heterogeneous. Fundus Ours loses 2.8068 pp to Base on REFUGE_Valid while gaining 2.7623 pp on Drishti_GS. Polyp Ours−N is −8.1081 pp on CVC-ClinicDB, +12.8162 pp on ETIS and −4.6034 pp on Kvasir. The small positive Polyp task mean must not be described as consistent benefit across domains.

## Source retention check

| Task | N | A | R | D | O | O−N (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus | 92.4427 | 84.1188 | 84.1545 | 84.2035 | 84.2294 | -8.2133 |
| polyp | 97.2214 | 96.7099 | 96.7050 | 96.7131 | 96.7173 | -0.5041 |

All adaptation arms remain below N on source-clean data. Ours only modestly reduces Base damage there. Source and target results are separate: the source-clean ranking does not imply N is best on shifted targets. These 20/32 source groups were previously used for development; source-checkpoint training membership remains UNKNOWN. Source scores did not select checkpoints or gate target execution.

## Paired tails and ASSD cohorts

The following target O−N summaries retain content-level pairing. Dice differences are pp; the worst decile is the ascending ceil(0.1*n) mean. ASSD is in final-grid pixels, lower is better; pairs require both values defined. Macro OD/OC ASSD is not constructed. All seven planned pair comparisons, means, medians, sign counts and per-domain/channel tails are retained in public_aggregate.json.

| Task / domain / channel | Median Dice Δ | + / 0 / − | Worst decile Δ | ASSD common / undefined pair | Mean ASSD Δ px |
| --- | ---: | --- | ---: | --- | ---: |
| fundus / REFUGE / OD | +0.3560 | 18 / 0 / 14 | -1.8004 | 32 / 0 | -0.1622 |
| fundus / REFUGE / OC | +1.4022 | 17 / 0 / 15 | -6.9881 | 32 / 0 | -0.9852 |
| fundus / ORIGA / OD | +6.7776 | 27 / 0 / 5 | -9.8970 | 32 / 0 | -5.1813 |
| fundus / ORIGA / OC | +16.1524 | 26 / 0 / 6 | -13.1220 | 30 / 2 | -6.6641 |
| fundus / REFUGE_Valid / OD | +7.6485 | 27 / 0 / 5 | -5.1707 | 32 / 0 | -22.4339 |
| fundus / REFUGE_Valid / OC | -1.4982 | 16 / 0 / 16 | -25.0378 | 30 / 2 | +1.5767 |
| fundus / Drishti_GS / OD | +1.7196 | 25 / 0 / 7 | -2.1423 | 32 / 0 | -2.6226 |
| fundus / Drishti_GS / OC | +1.4375 | 18 / 0 / 14 | -13.5994 | 32 / 0 | -1.1045 |
| polyp / CVC-ClinicDB / polyp | -0.6003 | 11 / 1 / 20 | -63.3840 | 28 / 4 | +11.5754 |
| polyp / ETIS-LaribPolypDB / polyp | +0.1230 | 17 / 1 / 14 | -23.0910 | 27 / 5 | -1.2328 |
| polyp / Kvasir-SEG / polyp | -0.2567 | 14 / 0 / 18 | -38.9734 | 31 / 1 | +4.7732 |

Empty/full predictions and labels stay in Dice. Counts and conditional ASSD denominators for every arm/channel/domain are in the public JSON; undefined values are never replaced by zero. ASSD adverse-tail summaries use the increasing-error side.

## Training and cost

| Task / method | Episodes | Train seconds | Peak allocated GiB | Saved final artifact bytes | First 100 mean loss | Last 100 mean loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fundus / D | 600 | 729.46 | 4.275 | 37750928 | 0.972464 | 0.944256 |
| fundus / O | 600 | 766.35 | 4.277 | 37750928 | 0.262490 | 0.259499 |
| polyp / D | 600 | 264.90 | 2.395 | 17844368 | 0.940419 | 0.796549 |
| polyp / O | 600 | 317.74 | 2.505 | 17844368 | 0.187120 | 0.181502 |

Losses from D and O optimize different objectives and are not directly comparable. The six fixed 100-episode blocks, including gradient norms and loss distributions, are provided in the aggregate JSON; no extra iterations or checkpoint selection followed the curves. Final artifact sizes include the image optimizer state, not just proxy pixels.

GPU-stage wall time including smoke, histories, training and scoring: **43.11 minutes**, below the six-hour cap. Private output at delivery is **378,118,228 bytes (360.60 MiB)**, below 2 GiB. Final four artifacts exist with mode 0600 inside the 0700 private directory. Source model files are 90,388,078 bytes (Fundus) and 130,784,857 bytes (Polyp). Full-resolution float32 RGB K=4 pixel payloads alone contain 12,582,912 and 5,947,392 bytes respectively, excluding shared masks and state.

Pipeline/host-step times, per-stage actual forward/backward/retrieval/push counts and allocated peaks are preserved in the JSON. Allocated memory is not reserved or whole-device usage, and shared-GPU timings are not isolated inference benchmarks. D/O both used 600 episodes, not equal FLOPs or equal time.

## Completion evidence and provenance

- Clean execution commit: `d92ed88e603eb68aea97a57493c96a084e98d91c`; delivery commit is the Git commit containing this final report. The execution checkout was not advanced during the experiment.
- Branch: `experiment/m1-adaptation-dd-validation-v1`; fixed dependency: `dbff0d985c6c95345d9fb78f5b1daef57b392564`.
- Registration: 906 selected file identities verified; 40 Fundus / 64 Polyp source training queries; 32 shared Base history visits per task; original K=4 masks/proxies and source validation selection retained.
- Target registration: 224 groups, 32 per domain; original train/test/combined provenance retained; six ETIS candidates were removed by cross-domain image deduplication before selection. No missing domains.
- Full supported CPU regression: 184 tests passed, zero failures/errors/skips, exit 0. Deployment M1 tests: 7 passed, exit 0 (five overlap the full regression; counts must not be added as unique tests).
- Single GPU smoke: 74 real online Adam, 4 image outer Adam, 2 differentiable inner calculations; both tasks passed with nonzero finite image meta-gradients and unchanged native Adam options.
- Formal source scoring: 260 records; target scoring: 1,120 records; total **1,380**. All four final step-600 artifacts were frozen before any scoring.
- Full measured updates including smoke: **1,242 real online Adam**, **2,404 image outer Adam**, **1,202 differentiable inner computations**. Actual stage counts are independently reconstructed.
- Run exit 0; independent CPU recompute exit 0; final status `M1_METHOD_VALIDATION_COMPLETE`. No failure record was found. Finite pipeline has exited; no automatic resume, monitoring or extra experiment was created.

## Interpretation and remaining limits

This run does not establish a clear general advantage of the adaptation-oriented objective over both GM-DD and Real. Fundus favors GM-DD; Polyp favors Ours by small task-level margins with conflicting domain directions. Preserve this mixed result rather than treating either a large gain over harmful source adaptation or tiny mean differences as proof of a new effective method.

This is exploratory development on previously evaluated target domains. Patient/video links are UNKNOWN where absent; counts are content groups, not independent patients. One seed and one domain order do not establish trajectory robustness or statistical significance. Base-history sampling is off-policy and truncated to one step; 600 episodes are a fixed initial budget, not proof of convergence. The GM baseline is prompt-space supervised gradient matching, not a complete reproduction of all dataset-distillation methods.

R retains original source images and masks; D/O retain synthetic image pixels with the same source mask/layout and were trained from source samples. Synthetic medical images can retain source information and are not claimed anonymous. Images, masks, synthetic tensors, optimizer/history states, asset identities, receipts and individual logs remain private on NAS; only source/config/tests and deidentified reproducible results are public. H2 remains NOT_EXECUTED, and M1 made no H normalization modification. No further training, normalization repair, hyperparameter search or new seed was launched.
