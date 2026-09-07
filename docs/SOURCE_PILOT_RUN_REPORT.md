# Source pilot results — SOURCE_PILOT_COMPLETE

Executed immutable code: `1f8f1fab8d7858b89ed2f238d4c48660f48a0265`. One run, physical GPU 7 (RTX 3090), with explicitly authorized co-resident processes. No other process was stopped. The commit containing this report is a later documentation/results delivery, not a different execution snapshot.

Science config: [original source_pilot_v0.json](../configs/source_pilot_v0.json), unchanged SHA256 `e1aeb2454899d024e00934e1f012b5a0fe8cb3bb656f5dd20f73468fdd7a3faf`. [Public execution record](../configs/source_pilot_execution_public.json) separates device/authorization from the disabled historical science file. [Machine results and audit](../results/source_pilot_v1/public_aggregate.json). [Release code contract](SOURCE_PILOT_RELEASE_CONTRACT.md); [173-test local regression](SOURCE_PILOT_RELEASE_LOCAL_REPORT.md).

## Main observation

No Adapt has the highest aggregate Dice in both source development streams. Fixed proxy region supervision changes Adam update magnitudes but provides only a small Fundus macro increment over native VPTTA and an extremely small Polyp increment. Adding the boundary term produces negligible aggregate metric change. This is a descriptive single-run result, not a significance claim or a DD/generalization conclusion. Negative changes are retained.

| Arm | Fundus OD Dice (%) | Fundus OC Dice (%) | Fundus macro Dice (%) | Polyp Dice (%) |
| --- | --- | --- | --- | --- |
| N | 95.583764 | 83.766628 | 89.675196 | 96.968148 |
| A | 95.355611 | 73.147679 | 84.251645 | 96.730906 |
| B | 95.317925 | 73.294037 | 84.305981 | 96.731440 |
| C | 95.318010 | 73.293906 | 84.305958 | 96.731440 |

N = standard source BN / No Adapt; A = native VPTTA; B = A + .1 × proxy region; C = A + .1 × (proxy region + .1 × boundary). Effective boundary coefficient relative to host loss is .01. All adaptation parameters, source coreset selection, query order, transforms and thresholds were frozen.

## Registration, mechanical checks and completion

- Existing source-only manifests, CSVs and seed-0 group splits were reused. Fundus critic_validation has 20 groups; Polyp has 124 and the original first 32 were selected. Pilot RNG/hash-selection seed remains 20260907. Each task uses four distinct basis_train proxy groups; no proxy/query group overlap.
- Actual bytes of 60 selected RGB files, 60 selected masks and both specified source checkpoints matched the pre-existing registrations. No target image/mask contents were read, no split regenerated, no replacement checkpoint selected, and no checkpoint or dataset files copied to a new storage location. Checkpoint training membership remains UNKNOWN.
- Existing remote environment: Python 3.10.6, Torch 2.2.1+cu121, CUDA 12.1, cuDNN 8902. TF32 off, cuDNN benchmark off, cuDNN deterministic on; deterministic algorithms flag false. No environment was installed/upgraded. Seed alone is not proof of GPU determinism.
- GPU smoke: 72 native steps total; each task used 17 direct plus 17 zero-extra wrapped procedural visits, then one K=4 B and one K=4 C step. Both tasks first retrieved at step 17; maximum same-device logit difference was 0.0 under predeclared rtol=1e-4/atol=1e-5. Native snapshots/RNG, clone equality/disjointness, source preservation, counters/Adam and GPU-to-CPU historical memory snapshots passed. Smoke is not included in formal metrics.
- Formal run: Fundus 80 visits per arm; Polyp 128 per arm; **832 visit records, 624 adaptation updates, 8 complete arms**. Registration, smoke and formal exits were all **0**. No failure, restart, resume or score-based selection occurred.
- A separate CPU process reread all private JSONL, validated exact registered order/coverage/channels/lifecycle and recomputed summary values equal to the saved summary. No CUDA was initialized by this verification process. All source-state completion checks passed.
- Private execution outputs before this public export used 1681569 bytes (under 256 MiB). No model/optimizer/prediction/feature snapshots were saved. Sample IDs, true paths, row JSONL and data/model digest lists remain on private NAS; public cohort identifiers are replaced by counts.

## Overall ASSD, empty/full and boundary-defined counts

ASSD is the pooled bidirectional distance between 4-connected foreground pixel surfaces on the final grid, in pixels. Empty prediction or GT makes it undefined. Conditional means can involve different cohorts; do not interpret them as unbiased all-query comparisons. N Fundus OC has one empty prediction, so its ASSD uses 79 visits versus 80 for the other OC arms. Empty/full visits remain in Dice.

| Task | Arm | Channel | Conditional ASSD (px) | ASSD valid/undefined | Pred empty/full | GT empty/full | Boundary defined / all visits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fundus | N | OD | 3.92728 | 80/0 | 0/0 | 0/0 | 80/80 |
| fundus | N | OC | 5.07411 | 79/1 | 1/0 | 0/0 | 80/80 |
| fundus | A | OD | 7.86214 | 80/0 | 0/0 | 0/0 | 80/80 |
| fundus | A | OC | 12.13769 | 80/0 | 0/0 | 0/0 | 80/80 |
| fundus | B | OD | 7.69223 | 80/0 | 0/0 | 0/0 | 80/80 |
| fundus | B | OC | 12.05128 | 80/0 | 0/0 | 0/0 | 80/80 |
| fundus | C | OD | 7.69261 | 80/0 | 0/0 | 0/0 | 80/80 |
| fundus | C | OC | 12.05163 | 80/0 | 0/0 | 0/0 | 80/80 |
| polyp | N | polyp | 0.84150 | 128/0 | 0/0 | 0/0 | 128/128 |
| polyp | A | polyp | 0.89529 | 128/0 | 0/0 | 0/0 | 128/128 |
| polyp | B | polyp | 0.89478 | 128/0 | 0/0 | 0/0 | 128/128 |
| polyp | C | polyp | 0.89478 | 128/0 | 0/0 | 0/0 | 128/128 |


## Paired increments (overall)

Dice deltas below use percentage points; ASSD deltas are pixels on the intersection where both arms are defined (negative ASSD delta is improvement). Private paired cohort identities are retained, and public cohort counts are given here.

| Task | Comparison | Channel | Dice delta (pp) | Paired ASSD delta (px) | ASSD paired count | All visits |
| --- | --- | --- | --- | --- | --- | --- |
| fundus | B-A | OC | 0.14635753 | -0.08640826 | 80 | 80 |
| fundus | B-A | OD | -0.03768514 | -0.16991140 | 80 | 80 |
| fundus | B-A | macro | 0.05433620 | not pooled across OD/OC | — | 80 |
| fundus | C-A | OC | 0.14622639 | -0.08605448 | 80 | 80 |
| fundus | C-A | OD | -0.03760023 | -0.16953251 | 80 | 80 |
| fundus | C-A | macro | 0.05431308 | not pooled across OD/OC | — | 80 |
| fundus | C-B | OC | -0.00013114 | 0.00035378 | 80 | 80 |
| fundus | C-B | OD | 0.00008491 | 0.00037889 | 80 | 80 |
| fundus | C-B | macro | -0.00002311 | not pooled across OD/OC | — | 80 |
| polyp | B-A | polyp | 0.00053346 | -0.00050543 | 128 | 128 |
| polyp | C-A | polyp | 0.00053346 | -0.00050543 | 128 | 128 |
| polyp | C-B | polyp | 0.00000000 | 0.00000000 | 128 | 128 |

Fundus B−A improves OC by 0.14636 pp but reduces OD by 0.03769 pp; the macro increment is 0.05434 pp. C−B Fundus macro is -0.00002311 pp. Polyp B−A is +0.00053346 pp, and C−B aggregate Dice/ASSD is zero. These magnitudes do not justify a boundary-benefit claim. No Adapt remains substantially better on Fundus OC.

## Per-segment Dice and paired deltas

Segments are sequential with persistent adaptation history; they are not independently reset causal comparisons. Each Fundus row averages 20 groups, and each Polyp row 32 groups.

### fundus

| Segment | Arm | OD Dice (%) | OC Dice (%) | Macro Dice (%) |
| --- | --- | --- | --- | --- |
| clean | N | 97.746073 | 87.139392 | 92.442733 |
| clean | A | 95.876437 | 72.361253 | 84.118845 |
| clean | B | 95.872333 | 72.436587 | 84.154460 |
| clean | C | 95.872333 | 72.436587 | 84.154460 |
| gamma_0.7 | N | 95.077897 | 82.460140 | 88.769018 |
| gamma_0.7 | A | 95.567978 | 72.769748 | 84.168863 |
| gamma_0.7 | B | 95.496498 | 72.704353 | 84.100426 |
| gamma_0.7 | C | 95.496266 | 72.704122 | 84.100194 |
| gamma_1.5 | N | 92.416858 | 78.491269 | 85.454063 |
| gamma_1.5 | A | 93.787364 | 72.782183 | 83.284773 |
| gamma_1.5 | B | 93.724602 | 73.157890 | 83.441246 |
| gamma_1.5 | C | 93.724893 | 73.158064 | 83.441478 |
| blur_5_sigma_1 | N | 97.094230 | 86.975710 | 92.034970 |
| blur_5_sigma_1 | A | 96.190662 | 74.677534 | 85.434098 |
| blur_5_sigma_1 | B | 96.178268 | 74.877317 | 85.527793 |
| blur_5_sigma_1 | C | 96.178550 | 74.876850 | 85.527700 |

| Segment | Comparison | Channel | Dice delta (pp) | Paired ASSD delta (px) | Defined pair count |
| --- | --- | --- | --- | --- | --- |
| clean | B-A | OC | 0.07533442 | -0.02067015 | 20 |
| clean | B-A | OD | -0.00410401 | -0.17452740 | 20 |
| clean | C-A | OC | 0.07533442 | -0.02067015 | 20 |
| clean | C-A | OD | -0.00410401 | -0.17452740 | 20 |
| clean | C-B | OC | 0.00000000 | 0.00000000 | 20 |
| clean | C-B | OD | 0.00000000 | 0.00000000 | 20 |
| gamma_0.7 | B-A | OC | -0.06539530 | 0.00894976 | 20 |
| gamma_0.7 | B-A | OD | -0.07148024 | -0.10237211 | 20 |
| gamma_0.7 | C-A | OC | -0.06562652 | 0.00991971 | 20 |
| gamma_0.7 | C-A | OD | -0.07171291 | -0.10074759 | 20 |
| gamma_0.7 | C-B | OC | -0.00023122 | 0.00096995 | 20 |
| gamma_0.7 | C-B | OD | -0.00023267 | 0.00162452 | 20 |
| gamma_1.5 | B-A | OC | 0.37570730 | -0.24902709 | 20 |
| gamma_1.5 | B-A | OD | -0.06276215 | -0.19574838 | 20 |
| gamma_1.5 | C-A | OC | 0.37588154 | -0.24890073 | 20 |
| gamma_1.5 | C-A | OD | -0.06247125 | -0.19585047 | 20 |
| gamma_1.5 | C-B | OC | 0.00017424 | 0.00012636 | 20 |
| gamma_1.5 | C-B | OD | 0.00029089 | -0.00010209 | 20 |
| blur_5_sigma_1 | B-A | OC | 0.19978371 | -0.08488556 | 20 |
| blur_5_sigma_1 | B-A | OD | -0.01239416 | -0.20699771 | 20 |
| blur_5_sigma_1 | C-A | OC | 0.19931614 | -0.08456674 | 20 |
| blur_5_sigma_1 | C-A | OD | -0.01211274 | -0.20700459 | 20 |
| blur_5_sigma_1 | C-B | OC | -0.00046758 | 0.00031882 | 20 |
| blur_5_sigma_1 | C-B | OD | 0.00028142 | -0.00000688 | 20 |

### polyp

| Segment | Arm | Macro Dice (%) |
| --- | --- | --- |
| clean | N | 97.221359 |
| clean | A | 96.709932 |
| clean | B | 96.705011 |
| clean | C | 96.705011 |
| gamma_0.7 | N | 97.128388 |
| gamma_0.7 | A | 96.739974 |
| gamma_0.7 | B | 96.727331 |
| gamma_0.7 | C | 96.727331 |
| gamma_1.5 | N | 97.119412 |
| gamma_1.5 | A | 96.930966 |
| gamma_1.5 | B | 96.931216 |
| gamma_1.5 | C | 96.931216 |
| blur_5_sigma_1 | N | 96.403433 |
| blur_5_sigma_1 | A | 96.542753 |
| blur_5_sigma_1 | B | 96.562202 |
| blur_5_sigma_1 | C | 96.562202 |

| Segment | Comparison | Channel | Dice delta (pp) | Paired ASSD delta (px) | Defined pair count |
| --- | --- | --- | --- | --- | --- |
| clean | B-A | polyp | -0.00492180 | 0.00067001 | 32 |
| clean | C-A | polyp | -0.00492180 | 0.00067001 | 32 |
| clean | C-B | polyp | 0.00000000 | 0.00000000 | 32 |
| gamma_0.7 | B-A | polyp | -0.01264326 | 0.00209593 | 32 |
| gamma_0.7 | C-A | polyp | -0.01264326 | 0.00209593 | 32 |
| gamma_0.7 | C-B | polyp | 0.00000000 | 0.00000000 | 32 |
| gamma_1.5 | B-A | polyp | 0.00024994 | 0.00271191 | 32 |
| gamma_1.5 | C-A | polyp | 0.00024994 | 0.00271191 | 32 |
| gamma_1.5 | C-B | polyp | 0.00000000 | 0.00000000 | 32 |
| blur_5_sigma_1 | B-A | polyp | 0.01944895 | -0.00749958 | 32 |
| blur_5_sigma_1 | C-A | polyp | 0.01944895 | -0.00749958 | 32 |
| blur_5_sigma_1 | C-B | polyp | 0.00000000 | 0.00000000 | 32 |

The machine JSON additionally contains per-segment conditional ASSD means, region/boundary loss, all empty/full and boundary-defined counts and each arm’s valid/undefined counts. No legacy metric is mixed into these columns.

## Observed update norms, time and memory

Norms are mean L2 values per visit. State delta includes memory initialization; optimizer update comes from actual Adam pre/post observations. Pipeline time includes read/transform/adapt/predict/label-distance/evaluation; host-step time includes normalization/device transfer/adapt/predict/push. Neither is pure inference latency. Peak memory is this process’s CUDA **allocated** peak per arm, not whole-device occupancy or reserved memory. Co-resident GPU processes make these contextual runtime observations, not isolated hardware benchmarks.

| Task | Arm | State delta L2 | Adam update L2 | Pipeline mean (ms) | Host-step mean (ms) | Peak allocated (GiB) |
| --- | --- | --- | --- | --- | --- | --- |
| fundus | N | 0.0000000 | 0.0000000 | 288.88 | 87.71 | 0.380 |
| fundus | A | 0.0787657 | 0.1688443 | 484.17 | 264.32 | 1.037 |
| fundus | B | 0.0788869 | 0.3010194 | 669.97 | 462.81 | 3.883 |
| fundus | C | 0.0788821 | 0.3009761 | 669.63 | 468.97 | 3.882 |
| polyp | N | 0.0000000 | 0.0000000 | 80.40 | 27.30 | 0.207 |
| polyp | A | 0.0083243 | 0.0167995 | 258.36 | 192.07 | 0.705 |
| polyp | B | 0.0079630 | 0.0375923 | 398.17 | 328.59 | 2.139 |
| polyp | C | 0.0079622 | 0.0375853 | 387.89 | 318.23 | 2.138 |

The largest formal allocated peak was 3.883 GiB. Maximum per-visit timings and update norms remain in machine JSON. No empty-cache operation was used to lower reported peaks.

## Interpretation and stop

This fixed static raw-source coreset pilot addresses B−A (region supervision), C−B (boundary increment) and C−A (combined effect) only. It did not recover No Adapt performance overall. The result does not rule out all style-conditioned proxies or DD; no such experiment was performed. The source development groups have been used before, checkpoint membership is unknown, 52 content groups are not verified independent patients, and repeated transformations do not increase independent sample count. One seed does not establish significance, unseen generalization, SOTA or clinical performance.

All requested finite stages are complete. No source training, DD, target evaluation, parameter search or background watcher was started. CI is NOT_CONFIGURED. No failure prefix exists for the real run; the N/A+B-prefix retention behavior was tested locally on explicit procedural failures. Stop status: **SOURCE_PILOT_COMPLETE**.
