# R49: C-background LSO x shared sensitivity

Status: **COMPLETE**.
Primary: C_JS_SHARED minus C. J is fixed R46 SHARED, without R48 descent constraint.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.691553 | 78.082091 |
| C_LSO | 78.882184 | 78.261537 |
| C_JS_SHARED | 78.677747 | 78.064395 |
| C_LSO_JS_SHARED | 78.849346 | 78.225666 |
| CW_LSO | 79.045619 | 78.412987 |

Frozen standalone gate passed: False. Interpretation: NO_REGISTERED_STANDALONE_SIGNAL.
The primary requires >=0.1pp, CI lower>0, both orders positive, >=5/6 positive trajectories, imageweighted>=0, worst seed-mean cell>=-2pp.
Interaction I=[(C+LSO+J)-(C+LSO)]-[(C+J)-C]. All four arms are coupled within content during bootstrap. A negative interaction is not proof of identical internal function.
CW_LSO stays unchanged; both J arms are compared to it. No secondary arm can replace the primary.
GPU-worker 8.875580/48h; CPU-worker 0.197419h. All attempts retained; T0 unchanged.
Exactly five conditions, three seeds, two orders, 1951 arrivals: 30 fresh trajectories/58530 CPU scores. No source images, online labels, retuning, scientific/technical retry or next round.
SEARCH/REVIEW were historically exposed; REVIEW descriptive only. Patient linkage UNKNOWN. ASSD/Brier/soft Dice uncomputed; no anatomical/clinical or independent generalization claim.
Only own source/protocol and anonymous all-positive/all-negative aggregates are public. Images, labels, content identifiers/hashes, predictions, model states and raw/private logs remain private.

## Paired SEARCH evidence and frozen decision

All changes below are percentage points of macro Dice, with paired 95% content-cluster bootstrap intervals. Secondary intervals are descriptive and not adjusted for multiple comparisons.

| Contrast | Mean pp | 95% CI pp | Order 0 / 1 pp | Positive trajectories | Imageweighted pp | Worst seed-mean cell pp |
|---|---:|---|---|---:|---:|---:|
| C_JS_SHARED − C | -0.013807 | [-0.036459, +0.012416] | -0.083924 / +0.056311 | 3/6 | +0.102742 | -0.825766 |
| C_LSO − C | +0.190631 | [+0.159716, +0.221283] | +0.113402 / +0.267861 | 6/6 | +0.396541 | -1.102970 |
| C_LSO_JS_SHARED − C_LSO | -0.032839 | [-0.053925, -0.008722] | -0.083279 / +0.017601 | 3/6 | +0.076506 | -0.785225 |
| C_LSO_JS_SHARED − C_JS_SHARED | +0.171599 | [+0.141714, +0.201198] | +0.114047 / +0.229151 | 6/6 | +0.370305 | -1.131684 |
| C_JS_SHARED − CW_LSO | -0.367873 | [-0.434359, -0.294945] | -0.273035 / -0.462710 | 0/6 | -0.355462 | -2.335734 |
| C_LSO_JS_SHARED − CW_LSO | -0.196273 | [-0.252150, -0.129626] | -0.158988 / -0.233559 | 0/6 | +0.014842 | -2.040470 |
| CW_LSO − C | +0.354066 | [+0.303809, +0.402994] | +0.189111 / +0.519021 | 6/6 | +0.458205 | -0.788639 |
| J_EFFECT_WITH_LSO − J_EFFECT_WITHOUT_LSO | -0.019032 | [-0.022713, -0.015495] | +0.000645 / -0.038709 | 1/6 | -0.026236 | -0.104596 |

J on C misses the registered mean, positive-CI, both-order and 5/6-trajectory criteria. It passes the imageweighted and worst-cell components, which cannot substitute for the failed criteria. The registered outcome is **NO_REGISTERED_STANDALONE_SIGNAL**.

LSO improves C by +0.190631pp and improves C+J by +0.171599pp. J itself has no registered standalone benefit; adding it to C+LSO lowers the macro mean by −0.032839pp (CI entirely below zero). The interaction is −0.019032pp, also below zero. This describes a more negative marginal J effect with LSO, but the preregistered standalone prerequisite for interference/redundancy interpretations fails. It does not support complementarity, does not establish internal functional overlap, and cannot attribute J failure to an overly strong LSO baseline. The conditional interval also extends below −0.05pp, failing the registered equivalence interval for redundancy. These results concern the C background only, not an LSO×J factorial under W.

CW_LSO remains the best of these five arms. Both J arms lose to it on all six macro trajectories: C+J by −0.367873pp and C+LSO+J by −0.196273pp. Neither J arm replaces the primary or the strong control.

## Local improvements and costs

These domain/channel values average the original seeds and both orders. The full per-seed/per-order positive, zero and negative cells are in ALL_PAIRED_DOMAIN_ORDER_CELLS.csv; all originally exported negative cells remain unchanged.

| Domain | Channel | J on C pp | J on C+LSO pp |
|---|---|---:|---:|
| REFUGE | OD | +0.072881 | +0.026194 |
| REFUGE | OC | +0.200035 | +0.177601 |
| REFUGE_Valid | OD | +0.367523 | +0.316953 |
| REFUGE_Valid | OC | +0.457606 | +0.430919 |
| Drishti_GS | OD | -0.185954 | -0.200486 |
| Drishti_GS | OC | -0.466647 | -0.442229 |
| ORIGA | OD | -0.191111 | -0.209799 |
| ORIGA | OC | -0.364785 | -0.361861 |

REFUGE and REFUGE_Valid improve, while Drishti_GS and ORIGA lose in both OD and OC. The strongest local gain without LSO is REFUGE_Valid OC +0.457606pp; the strongest mean domain/channel loss is Drishti_GS OC −0.466647pp. Order-specific losses are larger: the worst J-on-C seed-mean cell is order0 Drishti_GS OC −0.825766pp, and the corresponding J-on-LSO worst cell is −0.785225pp.

J on C has 10/16 negative seed-mean domain/channel/order cells; J on C+LSO has 9/16. Against CW_LSO, each J arm has 11/16 negative cells. Their worst cells are order0 Drishti_GS OC −2.335734pp and −2.040470pp, respectively. Gains in the larger REFUGE_Valid portion make imageweighted J changes positive (+0.102742pp without LSO, +0.076506pp with LSO), but equal-domain/channel macro changes remain negative. The frozen metric is not switched to imageweighted Dice.

## Mechanical action, missing diagnostics and compute

All quantities below are SEARCH means, not evidence of causal anatomical protection.

| Diagnostic | C+J | C+LSO+J |
|---|---:|---:|
| joint.rank | 4 | 4 |
| joint.acquisition_capture_fraction | 1 | 1 |
| joint.anatomy_capture_fraction | 1 | 1 |
| joint.proposal_span_energy_fraction | 0.0489315781 | 0.0513792194 |
| joint.relative_metric_update_change | 0.165069275 | 0.169621548 |
| joint.proposal_metric_norm | 0.000130435364 | 0.000142216754 |
| joint.filtered_metric_norm | 0.000127889055 | 0.000139306745 |
| structure_evidence_mean.0 | 0.00938920041 | 0.00932448332 |
| structure_evidence_mean.1 | 0.00671501894 | 0.00675262306 |
| BN_gradient_norm | 0.30856657 | 0.369136191 |
| BN_update_norm | 0.00315444552 | 0.00313359981 |
| pseudo_loss | 0.0596740934 | 0.0581757512 |

The mean filter spectra are approximately [0.015210, 0.042698, 1, 1] without LSO and [0.015071, 0.043076, 1, 1] with LSO. Rank4 and capture1 are expected from the constructed shared span; they establish mechanical coverage of the supplied probes, not validity of the anatomical proxy. Same-step metric filtering is active, yet Dice does not improve on the registered macro endpoint. The OD/OC evidence means are proxy values, not calibrated probabilities.

C_LSO has mean base loss 0.057930040, LSO loss 0.070528193; C_LSO_JS_SHARED has 0.058175751 and 0.071401842. The LSO coefficient remains 0.1. C/C_LSO/C+J/C+LSO+J have no existing regional logit-gradient hooks in this run; these diagnostics are missing, not zero. CW_LSO retains its six OD/OC foreground/boundary/background measurements. Spearman-Dice-change fields are unavailable because the inherited diagnostic baseline G is absent; no association is fabricated.

| Condition | Observed seconds/arrival | Forward | Main backward + auxiliary VJP | Adam |
|---|---:|---:|---|---:|
| C | 0.369988 | 8 | 1 + 0 | 1 |
| C_LSO | 0.502521 | 8 | 1 + 0 | 1 |
| C_JS_SHARED | 0.667083 | 10 | 1 + 4 | 1 |
| C_LSO_JS_SHARED | 0.663170 | 10 | 1 + 4 | 1 |
| CW_LSO | 0.519429 | 8 | 1 + 0 | 1 |

Actual online receipts confirm 10F / (1 main backward + 4 auxiliary VJPs) / 1 Adam for J and 8F / 1B / 1 Adam without J. End-to-end observed J elapsed overhead is about 80.3% over C and 32.0% over C+LSO; concurrent GPU work and initialization mean these are observed run costs, not isolated kernel benchmarks. Same-round profile peak reserved memory is 896MiB for J, 622MiB for C, 624MiB for C_LSO/CW_LSO.

## Verified completion, exposure and accounting

An independent recount confirmed 30 unique online jobs, 30 independent CPU scorers, 1,951 rows per scorer and 58,530 total rows; all 65 attempts (5 profiles + 30 online + 30 CPU scoring) completed with attempt0. All online workers retired before the first scorer; online/scorer/watch/supervisor processes are now retired. See VERIFIED_COMPLETION_AUDIT.json.

Each stream has 256 CONTEXT rows, 1,017 SEARCH rows and 678 REVIEW rows. The full 1,951 rows are scored for coverage; the macro efficacy tables exclude CONTEXT and use the 1,695 remaining development images. Repeated seeds and orders are coupled in the content-cluster bootstrap, not treated as independent new images.

GPU-worker cost is 31,952.088110s = 8.87558003048102h, including same-round profiles and initialization; CPU-scoring workers total 710.708923s = 0.197419h. The previous 13 rounds cost 108.22563222454656h, so 14 closed rounds total 117.10121225502758h. Including the earlier isolated profiles (668.9714405536652s), cumulative GPU-worker cost is 117.28703765518138h. Original T0 and old ledgers are unchanged.

Preparation accounting remains separate: server CPU checks took 24.458931s and local checks 11.823311s. The retained pre-T0 NAS copytree/copy2 EIO has unknown CPU elapsed time and zero GPU use; content-only preparation succeeded. Unknown failed-preparation CPU cost is not silently recorded as zero.

REVIEW means are descriptive: J−C −0.017696pp, J-on-LSO −0.035871pp and interaction −0.018174pp; full intervals and negative cells are retained. Neither SEARCH nor legacy REVIEW is an independent unseen test set. Patient dependence, source crop-center provenance, anatomical validity, ASSD/Brier/soft Dice and external-dataset generalization remain unresolved. No anatomical or clinical protection claim follows from these results.

No new experiment, retry, tuning, candidate selection or next round is launched. The authorized monitor is removed only after the GitHub commit and anonymous report access are verified.
