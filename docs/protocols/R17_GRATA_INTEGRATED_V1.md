# R17: integrate source correction into GraTa adaptation

Frozen before new source qualification or target access. User authorized this one experiment on 2026-10-04. Parent release: `72fae785018c4b441e8311426858ed289beef4f4`.

## Hypothesis and comparisons

A source-supervised contextual residual supplies useful local correction directions inside the target consistency backward. GraTa gradient alignment and dynamic learning rate can turn those directions into useful BN-affine adaptation. Predictions come from the updated model itself.

New conditions: `C_CONTEXT` (ordinary native Adam consistency plus contextual residual), `G_CONTEXT` (native GraTa plus that identical head), `G_LOGIT` (native GraTa plus equal-capacity logits-only head). Reuse exactly sealed, paired R16 scalar references `C0`, `DS`, `M_STATIC` (R16 D_CONTEXT), and intact stateful `G`. No rerun of historical inference. Primary comparison is G_CONTEXT versus G; G_CONTEXT versus C_CONTEXT separates the update rule, and versus G_LOGIT measures context contribution. M_STATIC is a system-level reference using an older source-trained head, not a head-matched update ablation.

Both orders contain the same 1951 images, 1695 primary remaining-development observations, four target domains, OD/OC. Six new physical trajectories: 11706 arrivals and 10170 primary observations. All methods are batch one, resolution 512, checkpoint/preprocessing/BN policy registered in the inherited bindings. Native augmentation seed 20260907 matches the historical G reference; source simulation/head seed is 20261004. These are not independent seeds, patients or held-out validation.

## Exact integration

For the official six weak views, capture original logits and up3 features, plus inverse horizontal-flip logits (official weak factor 0). Use the same current-image DS baseline, radius-6 boundary and reliable-seed exclusion as R16. Head input has 269 channels on 128x128; use the same float16 rounding and fit-only normalization. Each 9026-parameter residual head predicts a bounded logit correction `0.25*tanh(r)` on the editable support M.

The official strong-view backward uses

`BCE(z_strong, q_six_views) + 0.1 * mean_M[BCE(z_strong,q_head) - BCE(z_strong,q_DS)]`.

All teachers/features for the supplemental target are stopped. An empty mask contributes zero; a zero residual gives exactly the native objective and gradient. No extra backbone forward is added. Preserve the published native asymmetric entropy auxiliary, temporary subtractive entropy perturbation, gradient cosine, restoration and `base_lr*(cos+1)^2/4` rule. In G the teacher is captured in the temporarily perturbed state; in C it is unperturbed. The head and non-BN backbone parameters remain frozen online. BN affine is intentionally updated; running-stat policy remains native. No final-output correction head, memory, RL, extra adaptation step, target-label selection or tuning.

## Source roles and choice

Existing disjoint source fit and validation folds, frozen simulator and episodes 0/16/32/48 (four modes), each 32 visits, for both C and G: 256 fit and 256 validation captures. Every episode/fold/update starts from the registered checkpoint; adaptation is continuous within its episode. Only fit captures estimate feature statistics. Report actual distinct source groups used, not nominal fold sizes.

Both heads share initialization and batches (seed 20261004), AdamW lr .001, weight decay .0001, batch 8, 2000 steps. Existing balanced BCE + soft Dice + .1 originally-correct probability-change source loss. D_LOGIT masks channels 8 onward after normalization, retaining identical network capacity. Select each head among steps 500/1000/1500/2000 using mean source validation hard Dice, balanced C/G and four modes; ties choose earlier. Preserve every checkpoint result, including soft OD/OC losses.

Source captures use plain native C/G trajectories without learned-head feedback. This improves adapted-feature coverage but does not prove on-policy target distribution matching. After selection, report a descriptive selected-head closed-loop check: all three conditions, four episodes, first eight visits = 96 visits. No performance gate or parameter changes follow this check; only nonfinite/invariant/resource failure blocks target.

## Qualification and resource admission

Before training, native source parity uses four fixed source-validation images per C/G condition, paired wrapped weight-zero versus original hosts. Require exact logits, BN, Adam state and RNG equality. CPU tests check zero correction, empty support, analytic gradient and equal-capacity ablation. A four-step-per-head mechanical fitting profile and one active source update per C/G exercise the nonzero correction (qualification heads discarded). The active C/G hosts also verify exact next-image logits, BN and RNG after snapshot/restore (four additional source accesses). Total qualification image accesses: 22. No target access or selection.

Cost projection uses maximum measured native/active per-image time plus .12 source/.15 target seconds overhead, maximum measured paired training step plus .08 disk seconds, 600 source fixed seconds and 60 target setup seconds per job, all multiplied by 1.3. Source projection includes 608 trajectory visits; training 2000 paired steps. Require source <=10800, target <=21600, total normal GPU-worker <=36000 seconds and finish before normal deadline. If FULL does not fit, stop with NOT_RUN_BUDGET; no shorter run, arm deletion or score-driven adjustment.

T0 is conservatively fixed at 2026-10-04 17:00 Beijing, before implementation. Normal compute ends Oct 5 16:00; hard compute 16:30; absolute exit 17:00. GPU-worker cap 43200 seconds (36000 normal +7200 recovery), preflight <=3600, source <=10800, target <=21600; peak private artifacts/cache <=8GiB. Source feature cache retired only after retaining selected heads, fit statistics and receipts. Historical R16 costs remain separate. GPU pool physical 4/5/6/7, maximum two concurrent workers, actual free VRAM >=5GiB and NAS mount/capacity/write-read check. Neutral main/child argv, isolated leases and timeout watchdog.

All failures, preflight and restoration costs count once by phase/attempt; live estimate is replaced by terminal cost. One evidenced equivalent infrastructure recovery per physical target job from exact adapter/Adam/RNG/cursor/prefix snapshot. No restart for negative performance; no source qualification or scientific rerun without an evidenced correction recorded against original T0/cost.

## Scoring and decision

Do not read new target scores until all six physical trajectories are terminal and sealed/retired. Only independent CPU scorer reads labels. Check exact content/order/role/denominator matching, immutable head/non-BN parameters and native operation counts (C 8F/1BP/1Adam, G 9F/2BP/1Adam per image), receipts and ledger before review. Compute per-image hard Dice and ASSD, report macro over eight domain/channel cells, both orders, all conditions, matched per-image distributions, TP/FP/FN, ASSD valid/undefined counts and structural metrics.

Predeclared priority signal: G_CONTEXT mean over orders >=+0.5 percentage points relative to matched G with both orders positive. Any domain/channel loss >2 points versus G marks risk. Mechanism contrasts do not lower this threshold. Single stochastic initialization, two same-content orders, unknown patient dependence and prior development exposure prohibit an independent-confirmation claim. Target soft Dice/Brier and paired pixel transition statistics versus historical GraTa are NA when only masks/scalars exist, never fabricated.

After completion or honest terminal blocking, publish sanitized source, config, receipts, all result tables and report on this research branch; verify remote SHA and anonymous report access through the required proxy. Keep raw data, labels, identifiers, per-image predictions/features, model weights, paths and credentials private. Hourly monitoring remains quiet for healthy unchanged work, repairs evidenced faults only, and ends after delivery. No automatic successor.
