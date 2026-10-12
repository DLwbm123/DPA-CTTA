# R46: shared acquisition/anatomical sensitivity span

Status: **COMPLETE**. Frozen primary: CW_LSO_JS_SHARED.

Only replace acquisition-only Q with orth([X/||X||,Y/||Y||]), maxrank4. Same CW_LSO loss, two acquisition probes, four VJPs and actual Adam proposal, strength1/ridge0.01. Anatomy capture may suppress useful correction; no protection claim. Original JS remains a control.

| Condition | SEARCH macro Dice (%) | REVIEW macro Dice (%) |
|---|---:|---:|
| C | 78.6916 | 78.0821 |
| W | 78.0896 | 77.5324 |
| CW_LSO | 79.0456 | 78.4130 |
| CW_LSO_JS | 79.0387 | 78.4065 |
| CW_LSO_JS_SHARED | 79.0205 | 78.3859 |

Primary SEARCH versus C/W each>=0.3pp and versus CW_LSO/original JS each>=0.1pp. Every comparison requires both orders positive,>=5/6 trajectories positive,imageweighted>=0,worst seed-mean cell>=-2pp. Frozen primary passed: False. All negative cells remain reported.

GPU-worker 12.131/48h; CPU-worker 0.251h. Original per-round T0 and attempts retained.

Exactly five frozen conditions, three seeds, two orders, 1,951 arrivals each. No online labels, source images, source retraining, RL, output fusion or scientific retries. Only the registered two acquisition probes are added. Separate CPU masks only after all online workers retire.

SEARCH and legacy REVIEW were historically exposed. SEARCH guided this follow-up; REVIEW is descriptive only. Gains do not establish independent generalization. Dice, empty-mask convention and content-cluster bootstrap are unchanged; patient linkage is UNKNOWN. ASSD, Brier and soft Dice were not computed.

Only own code and anonymous aggregates/costs/audits are public. Stream, prediction, trace and scalar provenance hashes are omitted from public receipts; originals remain private.

## Main conclusion

The shared span mechanically captures both registered sensitivity blocks and improves several severe adverse cells, but does not improve the prespecified macro Dice objective over CW_LSO or original JS. This is an observed tradeoff, not a successful new module or proof of anatomical protection. The frozen development gate failed; no next round is authorized.

| SEARCH comparison | Macro delta (pp) | Content-bootstrap 95% interval (pp) | Positive trajectories | Order0 / Order1 (pp) | Imageweighted delta (pp) |
|---|---:|---:|---:|---:|---:|
| SHARED minus C | +0.328930 | [+0.290796, +0.367065] | 6/6 | +0.119995 / +0.537864 | +0.546601 |
| SHARED minus W | +0.930841 | [+0.719939, +1.145293] | 6/6 | +1.199996 / +0.661687 | +0.397151 |
| SHARED minus CW_LSO | -0.025136 | [-0.049302, +0.002837] | 3/6 | -0.069116 / +0.018843 | +0.088396 |
| SHARED minus original JS | -0.018238 | [-0.041453, +0.008933] | 3/6 | -0.061779 / +0.025303 | +0.094575 |

Both incremental intervals include zero; neither establishes a reliable average improvement or a statistically resolved average loss. Positive imageweighted changes coexist with negative macro changes because the objectives weight domains differently. C comparison satisfies all its frozen criteria. W comparison fails the worst-cell criterion. Both attribution comparisons fail the >=0.1pp threshold, both-orders-positive requirement and >=5/6 positive-trajectory requirement. Primary is retained and gates are not relaxed.

## Adverse cells: improvements and costs

Seed-mean SEARCH, same domain/channel/order cells, deltas against W:

| Cell | CW_LSO minus W (pp) | Original JS minus W (pp) | SHARED minus W (pp) | SHARED minus CW_LSO (pp) |
|---|---:|---:|---:|---:|
| Order0 REFUGE_Valid OC | -5.155135 | -5.138166 | -4.468401 | +0.686734 |
| Order0 REFUGE_Valid OD | -4.908927 | -4.896469 | -4.469088 | +0.439840 |
| Order1 REFUGE OC | -3.528396 | -3.525033 | -3.074185 | +0.454210 |
| Order1 REFUGE_Valid OD | -2.081046 | -2.089187 | -1.908503 | +0.172543 |

The overall worst W-relative cell changes from -5.155135 to -4.469088pp, a +0.686047pp improvement in the minimum across cells. The identity of that minimum switches from REFUGE_Valid OC to OD (both order0); the matched OC improvement is +0.686734pp, not the minimum-to-minimum difference. Original JS worst is -5.138166pp, so the corresponding minimum improves +0.669078pp. Three SHARED/W seed-mean cells remain below -2pp. Worst single-seed W cell is -5.292900pp.

New or larger costs versus CW_LSO include order0 Drishti_GS OC -0.861706pp, order1 ORIGA OC -0.363370pp, order0 Drishti_GS OD -0.330793pp and order0 ORIGA OC -0.310961pp. Against original JS, the worst cell is also order0 Drishti_GS OC (-0.848594pp). Worst C-relative deficit improves from CW_LSO -0.788639pp to SHARED -0.500398pp, but its location changes to order0 Drishti_GS OD. SEARCH_CELL_COMPARISONS includes all positive and negative seed-mean cells; ALL_NEGATIVE_CELLS preserves individual-seed as well as seed-mean deficits.

REVIEW is descriptive only: SHARED macro Dice 78.3859%, versus CW_LSO 78.4130% and original JS 78.4065%. It offers no independent confirmation and cannot guide tuning.

## Mechanism diagnostics

Across 6,102 SEARCH arrivals, original JS has rank2 and SHARED has rank4 at every step. SHARED acquisition/anatomy capture fractions are both approximately1.0 to floating-point tolerance, as expected from the chosen span construction; this is capture of the registered VJPs, not proof of capturing true anatomy.

SHARED mean eigenvalues are [0.015126, 0.042746, 1.0, 1.0], versus original JS [0.906365, 0.993027]. Its proposal energy fraction inside the span is mean5.3313%, median3.3968%, P05/P95 0.3870%/17.0365%. Relative metric update change is mean17.4989%, median15.7322%, P05/P95 3.9647%/36.5972%. This difference-vector measure must not be confused with whole-vector norm shrinkage.

Mean per-step filtered/proposed metric norm ratio is0.978109 (about2.1891% norm shrinkage) for SHARED versus0.998812 (about0.1188%) for original JS. No contraction violations above relative tolerance1e-6 were observed. SHARED evidence mean OD/OC is0.009455/0.006864 versus original JS0.009551/0.006947. Full distributions are in JOINT_MECHANISM_SUMMARY; base/LSO losses, BN and region gradients are in MECHANISM_DIAGNOSTICS.

The basis change demonstrably acts on the update and correlates with redistributed cell performance. It does not establish that missed anatomical directions caused R45's failure or that contraction causally protects anatomy. Both registered blocks are rank<=2; representing their union increases rank to4, but two shared-filter eigenvalues near1 preserve directions rather than making every represented direction strongly constrained. The proxy, domain tradeoff and fixed ridge remain possible limitations; no post-hoc tuning or additional run is performed.

## Completion and costs

All65 attempts completed:5 profiles,30 online trajectories and30 independent CPU scores. All30 unique jobs have1,951 score rows (58,530 total). Last online end1791817422.430897 precedes first scoring start1791817424.716101. Watch, supervisor, online and scoring processes are retired. No failures/retries or GPU2 use in R46. Original T0 and costs are retained.

GPU-worker cost12.130988740656111h, CPU scoring903.6620998382568s, preparation CPU17.094215631484985s (no preparation GPU/failures); wall25,017.678455114365s, approximately6.9494h. Prior experiments plus the isolated JS profile and R46 total91.70541420506108 GPU-worker h. Timing reflects shared GPUs and is not a controlled speed benchmark. Same-round peak reserved memory is939,524,096 bytes for either JS condition and654,311,424 bytes for CW_LSO; network calls remain10F/4 auxiliary VJP/1 loss backward/1 Adam for JS variants and8F/1 loss backward/1 Adam for CW_LSO. See COST and PROFILE_ADMISSION for all timings.

The main positive development evidence remains CW_LSO. R46 supplies a useful negative result and local safety/accuracy tradeoff, not an independently validated algorithmic improvement. ASSD/Brier/soft Dice, clinical safety and patient-level generalization remain untested.
