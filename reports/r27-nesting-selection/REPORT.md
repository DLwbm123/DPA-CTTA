# R27: nesting-only residual candidate selection

Status: **COMPLETE**. Development signal: **False**.

| Arm | Legacy REVIEW macro Dice (%) | Trajectories |
|---|---:|---:|
| ANCHOR | 78.2244 | 6 |
| RANDOM4 | 78.2272 | 6 |
| GREEDY | 78.1985 | 6 |

All content is development-exposed. The nesting reward was selected after the R26 SEARCH audit; legacy SEALED_REVIEW is not independent validation. Three seeds and two orders reuse the same content. Patient linkage and ROI provenance remain UNKNOWN.

Primary:512px hard Dice, equal domain/channel then order/seed; empty/empty=1. Image-weighted metrics, all fixed paired contrasts and negative cells are retained. Content bootstrap does not resolve patient dependence or multiplicity.64px proposal masks are diagnostic only and remain private.

Wall:9.007h; GPU-worker:22.272h; CPU-worker:0.117h. No cumulative GPU cap. Code:`afd02d24f68e06486939fa95f05971788392381f`.

No policy learning, covariance or DPO is added. RANDOM4 and GREEDY differ only in candidate choice, not candidate generation or committed update count. Nesting alone cannot guarantee anatomical correctness or avoid empty predictions. Final GitHub delivery is separate from execution completion.

## Completion audit and fixed comparisons

All18 online receipts and all18 CPU scorer receipts exited0. Every trajectory
sealed1,951 arrivals (35,118 total). Last online exit preceded label release, which
preceded the first scoring start. Online finished2026-10-08 20:26:12 Asia/Shanghai;
scoring finished20:29:48. No failed attempt was recorded. See COMPLETION_AUDIT.json.

GREEDY minus ANCHOR is -0.025883pp (content-bootstrap interval -0.039135 to
-0.013363pp), with order effects -0.059317 and +0.007551pp and4/6 positive
trajectories. Against RANDOM4 it is -0.028668pp (interval -0.042714 to -0.015197pp),
with order effects -0.064501 and +0.007165pp and2/6 positive trajectories.
The worst seed-averaged domain/channel effects are -0.396290pp and -0.408172pp,
respectively. All fixed comparisons and negative cells are included. These intervals
are descriptive on exposed content, not confirmation of independent efficacy.

Image-weighted deltas are +0.018609pp versus ANCHOR and +0.015910pp versus RANDOM4.
They do not replace the registered equal-domain/channel primary metric. Neither
removing the adverse seed/order nor changing the weighting is an acceptable rescue.
The development gate failed; this is not paper-worthy positive method evidence.

## Reward calibration and next-step decision

The offline SEARCH-only audit used12,204 candidate groups at64px, eight existing
components and uniform averaging among tied reward maxima. It did not train or run
additional online trajectories. In R27, retain is an alias of negative nesting,
so its duplicate audit row is not separate evidence. Full component and domain
aggregates are included. The unchanged RANDOM4 trajectories retain the same small
nesting selection gain as R26 (+0.007809pp). On GREEDY's own trajectories the gain
falls to +0.004642pp, with order0 -0.002039pp and order1 +0.011322pp (4/6 positive).
Other components do not offer consistently positive gains on GREEDY trajectories.
Thus the R26 offline ranking signal did not translate into the registered full
online outcome. State-distribution shift, cumulative updates and64px/512px metric
mismatch are possible explanations, not established causes from this audit.

No R28 training launch is justified by these reward diagnostics alone. Do not
choose another component, seed, threshold or metric simply to obtain significance.
Further expansion of this candidate/reward family is blocked pending a distinct,
mechanistically justified hypothesis. Independent confirmation additionally needs
an appropriate unused cohort and resolved patient/ROI provenance; no such claim is
available from the repeatedly exposed SEARCH/legacy REVIEW content. This decision
preserves the negative evidence and is not a claim that all CTTA approaches fail.

Runtime costs above include profiles and all online/scoring attempts. The lightweight
post-completion CPU audit is outside that frozen runtime ledger; it ran once, with
zero GPU training. Private images, labels, proposal masks, traces, identities,
checkpoints and environment paths are excluded from public delivery.
