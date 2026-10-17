# R66: completed negative directional-evidence diagnostic

Retry1 completed96/96 pairs,48 contents,2 historical orders and672/672 branch predictions. The output worker and independent scorer both retired exit0; source state was unchanged and generation read no target labels. This is a fixed output-correction diagnostic on exposed development contents, not a parameter-update or continual-learning result.

## Registered contrasts

| Contrast | Mean Dice pp | Descriptive95% CI | Order0 / Order1 pp |
|---|---:|---|---|
| local_minus_CW | -0.125790 | [-0.278304, 0.030485] | -0.068255 / -0.183325 |
| local_minus_shuffle | 0.067918 | [-0.080449, 0.207793] | 0.087019 / 0.048816 |
| local_minus_random_sign_mean | 0.188463 | [0.045150, 0.338378] | 0.213154 / 0.163773 |

The conjunctive directional gate FAILED. LOCAL-CW is negative in both orders and below the+0.10pp target; its interval crosses zero. LOCAL-SHUFFLE also has an interval crossing zero. LOCAL exceeds the mean of the four same-support, same-absolute-displacement random-sign controls, but this does not establish a useful correction relative to leaving CW unchanged. These observations support neither a new JS integration nor an efficacy retry.

## All domain aggregates (LOCAL-CW, equal orders/channels)

| Domain | Delta pp |
|---|---:|
| Drishti_GS | 0.869653 |
| ORIGA | 0.087885 |
| REFUGE | -0.837716 |
| REFUGE_Valid | -0.622982 |

Full domain/order/channel absolute and paired cells, including every negative cell, accompany this report. Coverage, inward/outward field counts, and edited-pixel means are in RESULTS.json. All nesting veto counts were zero. Spatial permutation changes support and edit area; randomized signs preserve displacement magnitude but rasterized edit areas can differ. Neither comparison alone establishes pure sign causality or clinical utility. Bootstrap intervals resample contents within domains with orders/channels coupled; repeatedly exposed development data and unknown patient grouping limit inference.

## Failure preservation and cost

Attempt1 failed at final file verification before any scoring. Its96 generated pairs and failed retirement receipt remain preserved and were not presented as a completed evaluation. Retry1 fixes only per-image permissions during the same integrity checks. The fixed geometry, seeds, controls and gates were unchanged. See FAILURE_REPORT.md and FAILED_ATTEMPT.json.

Cumulative GPU-worker wall time202.4713561535 seconds, including101.4996743202 failed and100.9716818333 successful seconds. CPU wall cost31.0496503254 seconds includes preparation, both regression processes and scoring. No source training or state replay occurred.

## Disposition

Keep the old CW baseline unchanged; do not promote the local correction. A possible next old-method step is a separately specified error attribution using existing outputs to distinguish systematic boundary bias from useful image-specific correction. It must not relabel post-hoc sign flips or selected domains as prospective validation, and must not launch another radius/layer/strength search. No next scientific run is registered by this report.
