# R64 RCA error attribution

Completed the preregistered descriptive diagnosis on all 96 frozen R63 pairs; selected-mask consistency, finite values, synthetic decomposition and aggregate identities passed. No image or label was decoded and no model or selection policy was fitted.

| Decision | Pairs | Mean VPTTA minus CW joint Dice pp |
|---|---:|---:|
| Beneficial switch | 15 | +3.1036 |
| Harmful switch | 18 | -8.1298 |
| Correct hold | 39 | -13.4795 |
| Missed beneficial switch | 24 | +8.3605 |

Harmful switching contributes -1.5243pp across all 96 pairs; beneficial switching contributes +0.4849pp, yielding R63's -1.0394pp. Missed beneficial switches contribute 2.0901pp oracle regret and harmful switches 1.5243pp, totaling 3.6145pp. Thus accuracy alone hides asymmetric error costs.

The pooled signed RCA margin/true advantage Pearson correlation is 0.1042. Mean absolute RCA margin is 0.08154 for correct choices and 0.10083 for incorrect choices. This does not prove every threshold fails, but gives no justification for treating larger RCA margin as calibrated certainty; no threshold search was conducted. Per-domain/order correlations and all mistake groups are in RESULTS.json. These small, dependent strata are descriptive, not significance tests.

Order0 Drishti and ORIGA account for 110.7883 of 146.3356 summed harmful-switch pp (75.71%). REFUGE_Valid has 14 missed improvements, contributing 159.5412 of 200.6532 missed-switch pp (79.51%). Both failure types matter; a global conservative switch rule could suppress opportunities as well as mistakes.

Engineering history: attempt1 failed with KeyError domain because the image-only manifest omits domain names. It produced no result. Attempt2 reads domain metadata from the existing scoring panel instead, with unchanged statistics and no image/label decoding; exit0. Both SSH calls together took under one second as measured by the caller, within the 60-second budget; successful analysis body was 0.00811 seconds. Failed-attempt process-only time was not separately instrumented. No GPU or persistent remote output was created. The original failure is preserved in ATTEMPT1.json.

Decision: close this RCA configuration and avoid unsupported confidence-threshold tuning. Next investigation should seek evidence for candidate-conditioned feedback that retains spatial/anatomical correspondence, which R63's raw-color/local-moment forest does not explicitly model. This is a hypothesis for literature and existing-code review, not an approved new algorithm or an efficacy claim. A concrete distinct intervention must receive its own finite registration before execution. The same eight-source-reference scope remains binding; no source expansion.

All data are exposed-development diagnostics. No independent validation or causal attribution to the reference representation is established.
