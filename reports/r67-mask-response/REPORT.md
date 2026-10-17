# R67: candidate-mask response diagnostic

**Complete; preregistered response gate failed.** Fixed image features reduced the average perturbation spread ratio from the mask-only control's 1.000000 to 0.972755, a difference of -0.027245 (95% content-stratified bootstrap interval [-0.033763, -0.020923]). This misses the required difference of at most -0.10. It is a label-free mechanism diagnostic, not a segmentation efficacy test.

## Coverage and controls

All 48 development images, two archived orders, 96 pairs and 192 channel pairs were retained. Every denominator was defined. The worker performed 49 frozen source feature forwards and 576 native proposal calls, with zero target-label reads and zero model updates. Frozen state remained unchanged. Worker and aggregation processes retired with exit code zero; see the completion audit and resource ledger.

The same image features were held fixed while each candidate mask was eroded, unchanged or dilated using the registered 3x3 operation. IMAGE used the frozen up3 features; MASK_ONLY used candidate-derived one-hot background/rim/cup features. Both used the same proposal implementation. Spread is the XOR area between the two perturbed outputs divided by that between the perturbed inputs. Identity/abstention gives one. Lower spread does not establish proximity to the true boundary.

## Registered decision

| Criterion | Observed | Pass |
|---|---:|---|
| All 192 channel-pair denominators defined | 192/192 | Yes |
| Mean IMAGE minus MASK_ONLY <= -0.10 | -0.027245 | **No** |
| Bootstrap upper bound < 0 | -0.020923 | Yes |
| Both order differences < 0 | -0.034039 / -0.020450 | Yes |
| Minimum shift/channel mean acceptance >= 0.25 | 0.570860 | Yes |

The acceptance criterion averages across domains and orders for each shift/channel; it is not a minimum individual-cell acceptance claim. The final conjunction is false.

## All domain/order/channel results

| Order | Domain | Channel | Contents | IMAGE ratio | MASK_ONLY ratio | Difference |
|---|---|---|---:|---:|---:|---:|
| 0 | Drishti_GS | OD | 12 | 0.911449 | 1.000000 | -0.088551 |
| 0 | Drishti_GS | OC | 12 | 1.004252 | 1.000000 | +0.004252 |
| 0 | ORIGA | OD | 12 | 0.952564 | 1.000000 | -0.047436 |
| 0 | ORIGA | OC | 12 | 0.992177 | 1.000000 | -0.007823 |
| 0 | REFUGE | OD | 12 | 0.838703 | 1.000000 | -0.161297 |
| 0 | REFUGE | OC | 12 | 1.036338 | 1.000000 | +0.036338 |
| 0 | REFUGE_Valid | OD | 12 | 0.945816 | 1.000000 | -0.054184 |
| 0 | REFUGE_Valid | OC | 12 | 1.046385 | 1.000000 | +0.046385 |
| 1 | Drishti_GS | OD | 12 | 0.966459 | 1.000000 | -0.033541 |
| 1 | Drishti_GS | OC | 12 | 1.017069 | 1.000000 | +0.017069 |
| 1 | ORIGA | OD | 12 | 0.971597 | 1.000000 | -0.028403 |
| 1 | ORIGA | OC | 12 | 1.001243 | 1.000000 | +0.001243 |
| 1 | REFUGE | OD | 12 | 0.932077 | 1.000000 | -0.067923 |
| 1 | REFUGE | OC | 12 | 1.027603 | 1.000000 | +0.027603 |
| 1 | REFUGE_Valid | OD | 12 | 0.872046 | 1.000000 | -0.127954 |
| 1 | REFUGE_Valid | OC | 12 | 1.048307 | 1.000000 | +0.048307 |

Full acceptance rates for all three perturbations are preserved in [ALL_CELLS.csv](ALL_CELLS.csv). Equal-cell OD mean is 0.923839 (7.616% spread contraction); OC mean is 1.021672 (2.167% spread amplification). These channel summaries are descriptive, not replacement endpoints or grounds for channel-specific deployment.

## Interpretation and decision

There is a small aggregate response to image information relative to the candidate-mask-only construction, but the response remains close to following the input mask. Cup spread increases on average. The registered test therefore does not support sufficient image anchoring for this signal. It does not prove that all frozen features lack useful boundary information, nor isolate which part of the prototype/crossing/reconstruction pipeline limits it.

Do not integrate this signal into the joint-sensitivity objective, select domains/channels after seeing these results, invert its sign, or scan perturbation radii. R66's negative Dice result remains unchanged. The mechanical straight-boundary fixture passed; that is implementation evidence, not real-image effectiveness. Existing SEARCH exposure and repeated images across orders preclude independent-generalization or independent-patient claims.

## Cost and delivery

GPU worker time: 40.603661 seconds. CPU total: 19.005909 seconds, including 17.052547 seconds of retained fixture/setup work and 1.953362 seconds of aggregation. These are process wall-time accounting values. No target scoring or new training was performed. Public delivery contains aggregate cells, qualification, completion and resource receipts; private images, masks, weights and content identifiers are excluded.
