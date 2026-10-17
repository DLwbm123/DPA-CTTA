# R66 post-hoc error attribution

This is explanatory analysis of the fixed completed outputs, not a new method, new efficacy evaluation, or prospective confirmation. All96 historical image/order pairs and all6 correction/control branches were included. No model inference or new predictions. The CPU subprocess exited0 in4.0793583393 wall seconds; GPU0.

## Main observation

The two REFUGE cup cells exhibit predominantly false-positive baseline errors, yet LOCAL adds many more cup pixels than it removes. In contrast, Drishti has predominantly false-negative baseline errors and benefits from expansion. This explains the observed domain divergence descriptively; it does not prove the cause of the residual estimator or provide a deployable domain-aware rule.

| Domain / channel | Baseline FP / FN | Added: correct / wrong | Removed: correct / wrong | Net area | Dice delta pp | Helped / harmed |
|---|---:|---:|---:|---:|---:|---:|
| Drishti_GS / OD | 2427 / 187035 | 26338 / 2972 | 380 / 6368 | +22562 | +1.023935 | 20 / 4 |
| Drishti_GS / OC | 7521 / 191839 | 11361 / 3146 | 976 / 5632 | +7899 | +0.715371 | 19 / 5 |
| ORIGA / OD | 59756 / 177069 | 11054 / 10588 | 3793 / 9634 | +8215 | -0.276948 | 7 / 17 |
| ORIGA / OC | 54431 / 123564 | 9680 / 6991 | 1659 / 4468 | +10544 | +0.452718 | 16 / 8 |
| REFUGE / OD | 35977 / 99789 | 10236 / 4647 | 4494 / 7902 | +2487 | +0.186522 | 13 / 11 |
| REFUGE / OC | 56900 / 5494 | 1470 / 11150 | 2595 / 1093 | +8932 | -1.861953 | 2 / 22 |
| REFUGE_Valid / OD | 187616 / 11035 | 1965 / 10589 | 15462 / 1904 | -4812 | +0.418864 | 17 / 7 |
| REFUGE_Valid / OC | 100972 / 11529 | 1654 / 11556 | 1132 / 524 | +11554 | -1.664827 | 3 / 21 |

Counts pool pixels over12 contents and2 orders per row; helped/harmed counts are24 correlated content-order units, not24 independent patients. Full order-specific data for every branch are in TRANSITIONS.csv.

REFUGE cup:11150/12620 added pixels are wrong (88.35%), and22/24 content-order units lose Dice. REFUGE Valid cup:11556/13210 additions are wrong (87.48%), with21/24 units harmed. Both channels have far more baseline FP than FN, and LOCAL increases cup area further. However, REFUGE Valid disc gains+0.418864pp with net contraction; failure is not universal to this domain or both structures. ORIGA disc is another negative cell, so the problem cannot be reduced exclusively to REFUGE cups.

Across all channels/orders/contents, LOCAL corrects104249 pixel errors and introduces99164, a net5085 fewer errors, yet mean Dice falls. Pixel pooling weights larger structures and cannot replace the registered equal-case/channel Dice metric. LOCAL helps97/192 channel-content-order units and harms95/192; the near-even count also does not establish positive mean benefit.

LOCAL net area change is+67381 pixels, whereas the four sign-randomized controls have net changes+1991,-2105,+3697,+881. Thus the positive contrast against random signs may partly reflect a systematic expansion tendency in this panel, not solely image-specific correct direction. The present audit cannot identify the relative contributions of features, mask-defined seed neighborhoods, boundary geometry or rasterization. No nesting veto occurred.

## Next bounded question for the existing method

Do image features specify a stable boundary location, or does the estimated correction move with the candidate mask that defines its side prototypes? A useful next diagnostic would hold each image/features fixed and perturb its candidate boundary in both directions with a predeclared symmetric magnitude, then measure the response before consulting labels. It must include a prediction-only geometric control; merely returning toward the original mask is not evidence that the original boundary is correct. This is a proposed measurement, not a registered or launched experiment. Read overlapping earlier diagnostics before implementing it.

Do not adopt REFUGE-specific disabling, a global sign reversal, or cup-only shrinkage on the basis of these target labels. Such post-hoc policies would need separate prospective development testing and would not turn R66 into an independent success. Do not add this failed local signal to JS. The baseline remains unchanged.
