# R33 matched-age parameter/Adam state exchange

Status: **COMPLETE**.

Finished at 2026-10-09 18:33:21 Beijing. All six online jobs and the CPU scorer exited successfully; all 48 branches and 64,854 condition-content observations are present. [Final receipt verification](FINAL_AUDIT.json) confirms 44,970 formal optimizer steps, 391,488 forwards, 15,864 diagonal checks for each of pre/post predictions, 24 first-pre-update invariance checks and zero online label reads. The planned scheduling handoff completed; its retired supervisor's cancellation was not a scientific failure.

| Domain | Contrast | SEARCH pp | Order0 / order1 | Positive trajectories |
|---|---|---:|---|---:|
| ORIGA | CC - CS | +0.760265 | [0.9815382315265898, 0.5389913515371987] | 6/6 |
| ORIGA | OPTIMIZER_BY_PARAMETER_INTERACTION - ALGEBRA | -0.030103 | [-0.06486046427248245, 0.004654371060112794] | 2/6 |
| ORIGA | SC - CC | +1.735144 | [1.1987786117349586, 2.2715090120705526] | 6/6 |
| ORIGA | SC - SS | +0.730162 | [0.9166777672541073, 0.5436457225973115] | 6/6 |
| ORIGA | SS - CC | +1.004982 | [0.28210084448085126, 1.727863289473241] | 6/6 |
| ORIGA | SS - CS | +1.765247 | [1.263639076007441, 2.2668546410104398] | 6/6 |
| REFUGE_Valid | CC - CS | -0.097209 | [-0.028846585026952216, -0.16557193771048398] | 1/6 |
| REFUGE_Valid | OPTIMIZER_BY_PARAMETER_INTERACTION - ALGEBRA | +0.060143 | [0.055055198563460785, 0.06523035720578463] | 6/6 |
| REFUGE_Valid | SC - CC | +0.403413 | [0.4667524311799009, 0.3400731592577652] | 6/6 |
| REFUGE_Valid | SC - SS | -0.037066 | [0.02620861353650857, -0.10034158050469937] | 2/6 |
| REFUGE_Valid | SS - CC | +0.440479 | [0.4405438176433923, 0.44041473976246454] | 6/6 |
| REFUGE_Valid | SS - CS | +0.343270 | [0.4116972326164401, 0.2748428020519806] | 6/6 |

First letter is BN-affine parameter history; second letter is Adam history. S=SAME64, C=CROSS64. Both histories have64 updates and Adam step64. SC-SS and CC-CS compare Adam donors at fixed parameter states; SS-CS and SC-CC compare parameter donors at fixed Adam states. Interaction=(SC-SS)-(CC-CS). Hybrid losses can indicate parameter/optimizer incompatibility, not universally harmful optimizer memory. Diagonals reproduce R32 pre/post hard masks. Frozen references reuse R32 on the identical query tails.

R32 already established prediction-relevant parameter history differences; this round tests subsequent-update sensitivity to Adam memory. No online domain labels, learned selector, LR grid, new source checkpoint or optimizer age reset. All ordered pairs, channels, fixed first64/quarter bins and negative cells are retained. SEARCH/legacy REVIEW are development-exposed; unknown patient linkage and ROI provenance prohibit independent or clinical claims. State exchange is an offline diagnostic, not an executable policy.

GPU-worker hours including profiles/failures: 6.046906. Public delivery verification is separate.

## Interpretation of the registered SEARCH contrasts

The result does not support treating CROSS64 Adam memory as universally harmful. On ORIGA, replacing SAME64 Adam with CROSS64 Adam improves the full query-tail mean by +0.730 pp at SAME64 parameters and +0.760 pp at CROSS64 parameters. Both orders and all six seed/order trajectories have positive mean effects. On REFUGE_Valid, the corresponding effects are -0.037 and -0.097 pp. The first contrast changes sign by order (+0.026 versus -0.100); the second is negative in both order means (-0.029 versus -0.166). These are conditional effects of the two specified donor histories at equal optimizer age, not evidence about all cross-domain memories.

At fixed Adam donor, SAME64 parameters produce larger full-tail gains in this matrix: ORIGA +1.765/+1.735 pp, REFUGE_Valid +0.343/+0.403 pp for S/C Adam respectively. The factorial interaction is -0.030 pp on ORIGA and +0.060 pp on REFUGE_Valid. Those point estimates are small relative to the parameter contrasts here; there is no broad collapse of the hybrid states. This weakens a strong generic incompatibility explanation for these particular swaps, without proving compatibility in other histories or isolating Adam m from v.

The positional results also resist a blanket optimizer-origin rule. On ORIGA, both conditional CROSS64-Adam effects are positive in FIRST64 and each fixed quarter, in both order means. On REFUGE_Valid, SC-SS is +0.016 pp for FIRST64 but -0.025/-0.034/-0.044/-0.043 across Q1–Q4; CC-CS is approximately -0.096 to -0.100 across the four quarter means. The published order/channel/seed rows retain their exceptions. Six trajectories share image contents and do not constitute six independent patient cohorts; these signs are descriptive, not a statistical significance claim.

## Absolute performance and the unresolved update problem

Values below are SEARCH Dice percent, averaged across the registered seeds, orders and OD/OC channels on the same query tails.

| Condition | ORIGA | REFUGE_Valid |
|---|---:|---:|
| SS | 76.990 | 74.273 |
| SC | 77.721 | 74.236 |
| CS | 75.225 | 73.930 |
| CC | 75.985 | 73.833 |
| SAME64_HOLD | 70.296 | 75.913 |
| CROSS64_HOLD | 66.146 | 74.562 |
| SOURCE_HOLD | 66.011 | 75.239 |
| Native C_CONT | 79.626 | 70.821 |
| ANCHOR | 80.082 | 70.741 |

All four UPDATE states improve ORIGA over their parameter-matched HOLD (+6.695/+7.425/+9.079/+9.839 pp for SS/SC/CS/CC). All four worsen REFUGE_Valid against their matched HOLD (-1.640/-1.677/-0.633/-0.730 pp); the direction holds in both order means. Merely choosing one of these Adam donors therefore does not eliminate the retained-update loss on REFUGE_Valid. This does not test a cold optimizer or a learned retention decision.

None of the four states dominates the existing references across domains. The highest ORIGA mean among the four, SC, remains 1.906 pp below C_CONT and 2.361 pp below ANCHOR. REFUGE_Valid SS exceeds those two references by 3.452/3.532 pp but remains 1.640 pp below SAME64_HOLD and 0.966 pp below SOURCE_HOLD. These comparisons describe the fixed diagnostic matrix; no winning deployment policy is selected.

## Decision and evidence coverage

Stop this completed diagnostic rather than introduce another Adam-origin swap or optimizer-reset sweep. A sparse decision about retaining an update remains a hypothesis motivated by the opposite HOLD/UPDATE effects across domains. R33 supplies neither a label-free decision signal, a deployable trigger, a reward nor evidence that RL is necessary. The user's discussion of RL at selected stages does not authorize an additional RL experiment. No R34, controller or new GPU run is started with this report.

The complete release includes [absolute cells](QUERY_CELLS.csv), [paired cells](PAIRED_CELLS.csv), [paired summaries](PAIRED_SUMMARY.csv), all 2,504 [negative cells](ALL_NEGATIVE_CELLS.csv), [state diagnostics](STATE_DIAGNOSTICS.csv), [costs](COST.csv), [completion audit](COMPLETION_AUDIT.json), the frozen protocol and scheduling receipts. Both SEARCH and already-exposed legacy REVIEW, both orders, OD/OC, every seed, ALL/FIRST64/Q1–Q4 are retained. Private per-content scores, raw images/labels, masks, model states and checkpoints remain excluded from publication. No additional scientific GPU tests were run for delivery; independent and clinical validation remain unavailable.
