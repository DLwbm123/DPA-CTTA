# R31: adapting the existing final classifier

Status: **COMPLETE**.

| Arm | SEARCH macro Dice (%) |
|---|---:|
| ANCHOR | 78.815873 |
| C_CONT | 78.691553 |
| C_HEAD | 78.695005 |

C_HEAD versus ANCHOR: -0.120867 pp; development gate False.

C_HEAD versus C_CONT: +0.003452 pp; development gate False.

Pass against both references: False. No learning-rate sweep is authorized by this result.

GPU-worker time 3.5416 h includes profiles and failures.

The 66 existing final-classifier scalars use fixed Adam LR 1e-5; BN affine uses 1e-4. Same one-step consistency objective and current-image readout. No extra model or source training. SEARCH and legacy REVIEW are development-exposed; REVIEW and coupled-content intervals are descriptive, with unknown patient linkage and ROI provenance. This only tests the registered parameter family and learning rate. Public artifacts exclude data, masks, identities, states and weights. Publication verification is a separate delivery step.

## Completion evidence and interpretation

All18 full streams completed1951 arrivals each (35118 total), with all process exits0 and96 exact ANCHOR reference probe checks. Last online retirement preceded CPU scoring and label release. Zero execution failures. Actual GPU-worker time3.541630h (including profiles), CPU270.792s, wall1.272008h. These are worker durations, not exclusive occupancy or energy. No cumulative GPU cap was imposed.

| Candidate contrast | SEARCH delta (pp) | Order0 / order1 (pp) | Positive trajectories | Descriptive REVIEW delta (pp) |
|---|---:|---:|---:|---:|
| C_HEAD - C_CONT | +0.003452 | -0.008976 / +0.015880 | 3/6 | +0.005612 |
| C_HEAD - ANCHOR | -0.120867 | -0.325264 / +0.083529 | 3/6 | -0.133350 |

The stronger fixed reference is ANCHOR in this round. Both registered candidate gates fail on magnitude, order consistency and positive trajectory count. The worst seed-averaged SEARCH cell is -0.071955pp versus C_CONT and -1.334974pp versus ANCHOR; these pass the adverse-cell floor but do not rescue the other failed conditions. The small positive content-bootstrap interval against C_CONT does not establish meaningful, independent or patient-level effectiveness, especially after repeated development exposure.

The candidate was active: across18 seed/order/role summary groups, mean head gradient norm ranged0.018013–0.023414 and mean head drift0.031533–0.040066, with maximum drift0.061017. This rules out the narrow explanation that the classifier never updated; it does not identify a general causal reason for the weak Dice effect. Fresh C_CONT and ANCHOR SEARCH aggregate scores reproduce the R30 references to displayed precision.

## Campaign decision

Close the registered classifier-at1e-5 hypothesis. No R32 is launched and no classifier learning-rate sweep follows. See [CAMPAIGN_CLOSEOUT.md](CAMPAIGN_CLOSEOUT.md) for the evidence-based stop decision. This is a lack of an actionable supported next experiment under the current setup, not proof that every classifier, backbone, objective or CTTA method must fail.

All paired contrasts, adverse cells, domain/channel results, state diagnostics and costs accompany this report. Source/protocol remain pinned to the runtime commit documented in REGISTRATION.json. Public-delivery verification is recorded separately after push.
