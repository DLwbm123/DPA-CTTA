# R73: mixed target-error changes, no one-step gain

All96 fixed anchor-image pairs completed;384 output masks and288 float32 targets were scored after GPU retirement. Both worker/scorer exited zero. Both anchors passed repeated full-state equality, historical native-mask equality, unchanged original weak targets/RNG, and COMPUTE_ONLY exact post-update state/output parity. No retry occurred.

The registered target-quality, one-step and joint mechanism gates all fail. Crop targets improve some soft-error summaries but damage ground-truth foreground. The evidence does NOT establish uniformly better supervision that the BN update then fails to exploit.

## Did the pseudo-target actually correct errors?

Brier reduction below is comparator error minus candidate error; positive is better. These are raw probability-squared units, not Dice percentage points.

| Region | CROP-NATIVE reduction | Conditional95% CI | CROP-RESAMPLE reduction |
|---|---:|---|---:|
| All pixels | +0.000571924 | [+0.000222120,+0.000933485] | +0.000556815 |
| GT foreground | -0.018870565 | [-0.020440005,-0.017367531] | -0.018946807 |
| GT boundary band, radius3 | +0.019874155 | [+0.015913229,+0.024080711] | +0.019677254 |

Both anchors have the same sign in each region, and the corresponding CROP-RESAMPLE intervals have the same signs. Foreground Brier worsens in ALL16 domain/channel/anchor cells. Mean foreground reduction is -0.017764388 for OD and -0.019976742 for OC. This fails the predeclared requirement that improvement hold in all three regions against both comparators; the whole-image result cannot override foreground damage.

Threshold-.5 corrected/spoiled fractions relative to NATIVE, expressed as percentages of pixels within each region and averaged per content/channel/anchor:

| Region | Originally wrong, now correct | Originally correct, now wrong |
|---|---:|---:|
| All | 0.208986% | 0.199584% |
| GT foreground | 0.004016% | 2.120290% |
| GT boundary band | 2.402474% | 2.500087% |

There is some real correction, but almost no thresholded foreground correction and substantially more newly wrong foreground. Boundary Brier improves despite slightly more spoiled than corrected hard decisions: soft probability accuracy and thresholded decisions are different endpoints. Regions overlap, and their per-region averages cannot be added to reconstruct the whole-image result. No background-only Brier table was registered; do not attribute all whole-image gain exclusively to background.

Target hard Dice changes +0.119920pp relative to NATIVE, but CI[-0.126541,+0.351982] crosses zero; relative to RESAMPLE it is +0.114563pp, CI[-0.131434,+0.347201]. Thus hard-target superiority is also unestablished. All144 target-error/correction cells are retained in TARGET_CELLS.csv.

## Did it yield a better update from identical state?

| One-step post-update Dice comparison | Delta(pp) | Conditional95% CI | Anchor0 | Anchor1 |
|---|---:|---|---:|---:|
| CROP-NATIVE | -0.002655742 | [-0.005560141,-0.000224346] | -0.001635850 | -0.003675634 |
| CROP-RESAMPLE | -0.002649926 | [-0.005554470,-0.000220359] | -0.001635850 | -0.003664002 |
| CROP-NO_UPDATE | +0.000863486 | [-0.002831348,+0.005204547] | -0.000299427 | +0.002026399 |

There is no update gain. CROP is slightly worse than NATIVE/RESAMPLE in both anchors and does not meet the +.05pp criterion. This is a tiny one-step effect, not a catastrophic single-update loss. Worst CROP-NATIVE cell is ORIGA/anchor1/OC at -0.020088pp, which passes the separate -.5pp floor but cannot rescue the failed superiority criteria. NATIVE-NO_UPDATE is +0.003519pp, CI[-0.001516,+0.010212], so even the baseline current-image one-step effect is small and uncertain in these late states. All96 target/output Dice comparison cells are retained in DICE_CELLS.csv.

BN update norms are0.002233492(NATIVE),0.002233700(RESAMPLE),0.002234797(CROP). CROP's mean gradient norm is0.185603 versus0.166855; mean actual-update cosine to NATIVE is0.971807. The executed update changes direction, while average norm remains similar; this is not a designed norm-matched causal experiment and does not isolate a loss or optimizer failure.

## Interpretation and decision

The crop-scale intervention has mixed true-error effects: whole-field/boundary soft errors improve, foreground errors worsen systematically, and one-step Dice does not improve. It is therefore incorrect to conclude either 'no pixel is corrected' or 'the target is demonstrably better but BN cannot use it'. The supported conclusion is narrower: this fixed crop fusion does not provide region-robust improved supervision or useful same-state updates under the registered criteria.

Together with R72's closed-loop -0.550385pp result, this provides no basis to promote, expand or tune this fixed crop route. Keep CW_LSO and close the configuration. No automatic four-crop/.75-scale/mixture sweep, LR change, new short stream or full stream follows. This does not prove all alternative observations or all BN adaptation are ineffective.

R73 uses two existing visit1900 states, seed17011,48 development contents; R72 used source-initialized192-step streams, seed20260907. The late-state diagnostic is not a reconstruction of R72's entire history, and tiny single-step effects must not be extrapolated linearly to explain its cumulative delta. All data are development-exposed, anchors share contents, and patient dependence is unknown. No independent clinical/generalization claim.

## Completion and resources

See RESULTS.json, QUALIFICATION.json, COMPLETION_AUDIT.json, UPDATE_DIAGNOSTICS.csv and RESOURCE_LEDGER.json. Full GPU-process cost291.019934s (inner body288.137896s is not substituted); CPU18.583375s = preparation/check11.574615 + scoring7.008760. Online label reads zero. Float32 targets, anchor states, individual outputs and input identities remain private; public tables contain only anonymous aggregates. Protocol/source d7cbb4f and startup8e8ab17 remain unchanged.
