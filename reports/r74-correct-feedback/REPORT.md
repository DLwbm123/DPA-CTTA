# R74: correct feedback transfers weakly; allocation threshold not met

Complete: all six fixed states,18 feedback outputs and900 frozen-future outputs scored. Both worker and scorer retired with exit0. Every state passed historical native-mask parity, retained-host full-state/output parity, repeated branch equality, unchanged native weak targets/RNG, exact truth-target substitution and read-only state checks. The protocol and three branches were unchanged.

## Frozen future endpoint

Dice differences are percentage points. Each state averages its50 future images and OD/OC; the primary averages three seeds and two orders equally.

| Contrast | Mean pp | Conditional95% interval | Order0 Drishti | Order1 REFUGE |
|---|---:|---|---:|---:|
| CORRECT minus NATIVE | +0.016903 | [+0.013401,+0.020159] | +0.032041 | +0.001764 |
| CORRECT minus NO_UPDATE | +0.019017 | [+0.015580,+0.022359] | +0.028187 | +0.009847 |
| NATIVE minus NO_UPDATE | +0.002114 | [+0.001184,+0.003086] | -0.003854 | +0.008082 |

CORRECT-NATIVE is positive in6/6 state means and all12 seed/order/channel cells. The worst order/channel cell after averaging seeds is+0.001570pp. However,+0.016903pp is below the registered+0.05pp minimum. The other four allocation criteria pass; the conjunctive allocation signal is FALSE. A positive conditional interval does not replace the practical threshold.

CORRECT-NO_UPDATE is not positive everywhere: seed17011/order1 has mean-0.000079pp and its OC cell is-0.000418pp. The overall nonnegative criterion passes, but the result is not uniform improvement over leaving the state untouched. All36 contrast cells, including adverse cells, are retained in FUTURE_CELLS.csv.

## Did correct supervision enter and affect the model?

The actual CORRECT BCE target equals the authorized binary truth, yielding target Dice100% and Brier0 by construction. This is an oracle intervention, not evidence for a deployable pseudo-label generator. Original target Dice averaged57.316188% on the Drishti feedback image and91.831133% on the REFUGE feedback image (seeds/channels averaged).

| Feedback-image output | Drishti Dice | REFUGE Dice |
|---|---:|---:|
| NO_UPDATE | 58.085229 | 91.367676 |
| NATIVE | 58.090047 | 91.367345 |
| CORRECT | 58.111875 | 91.373023 |
| CORRECT minus NATIVE (pp) | +0.021828 | +0.005678 |

Current-image foreground Brier also improves relative to NATIVE:0.500223 to0.499646 on Drishti and0.041987 to0.041932 on REFUGE. Thus a substantially better supervision target produces small changes in current outputs and small positive average transfer to the fixed future images. It is incorrect to conclude that the model cannot use correct feedback at all. These two images do not establish a general relationship between correction amount and transfer.

The Drishti actual BN update norms increase from approximately0.00273–0.00299 under NATIVE to0.00621–0.00693 under CORRECT; REFUGE norms remain around0.0015–0.0023. Gradient norms and the complete values appear in UPDATE_DIAGNOSTICS.csv. This intervention changes correctness, confidence, gradient magnitude and Adam response together; it is not norm-matched direction attribution. Training losses in that table use different branch targets and are not a pre/post same-objective loss-descent measurement.

## Interpretation and boundaries

The supported mechanism statement is narrow: under these late states, an oracle-correct BCE target can yield a small beneficial change in predictions of other, subsequently arriving images. The gain is below the frozen investment threshold even with perfect current-image BCE supervision. This supplies limited evidence of usable information transfer, not a passed rationale for building a complex correction module.

The experiment cannot isolate BN capacity, Adam history, step size, or interference from the retained original-target W/LSO. It does not justify choosing one of these explanations after the fact. In particular, the original LSO still uses the native pseudo-target. A failed threshold is not proof that better targets are irrelevant or all BN adaptation is ineffective.

Future predictions were made with frozen post-feedback states. They were NOT followed by native future updates. Therefore this result does not test whether a benefit would survive, be erased, or be amplified by subsequent adaptation. No continuous-adaptation stage or extra control was started.

There are only two distinct feedback images and100 distinct future images. Seeds repeat the same contents. Each future window is disjoint from its own branch prefix, but its contents occur in the other order's prefix. Patient linkage is UNKNOWN; the confidence intervals condition on these feedback events and do not capture feedback-patient uncertainty. The two tails cover Drishti and REFUGE only, not four domains or the REFUGE_Valid deficit. All contents retain historical development exposure. This is supervised diagnostic evidence, not an unsupervised CTTA gain, independent generalization, cross-patient transfer, or a mathematical performance upper bound.

Decision: deliver and discuss these results; do not automatically add BCE-only controls, open parameters, change LR, add feedback examples, scan horizons or enter continuous adaptation.

## Completion and cost

Total GPU-process wall time162.193387s includes the preserved pre-inference packaging failure2.219344s and successful worker159.974043s; the inner worker-body158.845614s is not substituted for process cost. CPU total15.587932s includes preparation/checks7.372304s and scoring8.215629s. The earlier wrapper-only environment-merge error has unmeasured CPU-process duration and zero GPU work. All registered budgets were respected. Recorded outputs are approximately117.4MiB; peak reserved GPU memory1.621GB. No budget expansion occurred.

The GPU worker decoded only the two authorized feedback labels; future labels were scorer-only after successful retirement. Raw images, labels, checkpoints, probabilities, predictions and content identities remain private. Public delivery contains own source/protocol, anonymous diagnostic tables and qualification/completion/resource receipts. Protocol490d6d5, repaired source55f4264 and startup4817862 remain preserved.
