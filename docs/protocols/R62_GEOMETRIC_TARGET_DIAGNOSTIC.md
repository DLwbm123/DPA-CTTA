# R62: fixed geometric pseudo-target diagnostic

Registered before new inference/label access. Hypothesis: the six-view probability mean can inject orientation-sensitive prediction bias into BCE supervision, and using the current student's original-view probability may improve a matched immediate update. This is a diagnostic, not a novel algorithm claim or an inference that augmentation is generally harmful.

## Evidence and overlap audit

R11 tested frozen output averaging; R12 qualified its source replay and measured H_ONLY minus C0 = -4.665606 pp on its short target cohort, with strong source probability degradation. Thus transformation sensitivity is observed, but only for that frozen inference setting. R13/R14 extend or gate output fusion. R20 T blends an EMA original/flip teacher into the six-view target (mix .25/.5); its current-student ablation uses that teacher pathway. R23 masks or bounds source/native disagreement. None of these registered implementations replaces only current-student BCE supervision with its original view while retaining native reliability and structural supervision at matched saved states. This bounded repository audit does not claim literature novelty.

Native `cal_consis_loss` uses original, horizontal/vertical flip, and 90/180/270 degree rotations, inverse-aligns logits, then averages sigmoid probabilities. The transform/inverse pairs are algebraically correct in the inspected implementation; no bug is asserted. R61 changed the strong training input and failed its gate. It did not modify weak target construction. R61's failure is not evidence that weak targets are biased.

## Matrix and exact intervention

Reuse only the two R51 CW_LSO visit1900 anchors (seed17011, orders0/1) and the same 48 balanced SEARCH contents, 12/domain. Restore every state before every branch. R51 itself remains NOT_EVALUABLE; fresh native historical-mask/repeated-state qualification is mandatory.

For each of 96 independent state-image pairs, retain NO_UPDATE, NATIVE and ORIGINAL_TARGET update outputs, plus the native six-view and original-view hard pseudo masks: 480 scored masks. The intervention changes only weighted BCE target q6 to detached q0. Implement as native total loss plus mean[W*(BCE(z,q0)-BCE(z,q6))]. W is still computed from all six native views; LSO still uses q6; strong augmentation, Adam/moments/LR, all model calls and original-image final readout remain unchanged. This is intentionally a partial target intervention, not removal of all geometric supervision. No branch state propagates. No coefficients, view selection, temperature, threshold or LR search.

Qualification: both native historical masks exact; repeat each branch with exact output/full-state parity; native and intervention weak views and terminal native RNG equal; operation counts equal; replacement q0=q6 must reproduce native output/full state exactly; restore original criterion even on exceptions; final anchor restoration exact. CPU fixture checks weighted BCE value/gradient algebra and zero substitution. Failure stops before the formal matrix, with receipts preserved.

## Frozen outcomes and decision

Primary ORIGINAL_TARGET minus NATIVE hard Dice, equally weighted domains/channels/anchors. Allocate a later trajectory only if mean >=0.1 pp, domain-stratified content-cluster bootstrap 95% lower bound >0, both anchors positive, worst domain/channel/anchor cell >=-2 pp, and REFUGE_Valid OC mean >=1 pp. Same bootstrap as R61: 2000, seed20261016. Additional mechanistic requirement: original-view minus native-target mean >0 and both anchor means >0. Do not substitute this target-quality outcome for update efficacy. Failure is a scientific negative and does not permit a retry or expanded grid. NO_UPDATE contrasts, all cells, all negatives, gradient/parameter norms and direct target quality remain reported.

All labels stay inaccessible online. Independent CPU scoring starts only after successful GPU retirement and complete 96-pair coverage. Development exposure and two late states limit interpretation; no patient-level uncertainty, independent validation or long-horizon bound. Gradient norms are not matched. Good pseudo masks do not guarantee a useful update, and unchanged native LSO/W can interact with the new BCE target.

Budget: one GPU, <=1800 GPU-worker seconds including qualification, <=900 CPU-scoring seconds, <=45 minutes after launch, <=2 GiB new artifacts. No additional seeds/states. Preserve inherited anchor journal validation; reuse real copied inputs inside a new run root, never overwrite prior results. Publish own code/protocol, aggregates, adverse cells and cost/retirement receipts; keep images, labels, masks, weights and per-content data private.
