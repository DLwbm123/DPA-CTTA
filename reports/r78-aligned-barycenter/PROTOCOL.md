# Aligned-view barycenter: frozen 3×2 development comparison

Authorized scope: implementation, bounded qualification, then 192 existing development contents × two retained orders × seeds 20260907/17011/29009 × three independent learners. Eighteen continuous trajectories, 3456 formal updates, two readouts per arrival, 6912 scored predictions. No FULL stream, new seeds, retuning, module combinations or automatic retry/extension.

## Fixed methods

All arms retain the original CW_LSO BN affine parameter set (19136 parameters), frozen head/non-BN parameters, Adam, six native weak-view soft targets, W (variance temperature 0.1, boundary boost 2), and orientation LSO coefficient 0.1. Each arrival takes one backward and one Adam step. No EMA, ASL, SDL, HVP, feature normalization or image replay. Model, optimizer and RNG state persist throughout each trajectory; no domain resets.

The original strong augmentation is drawn once. A deterministic horizontal flip of that exact normalized strong tensor supplies the second view. Separate batch-size-one forwards preserve native BN statistics. The flipped logits are inverse-aligned before computing either loss. With retained complete loss ell(z;q,w), fixed detached native q/W within each arrival:

- NATIVE: ell(z1;q,w).
- VERTEX: (ell(z1;q,w)+ell(z2;q,w))/2.
- BARY: ell((z1+z2)/2;q,w).

The frozen pointwise linear segmentation head makes aligned feature averaging equivalent to aligned logit averaging. Both gradient branches remain connected. No extra gradient rescaling or arm-specific learning rate. Across independent continuous trajectories the endogenous targets may diverge; only their generation algorithm remains identical.

Each updated state produces O (original clean-image sigmoid probabilities) and A (equal probability average of the original and inverse-aligned flipped clean image). These are two observers, not independent learners. A never feeds training. The observer restores full native state, optimizer, gradients, RNG, counters, buffers, modes and diagnostics. It adds one physically charged forward: NATIVE 9 total forwards/arrival; VERTEX/BARY 10. All have one joint backward, whose FLOPs/memory are higher for two gradient branches.

## Qualification and resource barrier

CPU algebra, flip alignment, linear-head identity, complete dependency imports and synthetic scorer/gate checks precede GPU work. One GPU qualification process exercises all three arms, exact full-state replay, original-host NATIVE output/state equality, exact observer state/diagnostic invariance, NATIVE next-arrival equality after observation, both branch gradients, optimizer/parameter ownership, finite values and frozen parameters. Eleven qualification updates maximum, separate from formal accounting. Failure stops the campaign; no repair/retry in the running campaign.

Before formal workers start, qualification must PASS and its conservative projection (maximum measured qualification step ×3456×1.3 +120 GPU process seconds) must fit remaining GPU time and wall deadline with 900 seconds reserved. All qualification workers retire before formal workers start. Existing frozen input manifests/checkpoint/reference dependencies are reused. At least 10 GiB free GPU memory is required on each registered device at admission. No fallback to other devices.

Hard limits frozen before qualification: 7200 aggregate GPU process seconds including qualification/failed processes; 3 hours from activation; 1 GiB output (32 MiB reserve for metadata); 8 GiB PyTorch allocator limit per worker on the registered 24 GiB GPUs; CPU scorer 600 process seconds. Qualification process cap 300 seconds. Formal worker caps split remaining GPU time equally. CPU preparation process costs are retained separately. Budget enforcement stops only this campaign's processes. No automatic retry, budget reset or extension.

Formal workers each execute one retained order across all seeds/arms. Each trajectory starts from the same retained checkpoint and seed, maintains continuous state, and preserves a private terminal snapshot. All formal/qualification GPU processes must exit successfully and mark GPU_COMPLETE before the CPU label scorer runs. No online label reads, source-image access or future-image previews. Stored prediction masks are private.

## Analysis and terminal stopping rules

Primary contrasts: BARY/O−NATIVE/O (training package), VERTEX/O−NATIVE/O (ordinary two-view training), BARY/O−VERTEX/O (barycenter-specific increment). Within-row A−O identifies output averaging; the difference of those readout effects between BARY and NATIVE identifies interaction. BARY/A−NATIVE/O is a combined effect only.

Primary endpoint: domain-macro OC Dice. Report OD/OC macro, all domain×channel×order×seed cells, all negative cells, image-weighted differences, worst cells, FP/FN normalized by full image pixels, foreground fractions and first/second-half aggregates. Anonymous domain labels remain stable. Historical NATIVE/O masks must match all six retained reference trajectories exactly.

Use 2000 domain-stratified paired content bootstrap draws; average paired anchors within content. CI is conditional development-content uncertainty, not patient-level independence or a complete seed uncertainty estimate. The 192 contents have already informed earlier development; this is not independent validation.

Learning gate for BARY/O−NATIVE/O: OC ≥+0.2 percentage points, OC paired 95% lower bound >0, overall macro ≥0, OC positive in both orders and at least 5/6 seed×order trajectories, image-weighted total gain ≥0.

Harm gate against NATIVE/O: each domain×channel mean ≥−0.5 pp; each domain×channel×order mean over seeds ≥−2 pp. List all individual-seed harms regardless of threshold. These are development allocation thresholds, not clinical safety limits.

Specificity gate for BARY/O−VERTEX/O: OC paired 95% lower bound >0 and overall macro ≥0. If only A improves, attribute to readout averaging; if V/O and B/O improve without clear separation, barycenter-specific learning remains unestablished. Failure of any main gate ends further investment within this campaign. CI crossing zero means insufficient evidence, not universal mechanism failure.

Performance gates apply only after complete frozen terminal scoring. No true-label-driven online early stopping or best-checkpoint selection. Nonfinite tensors, ownership/frozen-state/observer failures, access-boundary violations or exhausted resource caps stop execution and preserve incomplete evidence. Passing all gates does not authorize another experiment.

## Monitoring and delivery

User-authorized half-hour monitoring; remain quiet for unchanged/non-actionable states. Notify on meaningful progress, failure, reliable stall, completion or required action. On terminal state audit actual process retirement, preserve all costs/failures and publish sanitized source, aggregate results and report to the project's public GitHub repository. Never publish images, labels, per-image identifiers/predictions, model weights, terminal checkpoints, raw logs, credentials or private paths. After delivery pause monitoring. No experiment expansion.
