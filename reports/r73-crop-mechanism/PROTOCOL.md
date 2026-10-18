# R73: fixed same-state crop-target mechanism diagnostic

Registered after R72's negative result was known, before new target inference or label scoring. The user explicitly requested this explanatory diagnostic and unchanged R72 interpretation. This is not a retry, rescue, independent validation, or a new performance candidate. R72 remains closed with its original failed gate.

## Fixed assets and intervention

Reuse the existing R61/R62 two late CW_LSO anchors at visit1900, seed17011, orders0/1, and their fixed balanced 48-content development panel (12 per domain). Copy the original anchor payloads/receipts and qualification identities without selecting images by R72 outcomes. The existing anchor integrity checks and native qualification masks remain mandatory. No 1,951-step replay or new historical trajectory is needed.

These states differ from R72's seed20260907, 192-step source-initialized streams. They address the mechanism at two established late states, including the historically problematic setting; they cannot reconstruct every cause of R72's cumulative loss. Each of the 96 anchor-image pairs starts from the same saved state for every branch and restores it afterwards. No branch result carries to another image.

Import R72 implementation unchanged: two .8-scale affine crops, translations -.1/+.1, second horizontally flipped; coverage-normalized target `(6*q + sum(back))/(6+sum(support))`; native six-view W, original LSO target/.1, strong augmentation, BN-only Adam and learning rate unchanged. No four-crop/.75/constant-mixture alternative, no gamma/fCE or parameter scan.

Branches: NATIVE, RESAMPLE, CROP, plus read-only NO_UPDATE output. RESAMPLE and CROP both perform the same two crop-model previews; RESAMPLE instead uses crop/inverse resampling of q in the target. Capture the exact float32 target supplied to weighted BCE, not a separately approximated teacher. Verify original weak targets and native strong-augmentation RNG match across branches. Record the actual BN update vector norm, gradient norm and cosine to NATIVE; no norm matching or LR search.

## Qualification

CPU: reuse R72 affine/support/BCE tests; validate Brier decomposition, real target-statistics code on synthetic perfect correction/identity controls and corrected/spoiled pixel counts. No target labels in preparation.

At each anchor's existing qualification image: require historical NATIVE hard-mask equality; repeat every branch from the anchor with exact output/full-state equality; run COMPUTE_ONLY by restoring anchor, making/discarding crop previews, then performing NATIVE update, requiring exact NATIVE output and full-state equality. Final anchor restoration and frozen parameters must pass. COMPUTE_ONLY is qualification only, not a fourth trajectory. Every formal pair again verifies weak targets, native RNG and final anchor restoration.

## Offline target-quality and update scoring

Save 288 float32 two-channel targets and 384 two-channel output masks across96 pairs. Labels are available only to the separate CPU scorer after successful GPU retirement and full pair coverage. No acceptance decision influences any online update.

For each channel, compute Brier error on:

1. All pixels.
2. Ground-truth foreground.
3. A fixed GT boundary band: 7x7 dilation minus 7x7 erosion (radius3), constant-zero image boundary.

These GT-defined regions are offline diagnostic regions, never an online mask. An empty required region is not imputed as zero; scoring is not evaluable until the condition is explicitly reported. Record corrected and spoiled threshold-.5 decision fractions using each comparator's own errors. Brier improvement is `error(comparator)-error(candidate)`, positive is better; raw probability-squared units, not Dice pp. Use equal contents within domains, equal domains/anchors/channels. Report every region/domain/anchor/channel cell, not just whole-image background-dominated averages.

Also report target hard Dice and post-update original-image Dice. NO_UPDATE comparisons describe one-step effects. Preserve existing update-before-predict semantics; this diagnostic endpoint never replaces a continuous-stream endpoint. Content-paired domain-stratified bootstrap2000/seed20261016 keeps the two anchors and channels coupled; intervals are conditional descriptive development evidence.

## Fixed interpretation thresholds

Target-quality support requires CROP to reduce Brier versus BOTH NATIVE and RESAMPLE in ALL THREE regions: positive mean, positive95% lower bound, positive mean at each anchor. No region/channel is dropped after results. Per-channel/domain failures are fully reported even when the aggregate gate passes.

One-step support requires CROP-NATIVE and CROP-RESAMPLE both >=+.05 Dice pp, positive95% lower bound and both anchors positive; worst CROP-NATIVE domain/channel/anchor cell >=-.5pp. This is a newly registered diagnostic threshold, not a change to R72's -2pp floor. Joint mechanism support requires both target-quality and one-step criteria.

- Target fails: no demonstrated true correction from this fixed crop mechanism.
- Target passes but update fails: improvement is not translated into the current BN-affine update.
- Both pass: conditional local mechanism evidence; R72's failed closed-loop result still stands.

No automatic new short/full stream or crop search follows any outcome. Deliver the full report and decide further scope from evidence.

## Budget and public boundary

One GPU process <=2400s including qualification; CPU scoring <=900s; run storage <=2GiB. Qualification worst-pair time x96 x1.5 plus consumed time must fit. Preserve preparation/failed costs, original frozen artifacts, and all outcomes. No source training/new model. Public output: own source/protocol, anonymous target/update tables, qualification/completion and resource receipts. Keep images/labels/identities/anchors, per-image outputs and soft targets private.
