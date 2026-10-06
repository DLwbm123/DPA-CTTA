# R24: C strong baseline and anchored conditional adaptation

## Authorization and research question

The user selected ordinary consistency C as the main strong baseline and authorized continued experiments, hourly monitoring and autonomous engineering repair. This round is bounded to six runtime wall hours, twelve cumulative GPU-worker hours and 12 GiB of output. The original launch time and all attempts remain in the ledger. This authorization does not make negative scientific results into engineering failures.

R19 validation/history rejection, R21 historical prototypes, R22 local-graph selection and R23 disagreement masking/interval constraints did not establish benefit under their registered settings. These failures do not prove a common causal explanation, but provide little support for another reliability gate. R20 adapters were evaluated mainly on GraTa, and ordinary C was the stronger simple control. The present hypothesis concerns the capacity and consistency of the adaptation function, not selection of a supposedly correct pseudo-label.

**Frozen primary: ANCHOR.** Does an original-image condition shared by all augmentation views provide a useful online residual adaptation function, beyond C, a larger C learning rate, a static adapter, and the same conditional adapter with a separately computed condition for each view?

This is a candidate mechanism, not a claim of novelty or superiority. Conditional networks are established. HyDA learns domain-conditioned parameters with a source-trained domain encoder; DOME learns domain variables with sparse supervision. The present model-only test uses no source retraining, external encoder, task/domain labels, confidence judge or semantic memory. The potentially useful distinction to test is augmentation-consistent conditioning in online learning; this distinction alone is not proof of a publishable contribution.

- HyDA: https://papers.miccai.org/miccai-2025/0428-Paper0392.html
- DOME: https://arxiv.org/abs/2606.07646
- Ordinary consistency implementation inherits the pinned GraTa augmentation path: https://arxiv.org/abs/2408.07343

## Fixed implementation

All arms use the same source checkpoint, native six weak views, one strong view, BN-affine adaptation, Adam, input preprocessing, output convention and no W pixel weights. C uses fixed LR 1e-4. C_LR15 uses fixed 1.5e-4, reset to that constant at every step, not compounded.

At the existing up3 output, the three adapter arms use a rank-4 residual:

    r = sqrt(mean_channel(h^2) + 1e-6), detached
    g(d) = 1 + 0.5 tanh(M d + b)
    h' = h + 0.1 r tanh(U[gelu(V(h/r)) * g(d)])

U starts at zero, V uses the existing independent seeded initialization, M has seeded Gaussian entries with std 1/sqrt(context dimension), and b is zero. Residual initialization is exactly identity. BN LR is 1e-4; adapter LR is 3e-4 in all three adapter conditions. There is no source fitting of the new parameters. All adapter and BN-affine parameters persist across arrivals; the context resets on every arrival.

The descriptor uses detached first-BN-input channel means and population variances. Relative to checkpoint BN moments, concatenate (mu-current minus mu-source)/sqrt(var-source+1e-5) and 0.5 log((var-current+1e-5)/(var-source+1e-5)), then apply tanh. Upstream of the first BN the feature extractor is frozen. The descriptor is an appearance proxy that can also encode image content, not a proven domain identity or anatomical invariant.

| Arm | Condition supplied to the identical adapter |
|---|---|
| C | No adapter |
| C_LR15 | No adapter; larger fixed BN learning rate |
| STATIC | Zero descriptor for every view; gate bias may learn |
| VIEW | Descriptor recomputed from each current view |
| ANCHOR | Descriptor from the first, unaugmented current-image forward, reused across six weak views, strong view and final prediction |

STATIC matches the residual architecture and optimization but has less effective conditioning capacity; zero-input gate weights receive zero gradient. VIEW and ANCHOR exactly match trainable parameters and forwards. Thus ANCHOR versus VIEW is the key conditioning ablation. None of these conditions filters pixels or adds target-label information. All retain eight forwards, one backward and one Adam update per arrival; adapter math adds real overhead that is measured rather than assumed negligible.

## Data, stages and gates

Use the two existing complete 1,951-arrival streams, with 1,695 primary identities split previously into 1,017 SEARCH and 678 campaign-held REVIEW identities. The remaining 256 arrivals are inherited context/development roles. All data have historical exposure; patient linkage and original ROI crop provenance remain UNKNOWN. This is not a new independent clinical test. Online manifests contain only current-image fields, never domain IDs, roles, label paths or prior metrics.

1. CPU synthetic qualification; then one 12-arrival GPU engineering/profile job for each of C, STATIC, VIEW and ANCHOR. No labels. Profile attempts are charged. The C_LR15 cost estimate reuses C's profile.
2. Run all five conditions, native seed 20260907, both full orders: ten independent trajectories. Score SEARCH only after all workers in this stage retire.
3. Advance only if ANCHOR-C >= +0.2 pp, ANCHOR-STATIC >= +0.1 pp, ANCHOR-VIEW >= +0.1 pp and ANCHOR-C_LR15 > 0; each comparison must be positive in both orders, imageweighted change nonnegative, worst domain/channel/order >= -2 pp. No ranking substitution: STATIC or VIEW winning does not silently become a new primary.
4. If the gate passes, freeze confirmation for the same five conditions and seed 17011, plus seed 29009 only if the complete additional batch fits measured p95-based cost with 20% margin. The seed set is fixed before new-seed labels or REVIEW are opened. If even 17011 cannot fit, mark NOT_CONFIRMED_BUDGET. No parameter edits or extra candidates.
5. After all registered formal workers retire, open REVIEW once and score all completed trajectories. A negative SEARCH gate still receives final descriptive reporting; no confirmation or retuning follows it.

Primary metric is domain x OD/OC equal-weight hard Dice, averaged across orders and seeds. Dice uses 2TP/(predicted+GT), or 1 for mutually empty masks, matching R20. Report imageweighted mean, every domain/channel/order/seed cell, empty masks, containment violations, paired content-cluster bootstrap intervals (orders/seeds coupled), all negative cells, and measured cost. ASSD and soft-probability metrics are NOT_COMPUTED in this bounded mechanism round; do not invent them from masks or call them undefined because of an empty structure.

The final development signal requires at least two seeds, REVIEW ANCHOR-C >= +0.3 pp, both order means positive, at least 3/4 (two seeds) or 5/6 (three seeds) positive trajectories, imageweighted gain nonnegative, worst seed-averaged cell >= -2 pp; ANCHOR must also exceed STATIC, VIEW and C_LR15 on REVIEW mean. Otherwise effectiveness is not established. Exploratory bootstrap intervals do not remove historical exposure or patient-dependence limitations.

## Runtime, repair and delivery

Use the existing detached guarded runtime, GPU assignments 4-7 after current memory checks, prescribed NAS directory, neutral executable/entrypoint, and a single-start supervisor lease. Online computation stops by T0+5.25 h, all computation by T0+6 h, and the GPU ledger cap is 12 h. The watchdog only terminates this campaign's verified process identities. No source images, extra checkpoints, online labels or future images are permitted.

Reuse the existing checkpoint journal and its built-in identity/integrity checks. A recorded transient filesystem/network errno in {4,5,104,110,116} may resume once from its own validated snapshot, preserving all prior costs and physical attempt logs. Numerical failure, bad score or exhausted budget is not a transient recovery. Other concrete code/runtime bugs may be repaired under the user's authorization only with a recorded revision, unchanged scientific settings, preserved failed receipts and consistent affected comparisons; never erase an attempt or reset the original clock.

Hourly monitoring checks actual identities, logs, receipts, coverage and cumulative cost. Quiet when unchanged; notify on meaningful stage conclusions, faults/recovery, completion or needed user action. After a terminal round, publish code, protocol, aggregate results, negative cells and cost to the project GitHub through the configured proxy and verify anonymous access; then pause this round's monitoring. Keep images, labels, content identifiers, predictions, checkpoints, private paths and credentials private. Do not automatically start an unregistered next research round.
