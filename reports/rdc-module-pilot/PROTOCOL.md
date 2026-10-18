# RDC relational-loss transfer: one bounded development pilot

Frozen before target inference, 2026-10-18. User authorized implementation and one small controlled run. No automatic second round, tuning, seed expansion, or revival of the closed crop/compartment routes.

## Question and provenance

Does aligning the order of pixel-to-agent affinities improve the retained CW_LSO fundus segmenter more than aligning their values? Independently implement equations (4), (5), (7), (8) and Algorithm 1 of [RDC: Relaxed Distribution-wise Consistency](https://openreview.net/pdf?id=SzNyAJVTuM), the inspected ICLR 2027 **submission**, not an accepted-paper claim. This is a module transfer, not a reproduction of the paper's semi-supervised training or reported performance. No third-party source is copied.

## Fixed intervention

Retain the source checkpoint, six-view sigmoid pseudo-label, native strong photometric augmentation, weighted BCE, LSO coefficient 0.1, BN affine parameters, Adam, learning rate, and final original-image output. OD and OC are nested independent sigmoid channels, never a two-class softmax.

Capture the feature immediately before `seg_head`, from the first original weak forward (stop-gradient teacher) and the native strong forward (student). Both have the same spatial coordinates. Average-pool the 512x512 feature grid in disjoint 16x16 cells to a fixed 32x32 grid. There are no extra model forwards, feature heads, EMA teacher, target annotations, additional scans, source fitting, or future-image reads. Pooling and use of the current model rather than an EMA teacher are explicit transfer differences.

Select N=64 distinct teacher spatial indices by greedy min-max cosine similarity, starting from a pixel sampled with a separate CPU generator seeded by `20260907 + completed_arrivals`. Selection does not change the augmentation RNG. Gather teacher and student agents at identical coordinates. Affinities use raw dot products, not cosine-normalized logits:

`z_w[j,n] = dot(f_w[j], f_w[index[n]])`, and likewise for the student.

Three source-initialized arms:

- **NATIVE:** unchanged CW_LSO.
- **VALUE:** add `0.1 * mean_j KL(softmax(z_w[j]) || softmax(z_s[j]))`.
- **RANK:** add `0.1 * mean_(j,p<q) [1 - tanh(z_w[j,p]-z_w[j,q]) * tanh(z_s[j,p]-z_s[j,q])]`.

N=64, alpha=2 and coefficient=0.1 follow the submission. VALUE and RANK use identical feature, selection and coefficient rules. Their actual gradient magnitudes need not match; a difference is not proof of a pure directional effect. Teacher affinities are detached. This adds relational pseudo-supervision, not an independent semantic truth source.

## Fixed sample, checks and outcomes

Reuse the exposed R23/R72 SEARCH cohort: 192 contents (Drishti 22, ORIGA 57, REFUGE 57, REFUGE_Valid 56), two existing orders, seed 20260907. Six independent 192-arrival trajectories, 1152 predictions, with a freshly executed NATIVE control. No historic trajectory substitutes for a control. Patient linkage is unknown; this is development evidence, not an independent clinical test.

CPU checks: distinct and deterministic agents, isolated RNG, scale-invariant cosine selection, independent finite Kendall calculation, detached teacher, finite nonzero student gradients, decreasing loss along its gradient, zero KL for identical affinities, and zero rank gradient for tied teacher affinities. Check code imports before GPU work.

Real-model qualification uses the first image of each order: repeated complete state/output equality for each arm, exact NATIVE parity with the retained host, frozen non-BN parameters, and an additional zero-coefficient RANK check against NATIVE. The inherited host enforces 8 forwards, 1 backward and 1 Adam update per arrival. Admit the fixed matrix only if 1.5 times the slowest qualified step times 1152 plus elapsed qualification fits the 5400-second cap. Record actual reserved memory.

The GPU worker reads images only. After its successful exit, a separate CPU scorer reads labels, requires all six completion receipts and all 1152 predictions, and computes hard Dice with the established empty-mask convention. Primary: equal domains, equal OD/OC, equal orders. Report both order effects, all 48 contrast/domain/order/channel cells, absolute arm scores, image-weighted effects, update norms, relational losses, teacher margins, rank disagreement and tanh saturation. Relational diagnostics are not evidence of correct labels.

Predeclared allocation signal: RANK-NATIVE >=0.2 pp, its content-stratified paired 95% bootstrap lower bound >0, both orders positive, image-weighted delta nonnegative, worst domain/order/channel cell >=-2 pp; **and** RANK-VALUE >=0.1 pp, its lower bound >0 and both orders positive. These are investment thresholds, not clinical thresholds. Bootstrap uses 2000 draws, fixed seed 20261016, coupled repeated content across arms/orders. One seed cannot establish robustness. VALUE-NATIVE is descriptive and cannot replace the primary.

## Resources and stopping

One GPU worker on an available authorized device; check free memory with peak headroom. GPU cap 5400 seconds including qualification; CPU scorer cap 900 seconds; output cap 2 GiB. Preparation/failure costs remain recorded. Numerical or qualification failure stops evaluation; do not change coefficient, grid, temperature or sample after seeing outcomes. No scientific retry or automatic next experiment.

Runs expected over 30 minutes use the existing detached controller. Confirm startup once and return without creating monitoring. On a later status request, if completed, publish source, protocol, anonymous aggregate results and a report to the project repository. Never publish images, labels, private paths, identifiers, per-image scores/predictions, model states or third-party PDFs.
