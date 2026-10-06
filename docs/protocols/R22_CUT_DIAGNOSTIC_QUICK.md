# R22: causal relative-cut judge diagnostic

Authorized 2026-10-06 after R21: fixed short test of relative cut against confidence
and entropy. This is a diagnostic, not RN-DPO or a new trained method.

- Reuse R21's 192 SEARCH identities, both orders and seed 20260907. Replay CW
  unchanged. Require every final hard prediction to match its existing R21 receipt.
- Candidates are current six-view CW pseudo-labels and predictions from the same
  frozen source checkpoint. Nested OD/OC probabilities are used for both candidates.
- Freeze the source encoder. Pool up3 features to 32x32 query tokens and 16x16
  memory tokens; retain only the preceding eight images. Minimum two past images.
- Both candidates share a cosine mutual-16-nearest-neighbor graph, temperature .1.
  No current query votes for itself. Memory is written after the current step.
- Compute each model's distance-weighted neighbor-label support; cut = 1-support.
  Bilinearly interpolate three-class neighbor votes, then gather the full-resolution
  predicted class. Regions are connected three-class disagreements of >=32 pixels.
- At least half the region must have >=4 mutual neighbors. Abstain if choosing the
  source would erase an entire OD/OC structure present in the student's prediction.
  The guard mitigates one collapse case; it does not prove correctness or remove
  the effect of foreground/background imbalance and genuine anatomical boundaries.
- Cut preference score = mean(z_student-z_source). Confidence score = source minus
  student assigned-class probability. Entropy score = student minus source
  three-class entropy (nats). Fixed source-selection threshold .05 for each score.
  Their scales and coverage differ; ranking AUROC is the principal proxy comparison.
- None of the three judges affects model updates. No new model, source training
  images, external weights or online labels. BASE bits are registered prior outputs,
  consumed at the matching current index solely to verify replay parity.
- CPU scoring begins only after both GPU workers retire. Report candidate-selection
  Dice, correction/damage counts, low-view-variance (<=.001) counts, eligible-region
  preference AUROC (excluding true-error ties), each order, and measured costs.
- The region pixel-error oracle is diagnostic headroom, not an online method or
  a mathematical upper bound on global Dice. True labels never choose online actions.
- A promising judge requires >=100 informative eligible regions, cut AUROC >=.6
  and at least both simple proxies, positive net pixels and macro Dice in both
  orders, and positive low-variance net correction. No tuning or auto follow-on.
- Two workers on GPUs 4/5; eight-minute online deadline, twelve-minute total budget,
  sixteen GPU-worker-minute cap, 2 GiB disk cap. No retry or parameter search.
- Historically exposed SEARCH, one seed, patient linkage unknown. No independent
  validation, clinical claim or end-to-end adapted-performance claim.
- Publish code, protocol, aggregate results and report; retain data, private outputs
  and bindings on the existing approved storage.
