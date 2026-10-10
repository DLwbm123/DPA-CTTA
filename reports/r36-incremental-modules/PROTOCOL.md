# R36: fixed incremental modules on the retained positive candidate

Authorization: user requested prioritized module attempts, own LSO/LSM implementation,
hourly monitoring and continuous background execution on 2026-10-09.

## Fixed scope and evidence

Retain R20 W06_LR15: six aligned weak views, detached mean pseudo-target, variance
temperature 0.1, boundary gain 2, native GraTa BN-affine update and final LR multiplier
1.5. W beat G in R20 but did not beat the stronger plain-consistency C. Keep C as a
strong control. C+W was only a one-seed transfer signal. R21 regional prototype
correction and R22 causal cut were negative; do not repeat them or train RL here.

Seven conditions, ordered before execution: C, W, W_TP, W_LSO, W_LSM, W_LS, W_BAL.
Three seeds: 20260907, 17011, 29009. Both original orders, full 1,951-image stream.
Exactly 42 independent trajectories; original checkpoint and fresh state for each.
Run all fixed configurations, no SEARCH-driven promotion, coefficient changes, pruning,
extra seeds or post-review retuning. Freeze primary W_TP; LSO/LSM/LS/BAL secondary.
Schedule both baseline conditions first, then modules in that order, across all seeds.

## Modules and provenance

- TP: own simplified cross-threshold connected-component overlap scoring. Downsample
  detached mean probabilities 4x, 4-connected components at 0.5, match by maximum
  overlap at 0.3/0.4/0.6/0.7. Component score is minimum matched IoU over thresholds,
  empty/missing support gives zero. Weight factor 1+0.5*score is channel-normalized,
  multiplied by unchanged W reliability in BCE. No pseudo-label editing or extra head.
  Borrowed idea: TopoOT, arXiv 2601.20333, PDF pp3–6,14. This is **not** full PH/OT
  TopoOT and cannot inherit its reported gains or computational guarantees.
- LSO/LSM: independently implemented for each nested sigmoid OD/OC field, not
  softmax mutually exclusive classes. Current detached soft pseudo-target provides
  the reference. Pool4, replicate-padded averaging3, Sobel gradients, local tensor
  Jxx/Jyy/Jxy averaged3; normalized doubled-angle vector encodes axial orientation,
  tensor anisotropy magnitude encodes strength. Target magnitude times pooled W
  reliability weights the loss; flat target regions contribute zero. LSO = 1-cosine,
  LSM = L1 difference of channel-normalized magnitudes. Add 0.1*LSO or 0.1*LSM to
  unchanged W BCE; combined LS uses 0.1*(LSO+LSM)/2. Target stop-gradient, no ground
  truth or future image. Geometry at pooled resolution; no inference ensemble.
  Borrowed idea: official NeurIPS 2026 poster 154585, Local Structure Regularization.
  arXiv PDF not located; reviewed abstract/code, **not paper reproduction**. No copied
  third-party code, no claim its supervised-segmentation efficacy transfers to CTTA.
- BAL: bounded past-EMA foreground/background contribution weights, separately OD/OC.
  EMA initially0.5, momentum0.9; update only after the current native step. Frequency
  clip[0.05,0.95], inverse contribution factor0.5/freq or0.5/(1-freq), clip[0.25,4],
  factor1+0.1*(inverse_factor-1), channel-normalized then multiplied by W in BCE.
  Borrowed idea: DSBR, arXiv2606.02339. Original DSBR weights classification entropy;
  this is a small BCE adaptation, **not original DSBR**. No forced class-prior matching.

No SOAP-Bubbles/AURA in this campaign: changing optimization or meta-training would
exceed incremental scope. No SafeCut retry, new peer, source retraining, source images,
external pretrained features, online labels, target test selection or output fusion.

## Checks, resources and stop rules

Before launch: synthetic finite/nonzero geometry gradients, flat-target zero geometry,
independent-channel symmetry, empty/stable TP behavior, disabled-module exact W
output/parameter/optimizer/RNG parity, BAL snapshot/next-update parity. Real GPU
profiles for all seven candidates, 12 arrivals each, including frozen-parameter checks.
Profile cost counted from original T0; mechanical CPU preparation cost recorded separately.
Admit the complete 42-trajectory matrix using measured per-candidate max timing plus
I/O and 20% reserve. If full matrix cannot fit, report BUDGET_BLOCKED; do not choose a
smaller scientific subset based on scores. Three concurrent workers, one per GPU.

Default finite campaign protection: 24h wall clock, online22h, 48 GPU-worker hours,
32GiB output, per-task deadline based on profiles. Do not reset accounting or T0.
Retain every failed attempt and negative result. No automatic scientific retry. Engineering
repair after an identified fault must preserve old receipts and original limits; any
changed scientific definition needs user authorization. No terminating other workloads.
All visible Python entrypoints/arguments must be neutral, with config/modules in env.

## Evaluation and delivery

All online workers retire before a separate CPU scorer reads any masks. Split unchanged:
SEARCH1017 identities, legacy SEALED_REVIEW678, remaining context excluded from main
results. Both roles historically exposed: neither is independent medical evaluation.
Primary uses equal domain×OD/OC hard Dice then equal seed/order. Dice2TP/(pred+GT),
both-empty=1. No ASSD, soft Dice or Brier in this bounded campaign. Keep all adverse
domain/channel/order/seed cells, costs, coverage, empty predictions and containment.
Content-cluster bootstrap couples orders/seeds; patient linkage UNKNOWN.

Development investment signal for each fixed module: SEARCH gain>=0.3pp against
both W and C, both orders positive, >=5/6 positive trajectories, nonnegative imageweighted
gain, worst seed-averaged cell>=-2pp. No automatic expansion even if passed. REVIEW
descriptive only; no winning candidate replacement of the frozen primary after scoring.

Hourly heartbeat checks actual receipts, live logs and progress, never restarts healthy
workers. Notify hourly with compact progress/ETA when changed; immediately report fault,
budget stop or completion. At actual completion, inspect all results and negative cells,
copy anonymous public aggregates/report to the project repo and push using the required
proxy, verify remote commit and anonymous report access. Never publish private masks,
data, weights, model states, paths, patient/content identifiers or third-party PDFs.
Pause this monitor only after verified final delivery or user request. Continuous running
means this frozen background queue; it does not authorize unbounded new experiments.
