# R43 SCNP startup — scientific results pending

Frozen source commit: 601b72503955d2ba8b578463f50f9cd1c4ed02e0.
Original T0: 2026-10-11T07:55:23.774913+00:00 (15:55:23 Asia/Shanghai).

Conditions: C, W, CW_LSO, CW_LSO_SCNP. Three retained seeds and two orders:
24 new complete trajectories, 1951 arrivals each, 46824 expected CPU scores.
Only physical GPUs 0/1, at most two workers. No GC/GP/GDP added to CW_LSO.

The primary adds fixed 0.1 W-weighted soft-target BCE on 3x3 worst same-pseudo-class
logits, retaining LSO0.1 and original CW supervision/inference. This is our
unlabeled adaptation of supervised CVPR 2026 SCNP, not a reproduction or proven
medical/topological improvement. Pseudo-membership may reinforce errors.

Twelve CPU mechanical checks PASS; CPU preparation 33.13669395446777s, GPU0.
All four 12-arrival profiles PASS; whole fixed matrix admission is true.
The fixed queue is RUNNING_FIXED_MATRIX. No GPU failures at startup verification.
The conservative profile estimate is 32.51196756721507 GPU-worker hours, not actual
cost or a guaranteed ETA. Budgets: 22h online, 24h wall, 48 GPU-worker hours,
32GiB. Every profile and failure is charged. No retries, scans or matrix pruning.

Stage success requires SEARCH primary-minus-C/W each >=0.3pp and primary-minus-
CW_LSO >=0.1pp; each comparison requires both orders positive, >=5/6 trajectories
positive, imageweighted gain>=0 and worst seed-mean domain/channel/order>=-2pp.
All negative cells are delivered even on failure. SEARCH/REVIEW were historically
exposed; REVIEW is descriptive, not tuning or independent generalization.

Hourly monitoring covers only this round; healthy states stay quiet. After online
retirement and independent CPU scoring, verify all 24x1951 observations and costs,
analyze mean/robustness/SCNP diagnostics, then deliver anonymous complete results
and all negatives to this branch. Startup is not scientific completion.
No automatic next round. Private data, predictions, identities, states and
third-party PDFs are excluded.
