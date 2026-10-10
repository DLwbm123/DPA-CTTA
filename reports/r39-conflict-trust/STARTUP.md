# R39 startup: gradient-conflict selective BN trust cap

R38 completed24 trajectories/46,824 independent CPU score rows and was delivered
at f3aea1a49d1dc542db6180b5d2b3a75c25a6f783. Its primary failed: SEARCH changes
versus C/W/CW_LSO were-0.935696/-0.333784/-1.288808pp. Closed cumulative GPU-worker
cost across R36/R37/R38 is29.205472535888354h. All negative results remain public.

R39 keeps CW_LSO and uses the same0.0015 BN displacement cap only when the current
BN gradient has negative cosine with its preceding raw-gradient EMA (decay0.9).
First/zero-norm/aligned steps are uncapped. Adam moments/steps, losses/views and
LSO0.1 are unchanged. This is an unverified development hypothesis, not success.

Frozen four conditions C/W/CW_LSO/CW_LSO_GC, three seeds/two original orders,
1,951 arrivals each:24 complete trajectories. Seven CPU tests pass, including
controls/disabled/first-step parity, forced conflict cap and EMA state restoration.
Three pre-GPU preparation failures (environment and test fixture only) are retained;
all preparation CPU33.20613431930542s, GPU0. No scientific retry has occurred.

Original T0:2026-10-10T07:53:54.601723+00:00. Four GPU profiles pass and the whole
matrix is admitted; projected8.628230h GPU-worker including timing reserve.
First three formal jobs have progress, no failed GPU receipt, neutral process
commands checked. Caps22h online/24h wall/48 GPU-worker h/32GiB; no budget reset.
Other GPU workloads coexist with sufficient checked memory; duration may vary.

Gate retains >=0.3pp versus each C/W, both orders positive,>=5/6 positive
trajectories, imageweighted nonnegative and worst seed-mean cell>=-2pp; also
CW_LSO_GC-minus-CW_LSO mean>=-0.05pp. Primary is fixed, no scan/promotion/tuning.
All workers retire before scoring. Reused SEARCH/legacy REVIEW are historically
exposed, REVIEW descriptive only; no independent generalization claim.

Results remain pending. Hourly necessary monitoring continues on this round.
