# R42: apply the retained projection to the actual Adam descent direction

Registered after complete, verified R41 delivery; development hypothesis only.
User authorizes evidence-based non-RL increments and restricts GPUs to 0 and 1.

## Evidence and one hypothesis

R41 SEARCH GP-minus-C/W +0.344417/+0.946329pp, both orders positive, 6/6 positive,
imageweighted +0.590096/+0.440645pp, but worst W cell -4.126838pp fails -2pp.
GP-minus-GC +0.006058pp; worst cell worsens from -3.638530pp. Projection active
36.3324%; projected-gradient/reference dot minimum -5.59e-9 (no values below -1e-8).
Thus gradient projection operated correctly, but robustness did not improve.
Raw Adam displacement mean increases from GC 0.00331399 to GP 0.00351955;
actual displacement 0.00275426 to 0.00285547. These are associations, not causality.

Adam coordinate scaling and momentum need not preserve the sign of a gradient's
inner product with historical gradients. A two-coordinate CPU counterexample is
registered: g=(3,-1), m=(1,2), g dot m=1, but first Adam descent is approximately
(lr,-lr), with negative dot m. R41 did not log actual-step alignment; its role in
the adverse cells is unknown. Hypothesis: projecting the actual step as well as
the retained raw gradient reduces this unprotected direction while retaining
orthogonal adaptation. It may suppress necessary distribution-shift adaptation.

## Frozen science and matrix

Four conditions C/W/CW_LSO_GP/CW_LSO_GDP, seeds [20260907,17011,29009], two original
orders, 1,951 arrivals each: 24 new complete trajectories/46,824 scores. No borrowed
trajectories. GDP is primary; GP is the attribution control. Retain GP's raw-gradient
projection, raw EMA 0.9, loss/views/W parameters/LSO0.1, fixed C LR, and selective
post-Adam 0.0015 norm cap. No coefficient, radius, decay or threshold scan.

Save preceding raw-gradient EMA m before its current-step update. After normal
Adam, let u=theta_before-theta_after be the descent vector. If m exists and
||m||^2>1e-12, use u'=u-min(0,<u,m>/||m||^2)*m; otherwise u'=u. Set theta to
theta_before-u' only if a negative component was removed, then apply the retained
GC norm cap with the original raw-gradient conflict flag. Adam moments/counts and
raw EMA are unchanged by this post-step operation. First/zero-reference/disabled
operation is exact retained parity. Finite precision may leave near-zero dots.
No new forward/backward/optimizer step, persistent state, source retraining,
online labels, domain-dependent action, new dataset, fusion or RL.

## Evaluation, validation, diagnostics and budget

Same historical SEARCH/REVIEW split and hard Dice/both-empty1/equal domain-channel
means. REVIEW descriptive only, never future tuning or independent generalization.
Primary must gain >=0.3pp against each C/W, both orders positive, >=5/6 positive
trajectories, imageweighted >=0, worst seed-mean domain/channel/order >=-2pp.
Additionally GDP-minus-GP mean >=-0.05pp; only this extra comparison is mean
noninferiority. Primary fixed; no relaxation. Keep all negative cells, coupled
content bootstrap 2000; patient linkage UNKNOWN. No ASSD/Brier/soft Dice.

CPU checks: Adam sign counterexample, disabled/first-step exact output/full-state
parity, forced actual-step projection with finite outputs and unchanged Adam state,
retained cap, snapshot continuation. Reuse existing gates, profiles/memory guards
and GP tests. All four 12-arrival GPU profiles, whole-matrix admission with 20%
timing reserve at two workers. Diagnose SEARCH raw/projected/actual descent dots,
projection triggers/removed norms and cap/cosine/raw gradient distributions for
GP as well as GDP; GP instrumentation must not change numerical operation.

Only physical GPUs 0/1, at most two GPU workers, sufficient memory with existing
20%/512MiB reserve; invalid measured profiles refuse, cold profiles retain 5GiB
fallback. Neutral commands and registered mounted storage. New T0 starts before
profiles, 22h online/24h wall/48 GPU-worker h/32GiB. Charge all profiles and failures.
Six preceding closed rounds cost 45.08708110955027 GPU-worker h. No retries or
budget reset; user-withdrawal interruption of R41 remains its separate evidence.

## Delivery

All workers retire before independent CPU scoring. Publish all anonymous results,
negatives, mechanism distributions, costs and own source/protocol on the R42
branch in DPA-CTTA, explicit proxy for every GitHub operation; verify remote SHA
and anonymous report then update DELIVERY. Exclude private images/labels,
per-content predictions/scores/identifiers/hashes, model states, private paths/logs
and third-party PDFs. Follow only evidence-based separately registered increments;
pause monitoring only after frozen development success and verified delivery.
