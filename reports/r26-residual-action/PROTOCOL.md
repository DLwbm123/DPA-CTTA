# R26: bounded residual-output parameter actions

Registered development follow-up to the complete R25 negative result. R25's primary
improved over C by 0.1457 pp but over ANCHOR by only 0.000385 pp, with 3/6 positive
order-seed trajectories and a failed preregistered gate. The mean SEARCH gate extrema
were 1.417 and 1.471 (upper bound 1.5), while mean candidate mask variance was
0.00000314. These observations motivate an action-space intervention; they do not
prove saturation is the sole cause or that the new method will improve accuracy.

## Fixed intervention

Retain R25's model, optimizer, reward, four candidates, RNG rollback, memory,
policy distribution and policy update. Remove only its gate-logit action hook.
Apply each four-dimensional action to the completed adapter residual:

`output + (exp(0.75*tanh(action[group])) - 1) * (output - input)`

Feature channel `c` belongs to fixed group `c % 4`. Multipliers lie in
[exp(-0.75), exp(0.75)] (approximately 0.472–2.117). Zero action reproduces the
unmodified output exactly. This acts after the context gate and residual tanh,
including adapter bias, and therefore tests a different action geometry rather
than merely increasing the old gate-logit scale. It is an empirical intervention,
not an established novelty claim. The inherited `adapter.relative_rms` describes
the pre-action residual; `adapter.applied_relative_rms` describes the actual one.

## Matrix, controls and fixed decisions

Six arms: C, ANCHOR, REPEAT5, RANDOM4, GREEDY, PG_RETAIN. The first three are exact
R25 controls. RANDOM4 and GREEDY use the same fixed Gaussian four-candidate actions
and full unlabeled reward but select randomly or by maximum reward respectively.
PG_RETAIN uses the learned Gaussian policy and the same complete reward. Unlike
R25, the last three arms apply residual-output actions. There is no covariance or
RN_DPO extension because the current hypothesis is action separation, not another
policy family. REPEAT5 remains the repeated-update/cost control.

Three seeds fixed to 20260907, 17011, 29009; two full orders per arm. Exactly
36 trajectories x1,951 arrivals =70,236 arrivals. No extra seeds, no performance
pruning, no early label access, no online accuracy feedback. All six real GPU
engineering profiles must pass before formal trajectories begin. All registered
online workers must retire before CPU scoring starts. Each candidate starts from
the identical model/optimizer/augmentation RNG snapshot; only one state commits.

Five fixed contrasts: PG_RETAIN versus C, ANCHOR, REPEAT5 and GREEDY; GREEDY versus
RANDOM4. PG_RETAIN is the fixed primary and cannot be replaced retrospectively.
Use the R25 development gate: primary delta versus C >=0.3 pp, both order effects
positive, >=5/6 positive seed-order trajectories, image-weighted delta nonnegative,
worst seed-averaged domain/channel delta >=-2 pp, and positive mean effects versus
ANCHOR, REPEAT5 and GREEDY. Even passing is only a development signal.

Primary metric remains 512px hard Dice, equal domain/channel then equal order/seed,
empty/empty=1. Report image-weighted Dice, every domain/channel, all negative cells,
all five paired contrasts, content bootstrap intervals, costs and failures.
64px proposal ranges, ranking correlation and oracle regret are post-hoc diagnostics
only. Private proposal masks must not be published. Compare action separation with
R25 descriptively; cross-round cost/geometry differences prevent claiming a fully
isolated causal saturation ablation. Do not use positive content intervals as a
substitute for the gate or patient-independent inference.

## Exposure and boundaries

SEARCH (1,017 identities), legacy SEALED_REVIEW (678) and context (256) all come
from previously exposed development data. The legacy field name is retained for
scorer compatibility and denotes no new holdout. A tuned method requires suitable
independent validation and a novelty review before a paper-level superiority claim.
Patient linkage and ROI provenance remain UNKNOWN. No source images, new model
assets, future samples or online labels. FIFO memory and initial checkpoint remain
as in R25. Contextual bandit, not long-horizon RL; mechanisms are not paper replicas.

No cumulative GPU-worker time cap: JSON null. Cost is still recorded per attempt
and across rounds. Per-task timeouts derived from engineering profiles, 64 GiB
storage guard, and a seven-day supervisory safety timeout prevent hung processes;
these do not authorize extending the fixed matrix. Only registered GPUs4–7, with
adequate live memory and no interference with others. Preserve each failed attempt;
at most one evidence-backed transient I/O/network recovery per job. No retry for
numerical failure or weak performance. Never reset T0 or cost. Publish code,
protocol, aggregates and all outcomes via the configured GitHub proxy after each
completed round; preserve private data, identities, masks, checkpoints and paths.
