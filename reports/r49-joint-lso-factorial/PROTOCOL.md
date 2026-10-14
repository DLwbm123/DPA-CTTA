# R49: C-background LSO x shared sensitivity

Frozen before same-round GPU profiles, trajectories or labels. The user authorized
starting the previously agreed five-condition experiment after CPU wiring checks.
Only this round is authorized; no automatic subsequent experiment, retry, sweep,
new candidate/seed, data relaxation, retrospective primary change or tuning.

## Question and exact fixed conditions

Does the R46 shared sensitivity mechanism improve plain C on its own, and does
its incremental effect shrink or become harmful when LSO is present? This is a
C-background factorial experiment. It does not identify the interaction under
W-enabled consistency and does not establish an identical internal function.
CW_LSO stays an unchanged strong control, not an artificially weakened baseline.

| ID | Consistency loss | LSO | J | Calls per image |
|---|---|---|---|---|
| C | original unweighted C | off | off | 8F/1B/1Adam |
| C_LSO | original unweighted C | 0.1 | off | 8F/1B/1Adam |
| C_JS_SHARED | original unweighted C | off | R46 SHARED | 10F/5B/1Adam |
| C_LSO_JS_SHARED | original unweighted C | 0.1 | R46 SHARED | 10F/5B/1Adam |
| CW_LSO | original C+W | original0.1 | off | 8F/1B/1Adam |

Use the CPU-qualified factorial.Host/candidates from bd451797c73071f2b1cc9a138ee276d01e46dbc9.
The first four arms contain no W component in the consistency loss. C-background
LSO retains the original detached reliability (temperature0.1/boundary boost2)
inside its branch, so C_LSO=unweighted C BCE+0.1 original LSO. Combined loss uses
one backward. CW_LSO config/loss are delegated exactly to the historical entry.
Keep the source checkpoint, BN-affine-only C host, Adam/LR1e-4/moments/step count,
six weak views, strong augmentation, current-only detached pseudo-targets,
preprocessing and original final prediction unchanged.

J is R46, not the R48 descent-constrained package: fixed Sensor/structure evidence,
two imaging probes (contrast1.10; channel-logit shift0.05/0/-0.05), four VJPs,
strength1/ridge0.01, Q=orth([X/||X||,Y/||Y||]), rank<=4, original Gram cutoff
max(1e-12,1e-6*largest eigenvalue). H is the actual post-Adam diagonal metric.
Use the existing joint filter and parameter write with metric-contraction guard.
Missing blocks/rank0 retain exact native proposal; J=0 exactly follows its base
host. No DESCENT/commit repair, GC/GP/GDP/SCNP/SE/RL, new loss/model/permanent
memory, domain/label actions, extra optimizer step or output fusion.

## Matrix, resources and admission

Original seeds20260907/17011/29009, two original orders,1951 arrivals each:
30 fresh trajectories,58530 independent CPU score rows. Initialize every arm
from the original source; do not borrow completed controls from prior rounds or
promote profile prefixes. No scoring until all online workers have retired.

Only physical GPU0/1, at most2 GPU workers, sharing allowed; do not intervene in
other tasks. Verify R48 terminal delivery and retirement first. All five matching
12-arrival profiles must PASS (4 warmup+8 timed); whole30-job admission must pass.
No profile pruning or candidate changes based on speed. Workers use the matching
same-round PASS peak_reserved_bytes with free>=max(1.2peak,peak+512MiB); missing or
invalid profile refuses; cold-profile5GiB guard remains. Preparation freezes the
actual GPU UUIDs. All visible process entries are neutral.

Independent T0 immediately before first GPU work;22h online/24h wall/48GPU-worker h/
32GiB. All profiles/initialization/failures count, no reset. Previous13 closed or
stopped rounds cost108.22563222454656 GPU-worker h; isolated profiles668.9714405536652s;
aggregate before R49=108.41145762470036h. Prior CPU wiring qualification elapsed
22.42775321006775s, GPU0, including the retained test-only RNG-format assertion
failure. Record this round's CPU preparation separately; do not alter old ledgers.
Approximate runtime comes from same-round profiles, not a promised ETA.

## Frozen evaluation

Primary is C_JS_SHARED minus C on SEARCH. Require >=0.1pp macro Dice, paired
95% CI lower>0, both order means positive,>=5/6 seed/order trajectories positive,
imageweighted>=0,worst seed-mean domain/channel/order cell>=-2pp. This is a new
standalone mechanism question, not a relaxation or reinterpretation of old gates.
No secondary arm may replace this primary. All controls and negative cells remain.

Prespecified paired contrasts in both SEARCH and descriptive REVIEW:
C_JS_SHARED-C, C_LSO-C, C_LSO_JS_SHARED-C_LSO,
C_LSO_JS_SHARED-C_JS_SHARED, C_JS_SHARED-CW_LSO,
C_LSO_JS_SHARED-CW_LSO, CW_LSO-C. Report macro/imageweighted/each order/each seed,
all domain-channel cells and paired intervals. Strong-control comparisons do not
turn a gain over C into a gain over CW_LSO. Keep all positive and negative results.

Interaction I=[(C+LSO+J)-(C+LSO)]-[(C+J)-C]. Compute content-level differences of
all four arms before resampling, coupling every seed/order and all four arms.
Use the original domain-stratified2000-repeat content-cluster bootstrap with seed
20261005, equal-domain/channel macro Dice then equal seeds/orders. Report I and
its CI, conditional J effect, LSO effects and all interaction-negative cells.
The scoring-only signed contrasts are not predictions or runnable extra arms.

Interpret only after complete coverage. If standalone primary fails: no registered
standalone signal; do not blame LSO for that failure. If standalone passes and
conditional J CI upper<0: interference-compatible. If standalone passes and I CI
upper<0: diminishing-returns-compatible; additionally conditional J's entire CI
within[-0.05,+0.05]pp permits redundancy-compatible. If J passes its registered
positive gate in both backgrounds: J benefit in both backgrounds, without claiming
super-additivity. Otherwise interaction inconclusive. These are operational
interpretations, not proof of mechanistic overlap. The primary is confirmatory
within this exposed development test; secondary intervals are descriptive and
are not multiplicity-corrected evidence of multiple independent discoveries.

SEARCH/REVIEW have historical exposure. REVIEW never tunes/selects this method
and is not independent generalization. Patient linkage UNKNOWN. Dice/empty-mask
convention unchanged; ASSD/Brier/soft Dice uncomputed. Structural proxies and
metric contraction do not demonstrate anatomical or clinical protection.

## Execution, diagnosis and delivery

Reuse the existing guarded background supervisor, per-attempt receipts, costs,
filesystem current-image/model-only access boundary, profiled memory guard,
online-retirement barrier and independent CPU scorers. No checkpoint recovery,
scientific/technical automatic retry, expanded matrix or budget reset. On failure
retain partial outputs/all costs; incomplete/unscored means NOT_EVALUABLE, not0.

CPU preflight verifies config/dispatch, four-arm interaction formula and coupled
bootstrap, interpretation gates, incomplete reports, the full five-arm bitwise
parity/counters/RNG/restore, original shared dense solution and VRAM admission.
Record J rank/eigenvalues/capture/span energy/update change/norms, OD/OC evidence,
LSO/base losses and BN gradients/updates. Region-logit gradients remain as provided
by the existing loss hooks; do not fabricate missing measurements for C-only arms.

The supervisor handles profiling, fixed matrix, scoring and anonymous aggregation
without depending on this chat. No recurring monitor is created by this launch
request. A later status query can trigger the actual-completion audit/delivery;
further scheduled monitoring requires explicit authorization. No next round.

At actual termination verify watch/supervisor/online/scorers retired and all30
unique jobs/30 CPU scorers/58530 rows plus all attempts. Publish own source and
anonymous aggregate complete positives/negatives, interaction CI, costs and audit
to reports/r49-joint-lso-factorial on experiment/r49-joint-lso-factorial-v1 in
DLwbm123/DPA-CTTA. Every GitHub operation uses explicit local proxy, cleared bypass,
no direct fallback/force/overwrite. Verify remote SHA and anonymous report HTTP
before local/remote DELIVERY/CAMPAIGN closure. Startup validation is not completion.
Never publish images/labels/per-content predictions or scores, patient/content
identifiers including hashes, model states, private paths/logs or third-party PDFs.
