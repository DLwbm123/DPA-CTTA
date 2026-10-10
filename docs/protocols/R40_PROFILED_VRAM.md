# R40: technical recovery using measured worker VRAM

R39 ended with8 completed trajectories,1 pre-arrival memory-admission failure,
15 unrun trajectories and0 scores. No primary trajectory ran and no REVIEW
release occurred. Its report and costs were delivered and verified before R40.
Scientific result NOT_EVALUABLE. R39 T0/attempts/ledger are closed and unchanged.

## Concrete operational fault and recovery hypothesis

The inherited worker used default5GiB peak and required6GiB free, even after
same-round profiles measured less than1GiB. A subsequent check found5,884MiB
free on the affected GPU; exact failure-time memory was not logged. This was a
preflight refusal, not an observed CUDA OOM. The completed profiles were ignored
by formal admission. Hypothesis: consuming same-round measured peak reservation
with the existing max(1.2*peak,peak+512MiB) margin removes this false refusal while
retaining the required memory reserve. Concurrent usage can still change after
preflight, so this recovery may fail; no other user's process is changed.

R40 is explicit technical recovery of the previously untested R39 primary,
not a new scientific module or a response to scored negatives. It is justified
by the specific runtime fault, not a blind rerun. Use fresh full24 trajectories
without selective reuse of the8 unscored completed streams. Charge every old
and new failure/profile/attempt. No repeated labels or outcome-based retuning.

## Frozen implementation and matrix

The R39 method is unchanged: C/W/CW_LSO/CW_LSO_GC, three seeds[20260907,17011,29009],
two original orders,1,951 arrivals each:24 trajectories/46,824 formal rows.
Primary gradient-conflict selective trust cap0.0015, raw-gradient EMA0.9,
cosine<0, first/zero-norm steps uncapped, LSO0.1 and all losses/views/Adam states
unchanged. Fresh model/BN/Adam/RNG/gradient EMA each trajectory. No new candidate,
seed, dataset, RL, threshold/radius/EMA scan, teacher, fusion or source retraining.

Only operational change: opt in to use_profiled_worker_memory. Cold12-arrival
profiles retain the existing5GiB default; after all profiles PASS, formal workers
read their matching condition's current-round positive integer peak_reserved_bytes.
Missing/invalid/failed/mismatched profile refuses admission. Existing GPU UUID,
memory margin, storage, process neutrality and budget guards remain. Old rounds
without the flag retain original behavior. Initial GPU assignments selected from
currently available memory; this selection uses no scores and freezes at launch.

## Unchanged scoring and gate

All workers retire before separate CPU masks. Same SEARCH1017, historical REVIEW678
descriptive only; hard Dice2TP/(pred+GT), both-empty1, equal domains/channels,
three seeds/two orders. Success: primary-minus-C/W each>=0.3pp, each both orders
positive,>=5/6 trajectories positive, imageweighted nonnegative, worst seed-mean
domain/channel/order cell>=-2pp. Also primary-minus-CW_LSO mean>=-0.05pp. Fixed
primary, no relaxation. All negative cells and coupled-content bootstrap2000
retained. Patient linkage UNKNOWN; no independent generalization claim.
No ASSD/Brier/soft Dice. No REVIEW-based choice or score reuse across rounds.

Reuse7 R39-related CPU checks; additionally show empirical measurement admits
the observed5,884MiB scenario where the legacy fallback refuses, genuinely
insufficient memory still refuses, invalid profiles refuse and old/cold behavior
is unchanged. Profile all4 conditions, whole-matrix budget admission with20%
timing reserve. Record SEARCH conflict/cosine distribution/EMA-ready and cap
activation/scale/raw/actual displacement at completion.

## Cost and delivery

Original R40 T0 before GPU profiles;22h online/24h total/48 GPU-worker h/32GiB.
At most3 workers, neutral command lines, prescribed mounted storage. All GPU
failures/profile/formal attempts charged, preparation CPU separately recorded.
Closed prior4 rounds cost31.30581171128485 GPU-worker h (includes R39 failure).
Never rewrite old T0 or ledgers. No healthy restart, automatic scientific retry,
pruning or parameter changes; a new failure is preserved and reported.

Publish own code/protocol and all anonymous aggregates/failures/costs/audits on
experiment/r40-profiled-vram-v1 in DPA-CTTA, through explicit local proxy only.
Verify remote commit and anonymous report HTTP before DELIVERY. Exclude private
content/patient IDs and hashes, images, labels, per-content predictions/scores,
model states, private paths/logs and third-party PDFs. Hourly necessary monitoring
continues; only meaningful change notifies. Pause after frozen success and verified
delivery. Future scientific improvements still require concrete evidence and a
separately frozen bounded hypothesis; technical recovery is not scientific success.
