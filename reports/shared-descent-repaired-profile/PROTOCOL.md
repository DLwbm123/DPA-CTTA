# Repaired shared-descent real-model profile: fixed, no formal experiment

User authorized a bounded real-source-model GPU profile of the CPU-qualified
commit repair, original tolerance, fallback frequency, derivatives and overhead.
Only physical GPU0/1, at most two workers, sharing allowed; no interference with
other jobs. R47 has stopped and all its processes must retire before this queue.
Its immutable source, attempts, outputs, T0 and costs are not modified.

## Fixed conditions and inputs

Compare R46 CW_LSO_JS_SHARED with CW_LSO_JS_SHARED_DESCENT containing the commit
repair qualified in SHARED_DESCENT_NUMERICS_CPU.md. Preserve the fixed original
loss/probes/Adam/LR/strength1/ridge0.01, no retuning or added network passes.
Original source ResUNet checkpoint, batch1 RGB512, BN-affine updates, seed20260907.

Use exactly the first64 images of each of the two already authorized R47 online
orders, not new evaluation data, labels or a selected hard-image subset. This
prefix reaches the former failing steps16/42. Fresh source initialization per
job. Four warmup images and60 timed images per job, all64 checked numerically.
Both GPUs run both conditions on both orders: eight jobs,512 total arrivals,
480 timed arrivals. GPU0 SHARED then repaired; GPU1 reverse; repeat for order1.
No scoring, mask access, prediction persistence, model-state export or promotion
of these prefixes into formal trajectories. This is resource/numerical evidence.

## Frozen numerical and cost checks

Use the same guard: for a negative native derivative,

    actual - proposal <= 1e-5*max(abs(proposal),1e-10).

All derivatives must be finite. Keep the closed-form proposed update if its
representable commit passes; otherwise restore native Adam parameters exactly,
without another step/forward/backward. Keep native non-descent/rank0 behavior.
Calls are10F/5B/1Adam in both conditions; unchanged moments and step counts.
Record constraint triggers, native non-descent fallbacks, commit-repair count
and reasons, pre-repair versus final derivatives, gap and tolerance utilization.
K eigenvalues and local proxy derivatives do not prove Dice/anatomical protection.

Derivative steps are stored privately with visit counters and no content IDs;
public output is anonymous per-condition/GPU/order aggregates with min/median/
mean/P95/max and counts. Failures persist exact available derivative/visit fields
in receipts and private logs. Do not loosen tolerance or automatically retry.
Any failure stops later batches and is delivered with original cost evidence.

Independent T0 before this profile's GPU work; wall limit2h, total3600GPU-worker
seconds including initialization/failure, per-job450s, disk1GiB. Explicit neutral
entries/child commands, NAS-only storage. Require prior R47 terminal status
INCOMPLETE with exactly24 COMPLETE/2 FAILED_NUMERICAL/4 NOT_RUN and all processes
retired; this is an explicit stopped-predecessor admission, not pretending it
completed its scientific matrix. Physical0/1 UUID and6GiB free admission remain.

## Report and authority boundary

Report all eight attempted/unattempted jobs, retirement/arrival coverage,
original-tolerance violation count, repair/other fallback rates, derivative
distributions, four matched GPU/order ratios, CUDA-event and wall timing,
allocated/reserved memory and total costs. Timings include host dispatch and
shared-device contention; the prefixes do not guarantee all future updates.
Passing requires all8 receipts PASS,64 arrivals each, expected physical counts,
zero final guard violations, original resource limits and full retirement.
Passing is not permission to begin formal experiments, enlarge the profile,
change parameters or start a monitor. No next scientific round is authorized.

Publish own code/protocol and anonymous aggregates/report on the matching
feature/shared-descent-repaired-profile branch in DLwbm123/DPA-CTTA, using explicit
local proxy for every GitHub operation. Verify remote SHA and anonymous report
access; preserve private raw traces/checkpoint/input paths/content identities.
No third-party PDF, image, label, per-content prediction/score or model state.
