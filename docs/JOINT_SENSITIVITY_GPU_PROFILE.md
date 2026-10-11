# Frozen joint-sensitivity real-model profile only

Authorization: real-model GPU resource qualification only; physical GPU0/1,
preserve R44, no formal experiment. Frozen before GPU execution.

Compare unchanged CW_LSO and the CPU-qualified CW_LSO_JS at strength1, retaining
the joint acquisition/anatomical sensitivity formula without retuning. Original
ResUNet source checkpoint, batch1 RGB512, existing native normalization, seed
20260907. Reuse exactly the existing R44 profile's first12 previously authorized
images in order0; no source images, labels, new evaluation set, metrics, stored
predictions, model states or trajectory promotion. Every job initializes from the
same checkpoint. GPU0 runs base then JS; GPU1 runs JS then base. At most two GPU
workers. Only four profile tasks, never an online/full-flow task.

R44 retirement is a mandatory launch barrier: RUN_STATE COMPLETE, all24 jobs
COMPLETE, all worker/scorer process receipts inactive with no live matching PID,
and original watch/supervisor exited. Do not initialize CUDA or overlap profiling
with R44's GPU or CPU scoring work. Stop this queue if R44 is INCOMPLETE/BLOCKED.
Do not modify, stop, migrate or restart any R44 or unrelated process.

An independent immutable queue T0 covers at most24h resource waiting. Four jobs
each have a450s deadline, in two waves; total GPU-worker budget1800s including
model initialization, warmup and failures. Separate1GiB disk cap. Cold memory
admission requires6GiB free per authorized GPU, followed by the existing GPU UUID
and max(1.2*5GiB,5GiB+512MiB) worker guard. No automatic retry or budget reset.
Failure or incomplete profiles retain costs and do not authorize formal runs.

Each task performs4 warmup and8 timed arrivals. Synchronize CUDA before/after
each timed step; report end-to-end mean, median, P95, min/max seconds/image and
CUDA event elapsed. Event elapsed includes host dispatch gaps, not isolated
kernel compute. Report peak torch allocated/reserved memory including cold model
initialization, BN parameter count, physical10F/5B/1Adam (JS) versus8F/1B/1Adam
(base). Independently meter base backward, auxiliary VJP and optimizer counts.
For JS, aggregate evidence and acquisition-span rank plus verify metric-norm
contraction and frozen non-BN parameters. Nonfinite output/counter mismatch is
engineering failure, not a scientific negative or a trigger to tune parameters.

GPU0/1 may contain unrelated jobs; capture device utilization/process context
privately and report contention as a timing limitation. Same-device ratios and
reversed condition order mitigate, but do not remove, sharing/order confounds.
Twelve images cannot measure all future content-dependent performance cases.

Use a separate source snapshot/output root, neutral process entry/arguments and
existing guarded worker/attempt/ledger machinery. Publish only own source,
protocol, anonymous resource aggregates, all failures and cost summary after
actual completion; verify GitHub through the explicit local proxy. Raw data,
content identifiers/hashes, masks, predictions, model state and private paths/
logs remain private. Background startup or this registration is not a completed
GPU profile. No automatic next experiment or new monitoring automation.
