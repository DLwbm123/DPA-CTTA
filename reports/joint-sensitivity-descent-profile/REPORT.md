# Descent-constrained shared sensitivity: real-model GPU profile only

Status: **COMPLETE_PROFILE_ONLY**. Formal experiment started: false.

Original source ResUNet, batch1 RGB512, BN-affine R46 SHARED versus DESCENT. GPU0 SHARED then DESCENT; GPU1 reverse. Fresh source initialization per job, same12 prior authorized arrivals and seed20260907;4 warmup+8 timed. No labels, predictions or checkpoints saved.

| GPU | Condition | Mean s/image | P95 s/image | Peak allocated MiB | Peak reserved MiB |
|---|---|---:|---:|---:|---:|
| 0 | CW_LSO_JS_SHARED | 1.3393 | 1.5024 | 816.9 | 896.0 |
| 0 | CW_LSO_JS_SHARED_DESCENT | 1.3523 | 1.5083 | 816.9 | 898.0 |
| 1 | CW_LSO_JS_SHARED | 1.3305 | 1.4904 | 816.9 | 896.0 |
| 1 | CW_LSO_JS_SHARED_DESCENT | 1.3236 | 1.4822 | 816.9 | 898.0 |

GPU0: DESCENT/SHARED wall time 1.010x; allocated 1.000x; reserved 1.002x.

GPU1: DESCENT/SHARED wall time 0.995x; allocated 1.000x; reserved 1.002x.

Charged GPU-worker seconds, including initialization and failures: 123.630.

R46 must fully retire before this queue initializes CUDA. Its configuration, artifacts, workers and budget are not modified. Other users may share these GPUs; contention and eight timed images limit transfer of the latency estimate. All profiles/failures remain reported. No efficacy gate, new seed matrix, promotion or automatic formal run.

## Comparison and limits

Four/four real-model profiles PASS;48 total arrivals,32 timed. Same-device
DESCENT/SHARED mean wall ratios are1.009750 (GPU0) and0.994854 (GPU1), approximately
+0.975% and-0.515%; CUDA-event ratios1.009790/0.994826 agree closely. Opposite
signed tiny differences with shared-GPU contention and only8 timed arrivals do
not establish a precise zero-cost or speedup claim. No substantial latency
increase was observed in this bounded sample.

Peak allocated is816.874512MiB for both methods on both GPUs. Reserved changes
896→898MiB (+2MiB,+0.2232%). This is allocator-level observed peak behavior, not
proof that the extra gradient/vector algebra needs no memory. Each model has
19,136 trainable BN-affine parameters; calls remain10 forwards/four auxiliary
VJPs/one base backward/one Adam per arrival.

Each DESCENT profile activates the constraint11/12 times, with zero non-descent
or degenerate fallbacks. The remaining step needs no active correction. Native
calls and actual committed first-order loss checks pass with the pre-frozen CPU
rounding tolerance; no tolerance, parameter or scientific criterion was changed
from GPU results. The sample exercises active and inactive cases; it does not
measure efficacy or the frequency on a full stream.

All profile workers and queue retired. R46's30 jobs and old processes retired
before CUDA initialization. Only physical GPU0/1 were assigned; no other tasks
were modified. No online trajectory, target label, source image, scoring,
prediction/checkpoint output or automatic next experiment was created.

## Failure and cost accounting

GPU-worker cost123.62959313392639 seconds (0.034341553648312885h), including model
initialization and four complete attempts; zero GPU profile failures or retries.
The profile budget remains1800 GPU-worker seconds/450s per job/24h wall/1GiB disk.
Original T0 is2026-10-13T02:43:50.872104+00:00 (Beijing10:43:50) and was not reset.

Local profile-integration CPU checks11/11 PASS in5.148s; server CPU28/28 PASS,
19.034642457962036s elapsed. Initial queue dispatch exited on CPU before config
admission because the new campaign identifier differed from the retained runtime
identifier. No execution marker, CUDA initialization or GPU attempt existed.
Original config/launch/log evidence is retained privately. Only the metadata ID
was corrected to the retained profile-runtime ID, with unchanged model, method,
images, seed, ordering, source revision, T0 and budgets. The failed CPU-dispatch
duration was not instrumented and is UNKNOWN, not zero. There was one diagnosed
CPU-entry repair and no re-executed profile job.

Previously delivered experiment/profile aggregate91.70541420506107 GPU-worker h
plus this profile gives91.73975575870938h. No efficacy claim or full-matrix budget
admission follows from these resource measurements. Formal experiments remain
unstarted and require separate authorization.
