# Descent-constrained shared sensitivity: real-model profile only

Frozen scope: compare R46 CW_LSO_JS_SHARED and the CPU-qualified
CW_LSO_JS_SHARED_DESCENT. No formal efficacy experiment, scoring, new image
cohort, parameter search, automatic retry or monitor. Source model, W/LSO,
strength1/ridge0.01 and fixed Adam are unchanged.

Use the existing source ResUNet checkpoint, batch1 RGB512 and the same12
previously authorized profile arrivals, seed20260907. No target labels or source
images. Each of four jobs starts a fresh source model: GPU0 SHARED then DESCENT;
GPU1 DESCENT then SHARED. Only physical0/1, at most two GPU-workers. All R46
online/scoring/watch/supervisor processes and30 complete jobs must retire first.
Do not intervene in other tasks or require exclusive GPUs.

Each job has4 warmup and8 timed arrivals. Synchronize CUDA around each arrival;
report wall and CUDA-event elapsed times, including host dispatch gaps. Peak
allocated/reserved memory includes model initialization and adaptation. Record
BN trainable parameter count and10F/4 auxiliary VJP+1 base backward/1Adam calls.
Also report constrained-step trigger and fallback counts across all12 arrivals;
the sample may not exercise every branch. No outputs, logits, predictions,
per-content traces, models or checkpoints are saved by this profile.

Cold admission retains5GiB peak plus the existing max(1.2*peak,peak+512MiB)
margin; the queue additionally requires>=6GiB free on each authorized GPU.
Independent T0 before this profile's GPU work,24h wall including any wait,
1800 GPU-worker seconds total,450s per job,1GiB disk. Preparation CPU and all
profile/failed GPU attempts remain recorded; original round ledgers are unchanged.
Any failure stops the fixed queue and is reported, without automatic retry or
formal launch.

SHARED retains its existing metric contraction check. DESCENT validates the
original/committed base-loss directional derivative with the CPU-frozen rounding
tolerance1e-5*max(abs(proposal_dot),1e-10), and preserves native counters/moments.
No threshold is adjusted from timing or outcomes. Each profile must PASS before
being described as completed; a process ID or partial timing is insufficient.

Compare DESCENT/SHARED within the same GPU, retain both device ratios rather
than pooling unlike devices. Shared-GPU contention and only8 timed arrivals
limit precision; counterbalanced order is not an isolated latency benchmark.
Memory/call validity and throughput do not establish segmentation improvement.
Anonymous summary, all failures/costs and own code/protocol will be delivered to
reports/joint-sensitivity-descent-profile on the matching GitHub branch through
the required explicit local proxy, with remote SHA and anonymous access verified.
