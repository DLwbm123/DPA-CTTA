# Real-model joint-sensitivity profile: queued, not measured

Verified at 2026-10-11 15:33:29 UTC. Status: **WAITING_R44_RETIREMENT**.
No CUDA initialization by the waiting process, no GPU profile attempt, charged
GPU-worker seconds0. Formal experiment started: **false**.

R44 remains RUNNING_FIXED_MATRIX:12 complete,2 running,10 not yet run. This queue
does not modify R44. GPU0/1 profiling begins only after all24 R44 jobs complete,
workers/scorers retire and its original watch/supervisor exit. It stops if R44
fails or the24h resource-wait cap expires. This is a background startup receipt,
not a GPU benchmark result or a claim of admission to a formal experiment.

## Frozen profile

- Real source ResUNet checkpoint, batch1 RGB512, unchanged CW_LSO and CW_LSO_JS1.
- Only physical GPU0/1. Two profile workers maximum; base then JS onGPU0, reversed
  order onGPU1. Each job starts from the same source checkpoint and seed20260907.
- Same12 previously authorized profile images,4 warmup+8 timed. No labels,
  additional evaluation images, predictions or model snapshots saved.
- Measure synchronized end-to-end latency, CUDA event elapsed, torch peak
  allocated/reserved memory, physical calls, BN ownership and sensitivity rank.
- Four450s tasks; total1800 GPU-worker seconds including initialization/failures,
  separate1GiB storage cap, no retry, promotion, tuning or automatic formal run.
- All timing estimates remain pending. Other users sharing GPU0/1 can affect
  measured latency after R44 retires; event elapsed is not isolated kernel time.

## CPU preparation

Local retirement/aggregation/formal-mode checks:4/4 PASS. Server compatibility
checks for joint update, profile guards and CW_LSO controls:12/12 PASS in
8.815326 seconds on PyTorch2.6.0+cu124, explicitly CPU, two threads; GPU cost0.

Two CPU preparation imports failed before CUDA initialization because the source
archive omitted existing runtime static assets. The first lacked an existing
protocol JSON; its dispatch CPU elapsed was not instrumented and remains unknown,
not zero. The second lacked the tracked non-Python source protocol asset, charging
15.175003 CPU seconds and0 GPU seconds. The static import dependencies were
identified and supplied, failures preserved, and the subsequent12 checks passed.
No algorithm change, GPU retry, R44 change or T0/budget reset occurred.

Deployed source:77776e25e63cc24b7328fc97179e1539f3d807e1 on
feature/cw-lso-joint-sensitivity. Specification:docs/JOINT_SENSITIVITY_GPU_PROFILE.md.
The independent output/attempt ledger preserves the original queue T0
2026-10-11T15:27:47.445401+00:00. Scientific results and segmentation gains are
outside this profile's scope; benchmark results and their delivery remain pending.
