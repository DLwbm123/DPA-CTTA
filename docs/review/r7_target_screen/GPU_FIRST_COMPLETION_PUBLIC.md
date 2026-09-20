# R7 TARGET_SCREEN GPU-first completion record

The GPU-first TARGET_SCREEN run completed all 24 frozen jobs using three isolated order lanes (order 0, 1 and 4) on explicit `C_BASE_GPU_FP32_V1` FP32 execution. The frozen seed, arrival stream, recurrence, six SOURCE_PREP artifacts, adaptation semantics and 24-job matrix were unchanged.

- implementation binding: `c1763d00f4f29de1c168ccd75219199667478825`
- nominal calls observed: 122,913 backbone forwards, 5,853 backward calls and 5,853 Adam calls
- job completion: 24/24, including independent post-hoc scoring
- numerical policy: no AMP, TF32 or batching; retry/resume disabled
- execution-layer review: `USER_WAIVED` for the user-approved backend amendment; no external `PASS` is asserted
- scientific status: complete pending scientific review; no automatic winner or model-selection decision issued

Private receipts, target paths, checkpoints, raw logs and per-job result tables remain outside this public source-only delivery.
