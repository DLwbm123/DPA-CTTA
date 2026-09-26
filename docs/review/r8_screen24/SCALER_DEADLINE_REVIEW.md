# Scaler deadline stop — 2026-09-25

**Superseded on 2026-09-26:** user explicitly authorized removal of timing gates. Runtime 544640f resumed the verified scaler snapshot. The proposal below is retained as historical evidence; no approval is outstanding. See TIMING_OVERRIDE.md and RUN_STATUS.md.


Status: **GLOBAL STOP; continuation not authorized or launched**.

The scaler exceeded its admitted 4,313-second worker deadline after completing 493 of 512 anchors. Queue and GPU workers have exited. Oracle 768/768, basis and capacity 128/128 are complete. Source model training and target jobs have not started. The global 60 GPU-hour / 16 GiB / 24-hour caps have not been reached.

The original profile timed scaler_anchor directly on the first source anchor, without the full physical journal and cumulative snapshot path. Actual end-to-end execution was slower than the admitted 1.3-margin estimate. The current evidence establishes an underestimated worker deadline; it does not isolate all timing overhead to NAS. This is a resource-stop event, not an infrastructure exception or numerical failure.

Read-only review validated both scaler snapshots, including identity, digest, source groups, complete result shapes, RNG state and physical prefix. Latest valid snapshot: anchor 493, 54,723 forwards. The uncommitted tail contains 62 additional forwards and is preserved; no prior scaler recovery exists.

## Concrete proposed continuation

Subject to explicit authorization to lift this resource stop once:

1. Preserve original queue/ledger stop and failed attempt records before any controlled continuation; retain its entire reservation in cost. Do not relabel the resource deadline as infrastructure.
2. Use the validated anchor-493 snapshot and replay only the uncommitted tail according to the existing equivalent-recovery semantics, with an explicitly recorded resource-stop exception.
3. Reserve 600 GPU-seconds, 3,000 forwards, zero backward/optimizer/VJP, and the existing 169,477,736-byte worker disk bound for the remaining 19 anchors and final sealing.
4. Continue the unchanged 10-source/40-target graph. Do not replay completed oracle/basis/capacity work, reset the original 24-hour clock, or change any global cap, seed, LR, algorithm or data order.

Conservative sum of retained costs, the proposed continuation and every pending worker reservation: 197,150.87 GPU-seconds (54.765 GPU-hours), 10,109,314,298 bytes (9.416 GiB), 2,642,940 forwards, 586,725 backwards, 576,551 optimizer steps, 1,376 VJPs. All are within the frozen aggregate caps. Pending non-scaler reservations total 34.346 GPU-hours.

Private full review: `private/scaler-deadline-review-2329fb6.json` under the R8 server root. No stop flag, runtime config, source snapshot or scientific result was altered by this review.
