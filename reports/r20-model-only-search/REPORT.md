# R20 startup status

**RUNNING — experiment and final result delivery are not complete.** Snapshot: 2026-10-05T12:42:01.611615+08:00.

The detached controller has admitted all 30 registered conditions, two compressed SEARCH orders each (60 trajectories). Four workers run concurrently on the authorized GPU pool. At the startup check, all four workers had committed predictions:

| Trajectory | Committed arrivals |
|---|---:|
| S1_G_s20260907_o0 | 163/384 |
| S1_G_s20260907_o1 | 163/384 |
| S1_T01_s20260907_o0 | 144/384 |
| S1_T01_s20260907_o1 | 148/384 |

T0 remains **2026-10-05 12:01:05.993950 +08:00**. Normal online computation ends by **2026-10-06 10:01:05.993950 +08:00**; absolute campaign deadline is **2026-10-06 12:01:05.993950 +08:00**. The cumulative cap is **48 GPU-worker hours**, with a maximum of four concurrent workers. Existing GPU activity does not prohibit launch when measured free VRAM plus margin is sufficient.

Twelve generated-input tests passed, including native/zero-module parity, non-accumulating adapter gradients, U first-step and V later-step learning, read-before-write prototype memory, exact next-image snapshot continuation, SEARCH-label isolation, and content-clustered bootstrap coverage. Eight real 4-warmup/8-timed-arrival profiles passed. On all 12 real G-profile images, logits, gradients, BN/Adam state and native RNG exactly matched the pinned native host. These are mechanical checks, not accuracy results.

The split was frozen before new labels: **1017 SEARCH**, **678 SEALED_REVIEW**, and **256 retained context arrivals**. The common 384-image screen contains 22 Drishti_GS, 121 ORIGA, 121 REFUGE and 120 REFUGE_Valid identities. Patient linkage and original crop-center provenance remain UNKNOWN. This review population was historically exposed; it is not an independent unseen or clinical test set.

Closed profile attempts consumed **161.335 GPU-worker seconds**. The accompanying startup ledger covers settled attempts; ongoing formal workers are additionally charged live by the controller. The directory byte count is measured; per-attempt disk usage is NA, not zero. Mechanical test-body time is recorded separately from scorer-worker wall time. Final cost and disk accounting will be exported after execution. CPU-only preflight discovery/deployment-metadata errors were corrected before GPU work and remain in MECHANICAL_TESTS.json; T0 was not reset.

The controller proceeds through whole-stage SEARCH scoring, diverse full-flow selection, budget-admitted optional extensions, frozen primary/control confirmation, and key ablations. Review labels remain closed until all formal workers retire and the comparison configuration is frozen. Float32 probabilities are retained for the new-seed primary comparisons; compressed masks do not stand in for soft metrics.

Source checkpoint, target data, raw predictions, features, prototype memory, terminal weights and identity mappings remain private. This publication contains executable source, registration and startup evidence. SEARCH/SEALED_REVIEW performance tables and final conclusions are **pending**, not zeros or negative findings. No periodic monitor or automatic R21 is created.

Execution code: `5b0bcb0f1b2ab71f6537a7080e0f73c2de74f7c2`. Runtime configuration identity: `56ebf41069d1b21fd9a8a77ff040bd527ce93d4c148bddb319e1ba366b6a4c74`.
