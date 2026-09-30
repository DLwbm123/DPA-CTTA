# R10_CARRIER_DECISION_3H_V1 — admitted background run

Status: RUNNING; final source/target results are pending. No new training.

Execution revision 2 SHA: `da5edaf9714a1c1ee569958c6e4f333189bc8e7a`; base `2547c35f40ed6b401fd0b13935168ef72ceba161`.

Frozen revision-2 config digest: `09080af7940e5f17c2e2f76a396756e61594dedb7774a1580eba3e814696e335`.

T0 remains 2026-09-30 22:16:06.144414 Asia/Shanghai. Original package charge is 24757.142144730315 seconds: previous history/core charge 13234.584301998839 plus completed attribution actual wall 11522.557842731476. The remaining original 12h package is 18442.857855269685 seconds; the new cap is 10800 seconds. GPU-worker occupancy, reservations and already-closed report-wait time are not added as duplicate executed phases.

Normal computation deadline: October 1 00:16:06.144414; recovery cutoff 00:46:06.144414; absolute process-group exit 01:16:06.144414. Independent watchdog plus existing per-worker process-group cutoff cover descendants. Original source phase deadline remains fixed after the startup repair. At most one equivalent infrastructure target-job recovery; all physical attempts remain charged.

Revision 2 launched 22:42:20, 1574.375 seconds after T0, within the 30-minute preflight limit. Conservative admission with 1.3 safety factor: source paired validation 486.681 seconds; eight target trajectories plus independent scoring, loading/I/O and measured readout overhead 3218.338 seconds. No extra alpha selection, dataset, seed or training.

Seven checks passed: exact output-alpha endpoints and intervention location; native B FULL/reset state; intervention/cache identity; paired factorial interaction and score embargo; native visit-to-source-trace logging; owned child-process-group cutoff; image-only target-mask boundary. Real GPU revalidation performed 32 fixed source visits, eight per actual schedule mode. C0 same-path repeated-forward error, B_ZERO logits/probability error and mask mismatch are exactly zero. Both BN buffers/parameters and native alpha-one outputs remain identical. FULL/reset states match across output alpha. Preflight revision 2 cost: 480 actual backbone forwards, zero backward/optimizer/VJP calls, wall 45.442 seconds.

A startup engineering failure is retained, not hidden: revision 1 (`959311c`) stopped on its first source C0 visit because the trace builder duplicated the `visit` keyword. No target trajectory had begun. `source.0` consumed 16.643 seconds and one backbone forward. The repair changes source telemetry assembly and version/attempt accounting only; it does not change readout math, B recurrence, data, seeds or models. Revision 1 configuration/state/report, launch/start receipts and logs remain preserved in the private version ledger. Revalidation passed before revision-2 launch. Infrastructure-recovery allowance has not been consumed, and T0/stage/absolute deadlines were not reset.

Formal queue: five controls × 16 source-val episodes × 32 visits; four new target conditions × two original 1024-arrival manifests, 888 principal-eligible visits each. New target visit cap 8192, principal count 7104. Old N/G/C0/B short-stream results have verified compatible identities, seals, contents, order and eligibility and are reused. All new target metrics stay embargoed until terminal state. The new zero-probability digest is compared to the compatible retired old C0 seal; mismatch stops further carrier interpretation. Native B image state is z/d/counter, not R10 m/q/h.

The old target journal retains snapshots through visit 1000, so comparison with the original FULL state is labeled as a checkpoint comparison at that visit. New FULL/RESET state hashes cover every one of 1024 visits across their paired readout conditions; source paired state checks cover all 512 visits per control. Do not pretend an old visit-1024 state snapshot exists.

Previous post/SUP source validations and the 32-context writer diagnostic were read and retained; no writer counterfactual rerun. WARM-only source validation is unavailable in the prior core receipts. The new carrier-control source validations have no compatible prior complete result and are executed anew. Original R10 gain is available in saved logs; residual-relative-candidate diagnostics were not recorded and stay unavailable.

Only local code/report commits are authorized. No GitHub push or hourly monitor was created/resumed. Final reports and receipts remain pending.
