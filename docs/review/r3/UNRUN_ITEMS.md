# Unrun items and authorization boundary

- No GPU query, allocation, kernel, smoke or formal experiment was run in Stage I.
- No real target RGB or mask was opened. Only the existing registration JSON was read to bind the new metadata stream.
- No supplied source checkpoint was opened. Full ResUNet34 tests used programmatically initialized weights; an additional synthetic test scales its head weights to exercise reliable-region/VJP paths.
- No source RGB, source mask, source query, proxy, prototype or source training was accessed or performed.
- No formal scoring record was produced. The 85-job dry-run describes complete 1,951-visit trajectories; it does not execute them. CPU closeout tests use 12 fabricated scalar records per job and label them as fixtures.
- No persistent session, background waiter, monitor, automation, retry queue or task to run after approval was created.
- GPU memory, latency, CUDA numerical behavior and performance on the supplied checkpoint/data remain unverified. The 4 GiB free-memory threshold in the future adapter is inherited baseline headroom, not measured R3 peak usage; actual device smoke must validate it after fresh authorization.
- External review remains pending. Execution defaults remain disabled, with no approved code, science, stream, devices or 85-trajectory scope. No self-issued review conclusion or execution authorization exists.

The existing IO regression creates tiny synthetic PNGs and a tiny synthetic weight file in a fresh test directory. These procedural fixtures are permitted CPU test inputs and are distinct from the real target and supplied source checkpoint. Test guards block asset paths outside the fixture tree, and block CUDA initialization.

The only completion status for this delivery is `R3_IMPLEMENTATION_READY_FOR_REVIEW`. Stop at the review handoff. The code includes future thin execution and scalar-closeout wiring so the external reviewer can inspect it at the same implementation SHA; its presence is not permission to invoke it.
