# R2 temporary-file race repair and explicit continuation

Update: the first continuation later hit a second NFS temporary-file variant.
See [NFS follow-up and revised accounting](NFS_REPAIR.md). The original accounting
below describes only the first continuation plan.

The original September 13 R2 matrix stopped after 15 complete trajectories,
576 records in `o2a4`, and four unstarted trajectories. A capacity scan observed
an atomic evidence writer's `.write-*` temporary between its file-type check and
its second stat. The writer removed that temporary, causing `FileNotFoundError`.
The original run remains `INCOMPLETE`; its records and failure evidence are retained.

The user explicitly requested repair and continuation. The external review covers
implementation `0d515328a6cc42d8e0c6a458265b41e41c454fa6`; this engineering amendment
is separately authorized by that request and does not claim a new external review.
Science, registration, dependencies, five arms, orders, seed, losses, optimizer,
data-access rules and per-trajectory state lifecycle are unchanged.

## Repair

One shared capacity scanner uses a single stat per file. It tolerates only a
disappearing `.write-*` temporary. Missing ordinary files, permission failures and
other storage errors still fail. R1/R2 workers, their shared supervisor and both
CPU publishers use this scanner.

The existing R2 entry accepts `--continue-from` only with an explicit private
continuation authorization. This implementation supports this single incident,
not recursive recovery or automatic retry. It verifies the original run, failure,
process inventory, smoke evidence, completion metadata and device rotation.

The new output directory contains fresh two-device smoke evidence and only
`o2a4`, `o3a1`, `o3a2`, `o3a3`, `o3a4`. Their original device assignments are
preserved. The 15 carried trajectories are read from the original directory with
their original code/run bindings. Neither their records nor the failed prefix are
rewritten. The failed trajectory restarts from the registered source checkpoint;
there is no saved adaptive checkpoint from which to continue its 576-step state.

CPU closeout checks all 20 selected trajectories with the existing scalar and
historical-control validators. It verifies both runtime provenances and excludes
the failed prefix from the formal table. A truncated carried result still fails
closeout and invalidates the new result pointer.

## Explicit accounting amendment

| Quantity | Count |
|---|---:|
| Selected formal scoring records | 39,020 |
| Preserved but excluded failed-prefix records | 576 |
| Formal trajectories executed by continuation | 5 |
| Formal scoring records executed by continuation | 9,755 |
| Additional smoke backward/Adam calls | 28 |
| Actual scoring visits across both attempts | 39,596 |
| Expected actual backward/Adam calls including both smoke batches | 39,652 |
| Expected actual forward calls including both smoke batches | 317,216 |

These physical totals exceed the original budget by 604 backward/Adam calls and
4,832 forward calls. They are disclosed as the cost of this user-authorized repair,
not represented as a pristine single-pass execution. The 24-hour active and wall
limits and 2-GiB output cap are reduced by original usage; the wall allowance also
includes the stopped interval. Each trajectory retains its two-hour limit.

## Verification

`CPU_TEST_LOG.txt`: 19 tests passed on the existing local Python/Torch environment.
The selected checks cover the race, permission/I/O failure propagation, skipped
job rotation, continuation authorization, mixed-run scalar closeout, inherited
truncation rejection, original R2 execution/scalar gates and owned-process cleanup.
No real images, checkpoint weights or GPU inference were used in these CPU checks.

Command: `PYTHONPATH=src:tests CUDA_VISIBLE_DEVICES='' python -m unittest test_r2_continuation.ContinuationChecks test_r2.ExecutionChecks test_r1_fixes.ProcessChecks -v`

At this source commit, GPU continuation is not yet claimed to have started. Its
actual launch receipt and runtime commit are recorded separately after deployment.
Raw per-image records, private asset mappings, host paths, device UUIDs and PIDs
remain private. A successful launch is not a completed experiment.
