# Live-directory capacity scan repair

The second continuation exited at 19:14:14 Asia/Shanghai on September 13.
`o2a4` completed all 1,951 records; `o3a2` failed after 1,408 records because
`output_bytes()` could no longer stat `usage.json` after enumerating it. This was
a capacity-scan exception, not a failed read of the telemetry JSON contents.
The other three pending jobs never started. Total completed trajectories: 16/20.

The prior two fixes were incomplete: filename-specific exceptions do not make a
live directory snapshot transactional. Capacity measurement now skips ENOENT for
any entry that disappears between enumeration and stat, retaining single-lstat
counting of regular files. File names do not control this behavior. Permission and
other I/O errors still propagate; required evidence and asset reads retain their
independent validation. Nothing treats missing required results as complete.

The worker's redundant per-visit read of shared `usage.json` is also removed.
The owning supervisor already enforces cumulative active and wall limits from
its monotonic-clock state, and each worker still enforces its trajectory deadline.
Telemetry remains available for progress queries. This changes no method, seed,
order, optimizer, model, data access, or frozen scientific setting.

## Continuation and accounting

The original 15 complete trajectories remain at their original source; the newly
complete `o2a4` is referenced from the second continuation with its original
binding. All three failed attempts stay intact. Only `o3a1`, `o3a2`, `o3a3`, and
`o3a4` run, on their original devices, from the registered initial checkpoint.
The existing continuation validator was extended to carry this known completed
trajectory; no unbounded retry loop or arbitrary job-selection interface exists.

- Selected formal records: 39,020, unchanged.
- Records in the 16 complete trajectories: 31,216.
- Remaining formal records: 7,804.
- Excluded failed prefixes: 576 + 1,048 + 1,053 + 1,408 = 4,085 records.
- Recorded scoring visits after successful completion: 43,105.
- Four two-device smoke batches: 112 backward/Adam calls, 896 forward calls.
- Expected actual backward/Adam calls: 43,217–43,219.
- Expected actual forward calls: 345,736–345,752.

The ranges retain the two workers' possible unrecorded in-flight visits from the
first continuation; the latest worker failure has exact physical counters matching
its saved prefix. Resource caps account for every prior attempt and stopped time.
Repeated work remains explicitly above the original compute budget.

## Validation

The CPU checks now cover arbitrary disappearing names including `usage.json`,
symlink handling, permission/I/O failures, worker deadlines without telemetry reads,
finite parent-enforced caps, and full scalar closeout across the three prior
attempts plus a final synthetic continuation. Both complete-result origins are
checked without rewriting bindings. Local and target runtime evidence are recorded
with this repair; launch status is appended only after actual verification.
