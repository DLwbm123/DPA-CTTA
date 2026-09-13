# NFS follow-up to the temporary-file race

At 18:40:15 Asia/Shanghai on September 13, continuation
`a8693fe3eab44ddb802c6de6c0caafda` stopped. The capacity scanner tolerated removed
`.write-*` files but omitted NFS `.nfs<hex>` silly-rename files. The supervisor
correctly terminated its two owned workers and preserved their prefixes: 1,048
records for `o2a4`, 1,053 for `o3a2`. No new complete trajectory was produced.
The first repair was therefore incomplete; its startup snapshot was not evidence
of eventual success.

The shared scanner now also tolerates ENOENT for the exact `.nfs<hex>` name form.
It uses lstat semantics so internal report symlinks are neither double-counted
nor treated as missing ordinary files after result invalidation. Permission,
I/O, and missing ordinary-file errors still propagate.

The user's existing repair-and-continue authorization is carried forward to this
correction of the same IO incident. Both failed output directories stay immutable.
The continuation entry independently validates the original 15 complete jobs and
the NFS-failed attempt, including its two-device smoke, owned-process termination,
partial scalar records and bindings. Only the same five pending trajectories run
from their registered initial checkpoint in another fresh output directory.
No automatic retry loop or unrelated recovery path was added.

## Updated physical accounting

- Selected formal records: 39,020, unchanged.
- Excluded records across failed attempts: 576 + 1,048 + 1,053 = 2,677.
- Recorded scoring visits including a successful new attempt: 41,697.
- Three two-device smoke batches in total: 84 backward/Adam and 672 forward calls.
- Expected actual backward/Adam calls: **41,781 to 41,783**.
- Expected actual forward calls: **334,248 to 334,264**.

The ranges reflect up to one unrecorded in-flight visit per terminated worker.
There are no adaptive checkpoints to reconstruct those exact physical counts;
the report must not call the lower bound an exact total. These additional attempts
are disclosed above the original frozen compute budget. Cumulative storage,
active-time and wall-time caps include both prior attempts; wall time also includes
the intervening stopped periods. Scientific settings remain unchanged.

## Checks and limitations

The existing CPU suite plus updated regression checks covers `.write-*`, `.nfs*`,
symlink aliases, failure propagation, the two failed attempts' accounting, retained
source bindings and finite dispatch. The local probe initially had an intermittent
existing EIO process-reaping assertion failure; an immediate focused rerun passed.
The final local run is recorded separately. No full-model or real-data GPU work
was performed by these tests. Server/NFS startup evidence is recorded after actual
deployment, not inferred from local test success.

## Verified second continuation startup

- Runtime: `1560f28d41dda5665404ececbefc4f0a7c69d192`, clean separate checkout.
- Run: `6bb3ccee255243a684c95975ba2671bc`.
- Started: 2026-09-13 18:53:26 Asia/Shanghai, detached on the same two devices.
- Checked: 18:54:37. Both mechanical smokes passed; `o2a4` and `o3a2` each had
  67 records, beyond their first periodic capacity scan. No failure artifact or
  launcher error was present. Parent and both workers were alive with neutral
  process arguments and the expected device assignment.
- Target-server checks: 19 tests passed in 16.910 seconds. A separate real NFS
  probe held an unlinked file open, observed its `.nfs` rename, then closed it
  during stat and verified the scanner handled its disappearance. See
  [server verification log](NFS_SERVER_TEST_LOG.txt).
- CPU source checks: [19 local tests passed](NFS_CPU_TEST_LOG.txt).

The startup pipeline rate was about 0.59 seconds per record. The longest device
queue has three trajectories, giving an approximate 19:55–20:05 finish including
CPU summary if no further interruption occurs. This is a dated startup estimate,
not a monitoring promise or completion claim. Final public result delivery is
still pending. Both failed attempts and their private records remain preserved.
