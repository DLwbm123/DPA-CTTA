# Review repair: 8f3222a → current commit

This document accompanies the new commit on `experiment/r9-current-first-v1`.
Reviewed baseline: `8f3222a8e897a2e03122927242800b9df5a75064`.
Authority: user requested these implementation repairs; **execution remains disabled**.
No private assets, real GPU operations, real profile, remote experiment or hourly monitor.

## Fixed scientific choice

`lr_source_policy=first_two_mean`, seeds **20260924 and 20260925**.
For each gradient arm, compute each seed's `0.5*mean(mode Dice)+0.5*min(mode Dice)`,
then average the two scores equally. Select one global LR from the original three-point
grid, breaking ties within 1e-8 toward the smaller LR. Use it for all deployment seeds
and orders. The four-visit source-cal schedule remains positions 0/8/16/24, 64 episodes,
16 per mode; no intermediate visits are claimed. Other policies are no longer accepted.

The imported scientific matrix/plan bytes, 43 source jobs, 610+161 target slots,
16k fit definition, checkpoint/calibration/selection, gradient algorithms, seeds/orders,
and internal scoring with end-of-round release are preserved. R8 files are unchanged.

## 1. Resource projection and finite recovery

The old unconditional x2 envelope is removed. `profile.node_units()` expands the same
complete graph. Every node budget must cover its measured unit costs; first attempts
reserve the sum of full node budgets. The additional compute reserve is the sum of the
three largest full node budgets **for each resource separately**, including CPU nodes
where applicable. These can be different nodes per resource, making the bound conservative.

Protocol and queue freeze **at most three recovered jobs globally, one extra attempt
per job**. A target's online and scoring phases share its job allowance. The ledger
atomically claims an allowance against an immutable failed receipt before RETRY;
claims survive restart and are never refunded. A repeated failure is terminal; a fourth
job cannot retry. Numerical failures cannot claim recovery; identity/isolation/resource
failures stop execution. Source/target snapshot equivalence checks remain required.

Disk projection uses the full retained-output bound, one maximal LONG10 temporary
probability file, and three largest additional retained-output bounds. The retained
bound must cover all node disk budgets minus temporary prediction files. Recovery uses
the same probability file, so admission credits the evidenced existing prefix rather
than reserving a second complete file. Failed reservations and operation history remain.

Caps are unchanged: 512 GPU-worker hours, 64 GiB, 32M forwards, 3M backwards,
3M optimizer steps, 4096 VJP calls. Per-task time estimates remain soft; actual aggregate
caps remain hard. This does not promise that a real measured envelope will fit.
`PROFILE.unmeasured.json` is still NOT_MEASURED; no PASS flag or measured cost is invented.

## 2. Recovery evidence interface

The real worker creates structured evidence bound to runtime identity, node, attempt,
and checkpoint/metadata file digests. The queue wraps it with the immutable failed
attempt receipt's canonical digest and exact path. A resumed worker checks the receipt,
class, identity, and unchanged snapshot manifest before constructing the recovery path.
The inherited TargetJournal then validates host/context/stream identity, full snapshot,
RNG/state and committed output/log prefixes. No dictionary-only bypass or weakened R8
validation was introduced. Failure to capture evidence remains a failed attempt.

## 3. Scheduler restart

Normal wait and restart call the same `verify_attempt` and `accept_receipt` functions.
A durable final receipt can settle an outstanding RUNNING ledger entry idempotently.
Retry eligibility comes from the persisted attempt chain and global recovery claims,
not an in-memory `prior` flag. A COMPLETE receipt advances dependencies without rerun;
score completion also restores the exact-trajectory reuse key after restart.

An absent final receipt does not imply success or permission to rerun. That case remains
fail-closed for process/evidence inspection. No new unconditional crash retry is added.

## 4. Storage across attempts

Cumulative compute accounting and current disk occupancy are separate. Failed attempt
compute reservations, observed costs and receipts stay intact. Filesystem occupancy is
reconciled at admission, settlement and verified retirement, with outstanding additional
space reserved only for an active attempt. It is not subtracted from a successful
attempt's growth counter. Existing probability prefixes are credited at retry admission.

The score seal authorizes retirement, an intent receipt precedes unlink, and a new ledger
instance can reconcile deletion after a restart. Repeated reconciliation records the
same evidence idempotently; it cannot double-subtract bytes or produce negative storage.
Occupancy is a metadata/size measurement, not a bytewise verification claim. The serial
queue and private output root are retained; no parallel storage guarantee is introduced.

## Connected fault injection (CPU synthetic only)

The new tests call actual queue, runtime worker/dispatch, ledger, TargetJournal, score and
retirement functions. Only private asset/model constructors, data readers, GPU hardware
checks and process transport are substituted. Workers produce their own failure/final
receipts; the test does not fabricate successful journal output or recovery evidence.

- EIO after visit 51, with a legitimate checkpoint at 50; worker final receipt written,
  then injected death before ledger settlement. Queue restart reconciles and retries once.
- Receipt-digest and checkpoint-manifest tampering rejected before resume.
- Recovery from visit 50 produces all 52 visits and the same probability digest as an
  uninterrupted real synthetic host; physical replay tail remains recorded.
- Actual CPU scoring, retirement-intent/unlink interruption, reopening ledger after
  unlink, repeated retirement/reconciliation, and next LONG10-sized reservation pass.
- Queue death after COMPLETE score receipt but before queue update restores completion
  and reuse key, without another worker execution.
- Separate successful receipt-before-settlement recovery is idempotent.
- Second infrastructure failure is terminal; numerical failure never retries; identity
  failure triggers global stop. Persistent three-job allowance rejects a fourth job.
- Full-graph projection checks enforce unchanged counts, per-node measured lower bounds,
  finite recovery reserve and fixed LR policy. Unbound real profiling is rejected before
  the supplied function can run.

Validation: 24 R9 tests + 6 inherited metadata/isolation tests; original plan arithmetic.
These are not actual ResUNet34/CUDA/private-data acceptance or scientific results.

## Remaining explicit authorizations

1. Bind actual assets/snapshots, GPU UUIDs and independent output root; obtain explicit
   `REAL_DATA_PROFILE_ONLY` authority tied to the new SHA and those bindings.
2. Measure production kernels/IO/scoring under that exact tracked runtime. Measurement
   wrappers verify SHA/inventory before invoking the kernel and attach binding evidence.
   No use of Screen24's old execution permission or synthetic timings for admission.
3. Verify the real full-matrix resource proof and all asset/regression requirements.
4. Obtain separate explicit full-round authorization binding this new SHA, asset digest,
   GPU UUIDs, output root, fixed protocol choices and measured profile digest.

Until then, `execution_authorized=false`. No automatic matrix reduction, cap increase,
launch, or hourly monitoring is part of this release.
