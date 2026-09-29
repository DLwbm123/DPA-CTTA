# R10 implementation and execution boundaries

Base: `13bd6a8cdf9c30a0a5ed7fa464c67e72301ac8c6`.
The supplied input specification is preserved verbatim. R10 uses its own namespace,
25-source/410-target graph, source selection lock, authorization, profile projection,
256-hour ledger, attempt receipts and final score release. R9 protocol and experiments are unchanged.

The frozen B mathematics and generic serialization/journals/metrics are reused.
The independent use/write networks, connected source PPO/SUP objectives, exact
complete-round RNG/optimizer/EMA snapshots, source-only schedule/cache and probes,
D0, checkpoint selection and deployment diagnostics are implemented in this namespace.

Synthetic checks exercise the real B solver, all eight training paths including
warmup, on-disk snapshot replay, write causality, true worker EIO evidence and
queue reconciliation through scoring, probability retirement and disk admission.
The reference package's math checks are run separately and are not CUDA proof.

Profile measures complete 32-round schedule blocks, so frozen-observation caching
is measured rather than assuming every candidate repeats a zero-modulation forward.
SUP recomputation and all native baseline backward/optimizer operations are charged.
Actual CUDA/profile PASS and a bound private authorization are required for launch;
the public disabled template remains false. No timer or hourly monitor is created.

The finite execution entry now chains a source-only profile child, recomputed admission,
profile-bound private authorization, serial queue and terminal report. It creates no monitor.
Failed source methods remain recorded; family selection uses only complete discovery pairs,
and unavailable dependent branches are marked BLOCKED while independent work continues.
18 connected CPU and supplied mathematical regressions passed before real-data deployment.

## Mixed precision repair

The first real profile stopped in SUP_SEQ before formal execution. A float32
writer `w=sigmoid(-1)` gives `(1-w)*float64(1)+w=1.0000000298023224` because
its complement was rounded in float32. The state update now casts w to the
persistent memory dtype before forming its complement. No clamp, tolerance,
seed, objective or matrix change is used; gradients through the cast remain live.
A saturated-memory regression reproduces the previous error and checks the fix.
Failed profile evidence is retained. Subsequent profile guards and full admission
include prior profile operations and time, within the original R10 limits.

## Native optimizer accounting repair

The next real profile recorded G as 9 forwards, 2 backwards and 2 optimizer
steps, while the independent native host recorded its one actual Adam update.
The shared physical meter counted both the GraTa orchestration wrapper and its
base Adam. It now excludes optimizer wrappers with a base optimizer and counts
the delegated base step through its own hook. Both R10 profiling and execution
use this meter; the frozen 9F/2B/1Adam contract is unchanged. A pinned native C/G
regression checks repeated steps, measured counts and exact optimizer/output
parity, plus direct AdamW accounting. Prior failed costs retain the conservative
double counts. Existing R9 deployments and ledgers are not modified.
