# R10_12H_CORE_V1 admission and launch

Status observed at 2026-09-30 03:24:07 UTC: **RUNNING, not complete**. Source WARM had completed 54/1000 physical updates, with an atomic checkpoint at update 50. POST, validation and all eight target trajectories were NOT_RUN at this observation. No result or difference is available yet.

Execution branch: `experiment/r10-12h-core-v1`.
Base SHA: `84837e997a60d7c98e4751b7ee737b1f074693b3`.
Execution SHA: `116f86b3421df4bfd582d9fb034b5443146ef6bf`.
Frozen private config digest: `3df1e5ba6ba36ea5b6d22c5cf887116ca657a4391fc436955ec2ee05dacdc02f`.

T0 is irrevocably 2026-09-30 03:07:54 UTC. Preflight and launch took 900.07 seconds, below the 1800-second cap. Normal computation ends by 11:07:54 UTC; hard compute deadline is 12:07:54 UTC; final closing deadline is 13:07:54 UTC. Historical conservative accounted cost is 6064.481842329726 seconds. New wall cap is 36000 seconds. Existing training assets carry their original offline preparation cost; only their reuse is charged here.

Chosen tier: **B, budgeted variant**, 1000 WARM optimizer updates and 1024 GR_RET_EMA sampling rounds (2048 post optimizer updates), preserving original objectives and hyperparameters and mapping cosine endpoints to the selected lengths. Source validation uses the fixed 16 complete episodes in PROTOCOL.md, without checkpoint selection. Tier A exceeded the source-stage budget; B was the first feasible tier. No alternate tier was trained.

| Quantity | Frozen value |
|---|---:|
| Source-stage conservative projection | 6075.63 s / 1.688 h |
| All targets + scoring/I/O projection | 7069.36 s / 1.964 h |
| GPU workers at a time | 1 |
| Visits per order per method | 1024 |
| Principal scored visits per order per method | 888 |
| Total planned visits | 8192 |
| Total planned principal scored visits | 7104 |
| Policy / native G seed | 20260924 / 20260907 |

Both target orders contain the same selected content set. The original within-order relative order is retained. Quotas preserve domain and original scoring-subset proportions by largest remainders. Registered manifest identities are `16aecbe1fdcb03767ba6c814df4e97bdb5e563d32da07ab414633ac713ecbc7c` (order 0) and `f09b6da9fe531752d51223389455d9fc1af71a33ae9b68d0e168fb72039b4bb9` (order 1). Each order has 176 REFUGE, 307 ORIGA, 386 REFUGE_Valid and 19 Drishti_GS principal scoring visits; the remaining 136 preserve their nonprincipal eligibility.

Validation: 13 relevant existing/new tests passed in 9.334 seconds; an additional domain-weighting/report regression passed in 0.069 seconds. They cover current prediction/write causality, zero writer state preservation, source parameter boundaries, exact optimizer/RNG/EMA restore, scheduler identity, independent order state, target-mask isolation, native optimizer accounting, journal/scoring recovery, process-group child termination, domain-equal aggregation and OD/OC paired differences. The real source-only smoke completed two WARM updates and two post rounds in 21.413 seconds, with 96 measured model forwards, 20 backward calls, six optimizer steps, and zero VJP. Smoke artifacts are explicitly separate and are not formal training checkpoints.

Old profile model/assets, GPU, interpreter and NFS conditions match this run. Both old cold and after-64 native G samples explicitly passed 9F/2B/1Adam. Existing conservative 1.3×max costs were reused without deleting cold spikes; target projections also add 0.13 seconds per visit for unmeasured reads/I/O plus initialization, journals, scoring and checkpoint costs. No full profile was repeated. Source smoke did not use target data.

The serial queue is supervised independently of the SSH/Codex connection. Each worker is in an owned process group with an absolute stage timeout; a separate watchdog bounds the supervisor and closing period. The launch check found one GPU worker on authorized GPU 5, with no immediate error, and neutral process arguments throughout this run's watchdog/supervisor/worker chain. Other users' processes were untouched. Original R9 and the full R10 graph remain unchanged.

No main result table is claimed. On termination the queue writes private `RESULTS.csv`, domain aggregates, paired differences, behavior, full resource receipts and `REPORT.md`. Anonymous final results still require collection and a local results commit when execution finishes. No GitHub push was performed or authorized for this task.
