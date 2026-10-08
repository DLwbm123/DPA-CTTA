# R30 startup: complete matrix admitted and running

Runtime code: `a46f0cd8d670b748278eb00ebdf1191445984923`.

All generated-input mechanical checks passed, including exact unchanged C/ANCHOR replay, independent affine/Adam resets, arrival-33 timing, persistent global RNG/accounting, frozen source configurations, committed delayed branches, and whole-window oracle semantics. A separate synthetic 48-stream scorer check passed endpoint separation, reference selection, units, interaction arithmetic and negative-cell reporting.

Three real concurrent A100 profiles passed. Each GPU profiled all eight B arms over 34 arrivals, including periodic reset and read-only diagnostics, plus one three-branch delayed window. Formal admission reserves 20% for recovery. Measured profiles consumed 467.613 GPU-worker seconds. The conservative formal-work projection is 15.862 GPU-worker hours; including the recovery allowance and measured profiles, the bound is 19.957 hours, below the 36-hour cap. Initialization, I/O, diagnostic and profile parity work are included in the estimate. This is an estimate, not a completion receipt.

The first three ANCHOR streams were confirmed active at 62, 53 and 62 of 1,951 arrivals, with fresh heartbeats, matching process identities and neutral command lines. All were on the registered A100 GPUs 0/1/2. No failure was recorded. The full matrix has 48 complete streams (93,648 arrivals) and six delayed-diagnostic jobs (96 initial states; 17,055 branch arrivals including truncated tails). H=64 has 84 eligible states, H=32 has 87 and H=8 has 90; all 96 states remain represented at H=0.

B runs before A so each formal FULL delayed branch can be compared against the full ANCHOR reference. All online workers must retire successfully before CPU labels are released. No intermediate score is used for pruning or intervention selection. Full results remain pending. The conservative profile estimate places online completion around 2026-10-09 06:47 +08:00, followed by CPU scoring; actual contention and I/O may change this estimate.

The immutable limits are 24 wall-clock hours from 2026-10-08 23:21:58 +08:00, online cutoff at hour 22, 36 GPU-worker hours including profiles/failures, and 64 GiB of new private output. No hourly monitor or R31 was created. Background execution is independent of the chat session.

See [protocol](PROTOCOL.md), [registration](REGISTRATION.json), [profile admission](PROFILE_ADMISSION.json), [mechanical checks](MECHANICAL_TESTS.json), [scorer self-check](SCORER_SELF_CHECK.json), [state inventory](STATE_INVENTORY.json), and [window eligibility](WINDOW_REGISTRATION.json).

SEARCH and old REVIEW are development-exposed. No independent or efficacy claim follows from successful startup. Public files exclude private images, labels, masks, identities, states, weights and storage paths.
