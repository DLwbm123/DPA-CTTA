# R10 implementation and execution boundaries

Base: `13bd6a8cdf9c30a0a5ed7fa464c67e72301ac8c6`.
The supplied input specification is preserved verbatim. R10 uses its own namespace,
25-source/410-target graph, source selection lock, authorization, profile projection,
256-hour ledger, attempt receipts and final score release. R9 files are unchanged.

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
