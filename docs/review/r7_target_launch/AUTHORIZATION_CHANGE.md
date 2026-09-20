# Explicit user-authorized launch

User instruction on2026-09-20: “撤销门槛，快速启动”. This explicitly replaces the prior mandatory external-PASS condition for this frozen TARGET_SCREEN launch. It does not alter science, seeds, ordering, budgets, source identities, device qualification, path isolation, no-retry/resume, error preservation, output caps or owned-process cleanup. Monitoring stays deleted.

Runner accepts USER_WAIVED only with an explicit waiver tied to the entire binding digest and TARGET_SCREEN scope. It never reports external PASS. Source artifacts use USER_ACCEPTED_VERIFIED_ARTIFACTS with trusted-loader verification and the independently supplied inventory; all fresh factory loads still execute before any target job.

Implementation00dc8f630ae2d4ef1b7d931cf21a0c2c32b09eab has31 target procedural tests passed, plus renewed exact-SHA mixed-device qualification79F/4B/4Adam passed. Historical all-GPU failure remains preserved. C_BASE CPU; other seven arms source-compatible GPU6. One serial worker, fixed24 jobs, seed20260907, orders0/1/4; each1951 arrivals and1695 scored contents. Nominal122913F/5853B/5853Adam,46824 total visits and40680 scored rows.

12-hour per-job and96-hour aggregate wall caps,256MiB per-job and7GiB aggregate output caps. These are limits, not ETA. No automatic retries, source dispatch, mechanism/extension or continued monitoring.
