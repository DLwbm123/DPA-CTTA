# R8 implementation status (2026-09-25 Asia/Shanghai)

Status: **IN PROGRESS; NO R8 MODEL OR TARGET JOB STARTED.**

The user authorized the R8 plan on `jiangsuiyang` physical GPUs 5/6/7 and requested hourly monitoring with bounded repair and continuation. The older statement in the attached design that the then-current request was only to write a plan is superseded by this newer user request. The numerical, identity, and resource boundaries in the plan remain the implementation target.

This worktree is `experiment/r8-ba-performance-envelope-v1`, based on R7 handoff commit `5c1d4588094d7967e5f1c81a7d7d6896470948ca`. `docs/review/r8/input/` contains the six original ZIP members. `scripts/r8/manifest.py` expands a disabled static graph with 65 source training and 724 target jobs, totaling 2,325,592 target arrivals; selected A/B IDs remain placeholders until source-only selection. `src/dpa_ctta/r8_ba/` currently has parameterized FiLM, A/B/MLP modules, source schedules, oracle and basis primitives. Small CPU tensor checks are under `tests/r8`.

Missing before a real launch: complete source trainer with 32-visit/8-visit TBPTT, five independent snapshot calibrations, source-only selection, CURRENT-MLP training, target host including B gradient arms and all native baselines, complete state snapshot/resume, exact source/target identity binding, output and cost ledgers, maximum-rank and native-method GPU profiles, resource projection with 1.3 margin, clean code SHA, launch receipt, and end-to-end checks. R7 jobs and artifacts must remain untouched. No external review PASS or resource qualification has been claimed.

Remote check on 2026-09-25: `jiangsuiyang` is `zmic44`; GPUs 5/6/7 are RTX 3090 24 GiB and showed about 24,124 MiB free each. `/data_nas/jiangsuiyang` is an NFS mount with 27 TiB available. These are point-in-time observations, not launch preflight. The hourly Codex heartbeat is `r8-b-a`; it should stay quiet on unchanged state and pause after final completion.
