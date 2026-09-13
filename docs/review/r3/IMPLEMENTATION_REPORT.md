# R3 Stage I implementation report

Status: **R3_IMPLEMENTATION_READY_FOR_REVIEW**. External review is pending; no GPU or formal execution is authorized by this handoff.

Implementation commit: `6d8fc7317506400b039d3d2ed41bca18523beb27`. Base: `f52f132e4be576ea871467432a6f4b12337209a5`. Science byte SHA256: `73879c29a33897beb9a79e6498258998abd8c0a964f8befc72b4ac1cdbd9c06e`. The JSON is byte-preserved from the supplied proposal. Base registration remains `8afb0a9ea4e5fe1701bd89a04778de1b2ef0fac70e7aa8ec62fe288f071ffeaf`; new stream digest is `cc5fb8e0071ec9a0d25210bb34f4505359653dcc8e6bd884de47edb21b2253db`.

## Implemented scope

All 17 arms are present: C, RP, T_LR/T_ISO/T_DIAG, U_PCA/U_RAND/U_SCALE, S_JOINT/S_SHARED/S_NOPCA, M_TRANSPORT/M_IDPOST/M_SHUFFLE, G_PCA/G_ISO/G_ORDER. The five independent roles are teacher density, actual Adam displacement, context state retrieval, post-update statistics transport and graph teacher geometry. No score gate or framework combination was added.

The implementation adds 20 code/config/test files and modifies **zero existing files**. Original C and RP delegate to R1 C/REGION. The new-family engine preserves pinned six-view CPU reduction and strong augmentation. Verified asset reads, mask-after-output pipeline, live-directory ENOENT handling, atomic publication and process ownership remain in their existing modules. New execution wiring passes R3 resource caps into the original supervisor; it does not rewrite the supervisor or relax missing-evidence validation.

The future entry requires a separate enabled authorization bound to exact code/science/registration/stream, explicit devices and all 85 trajectories. Default authorization is disabled. No background launcher, future waiter or automatic retry was added. Future scalar aggregation validates the complete same-batch C/RP and all candidates, reports four primary orders separately from recurrence, and invalidates stale completion on failure. CPU fixtures validate this wiring; it has not been run on a GPU or real data.

## Actual CPU checks

- Supplied untouched reference suite: **24 passed**, 0.605 seconds.
- Final combined R3 + existing R1/R2 math/host/IO/process/continuation regression: **97 passed**, no failures/errors/skips, 210.024 seconds. [Actual log](logs/cpu-06.log), [result JSON](logs/cpu-06.json).
- Final focused check after the added S host-switch test and density-snapshot validation: **4 passed**, 9.437 seconds. Three overlap the combined suite; the added S test brings covered distinct tests to 98. [Log](logs/final-focused.log).
- Real ResUNet34, random programmatic weights: 41 BN layers, 82 affine tensors, 19,136 scalars; all 17 cold/ready paths plus old C/RP references. **38 Adam / 38 loss backward / 316 network forwards / 18 actual Jacobian VJPs / 5 explicit parameter replacement operations**. These are CPU mechanical tests, not GPU smoke or target-data scores. [Traces](FULL_MODEL_CPU_TRACES.json).
- Cold new-family steps match original C; C/RP two-step parity holds. U finite differences cover parameter-space mapping and unused columns; moments remain the single raw-gradient Adam proposal. S's actual host switches away and back without replaying augmentation, copying the full model or mutating the unselected Adam state. M and G controls and fallback paths are exercised.
- 85-job synthetic scalar closeout, truncation, missing evidence and disabled execution entry passed. Synthetic fixture rows (12 per job) are not the metadata dry-run's formal 1,951 visits. No simulated metric is published as a real result.

Environment: existing Python 3.12.9, PyTorch 2.6.0, pinned model/GraTa dependencies, no installs. The combined reference suite sets one Torch CPU thread; focused checks use the runner's two-thread setting. CUDA initialization is explicitly blocked. IO regression permits only tiny test-created assets in a fresh temporary tree; no registered RGB/mask/checkpoint is read.

## Frozen metadata plan

17 arms × 5 streams = **85 complete planned trajectories**; 1,951 visits each. Formal budget: **165,835 scoring / loss backward / Adam calls**, **1,385,210 network forwards**, at most **234,120 Jacobian VJPs**. Four primary streams: 68 trajectories / 132,668 records. Recurrence: 17 trajectories / 33,167 records, separately summarized. Devices remain unassigned; all jobs are NOT_RUN.

The new recurrence has **33 fixed chunks of at most 64 contents**, rotating the base domain order by round. It covers 1,951 distinct groups exactly once, preserves registered within-domain sequence and subset counts, and leaves all four primary streams unchanged. Public metadata exposes counts/chunk order and digest only; private ordered identities are not published. JSON and CSV both include every job and worker rotation for 1/2/3 slots.

## Residual issues and limits

One inherited persistent-EIO ownership test failed intermittently in CPU 05 (owned-return/reap assertions). The test's fallback cleaned its temporary processes; outsider survived. Nine independent diagnostic reruns and the final combined regression passed, but the original failure's cause is **not established**. This is retained as a review item, not described as a repaired bug or deleted from the logs. [History](DEVELOPMENT_LOG.md), [failure log](logs/cpu-05.log), [diagnostics](logs/eio-diagnostic-02.log). No shared supervisor code was changed speculatively.

The code explicitly uses lagged density/graph covariance snapshots at the fixed contribution refresh cadence; this interpretation and zero-rank density readiness are called out in [method provenance](METHOD_PROVENANCE.md) for external review. No conflict was silently resolved by tuning science values. Programmatic correctness does not establish density calibration, global forgetting protection, useful context expansion, accurate historical re-encoding, graph benefit or clinical safety.

No supplied checkpoint, actual target pixels/masks, GPU environment, real latency/peak memory, formal trajectory or segmentation performance has been evaluated. No source data, source proxy/prototype or source retraining was accessed. See [unrun items](UNRUN_ITEMS.md). The earlier failure logs' `checkpoint_reads=0` denotes no registered source checkpoint; tiny self-created weight-file IO tests are separately disclosed.

## Reproduce Stage I

Use the fixed implementation commit and existing pinned dependencies. Set DPA_CTTA_BASE_ROOT / DPA_GRATA_ROOT to their authorized local checkouts and PYTHONPATH to repository `src` and `tests`. Use the existing Python with Torch and batchgenerators 0.25.2. Set RUN_FILE to `scripts/check_r3_cpu.py`, R3_REGRESSION=1, CUDA_VISIBLE_DEVICES empty, then run the neutral entry `python -c 'import os,runpy;runpy.run_path(os.environ["RUN_FILE"],run_name="__main__")'`. Optional CHECK_OUTPUT and TRACE_OUTPUT receive CPU JSON evidence. Current full suite includes the subsequently added S switch test (98 tests).

For metadata, use `scripts/run_r3.py` with registration JSON and output directory through environment-supplied arguments; no pixel decoder runs. Keep `configs/r3_execution.defaults.json` disabled. The private registration is necessary to recompute its exact stream digest and is intentionally not published.

Deliverable contents are [indexed here](REVIEW_INDEX.md), including the complete base-to-implementation patch. Code and report publication commits are intentionally separate. Stop after handoff; wait for external review of this exact implementation.
