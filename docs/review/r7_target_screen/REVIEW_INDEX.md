# R7 TARGET_SCREEN execution shell — GPU-first completion pending scientific review

The GPU-first run completed the frozen 24-job matrix with the user-approved backend amendment. See `GPU_FIRST_COMPLETION_PUBLIC.md` for the public, source-only completion record. The execution-layer review remains `USER_WAIVED`; this package does not assert an external `PASS` or issue a scientific winner.

Status: **R7_TARGET_SCREEN_IMPLEMENTATION_READY_FOR_REVIEW**. This package does not issue an execution-layer PASS. SOURCE_PREP artifacts and exact runtime authorization remain PENDING. No real target pixels, source RGB/masks, real checkpoint or GPU were accessed by this implementation task.

- `IMPLEMENTATION_SHA`, `IMPLEMENTATION_BINDING.json`, `execution-layer.patch`: exact code revision and patch from `ecde8ba`.
- `DELIVERY.json`, `CPU_RESULTS.json`, `logs/`: status, synthetic/procedural checks and retained development logs.
- `EXACT_24_JOB_MANIFEST.json`: unchanged arm/order list, seed 20260907, 1951 arrivals and 1695 remaining_dev contents per job. Frozen totals: 24 jobs / 46824 visits / 122913 F / 5853 backward / 5853 Adam. The report filters remaining_dev, giving 40680 scalar rows across jobs; this does not change the frozen 46824 nominal visit count.
- `SCHEDULER_PLAN.json`: one worker follows the complete frozen list; three reviewed workers own orders 0, 1, 4 respectively, 40971 nominal forwards each. Each job gets a fresh OS process, model, artifact instance, optimizer/state and output directory.
- `RESOURCE_PROPOSAL.json`: finite wall/output cap proposal, **not approval or a measured ETA**. Exact hostname/Python/Torch/CPU-count/dtype/thread/storage binding required. This release enables only the existing CPU tensor route after authorization. GPU is rejected even if GPU IDs are supplied: GPU conversion and numerical qualification are separately reviewable work, not a receipt toggle.
- `ARTIFACT_BINDING.md`, `ARTIFACT_INVENTORY.schema.json`: compatibility with SOURCE_PREP's existing six-artifact release and trusted loader.
- `ISOLATION_AND_FAILURE_EVIDENCE.md`: target reads, labels, outputs, terminal audits, failures and test mapping.
- `RUN_CPU.md`: reproducible local checks, neutral process arguments, synthetic-only guards.
- `configs/r7_target_screen.authorization.disabled.json`: disabled template (relative to repository root). Artifact identity=PENDING, GPU IDs=null, authorization=false.

## Review boundary

Only new execution-shell files, tests, disabled configuration and review material are changed. Existing source preparation, tensor algorithms, OnlineHost, C recipe, frozen science and shared report are reused without modification. No historical 166-test suite rerun. Source preparation is a separate task, never dispatched here. No automatic smoke, mechanism, extension, retry or resume.

Before any future launch, externally review the actual SOURCE_PREP release, fill exact artifact/checkpoint/registration/context/resource/output identities, and obtain the external TARGET_SCREEN execution review over `json_digest(binding)` plus a matching scope-specific user receipt. The authority JSON is a reviewed trust input, not a cryptographic signature. A local edit saying PASS is not an external review. Bind the exact clean checkout HEAD used to launch; the final documentation publication commit may be a descendant of IMPLEMENTATION_SHA with identical executable files. Re-review any executable change.

The periodic heartbeat follows source/review readiness and authorized results every hour. It is not execution authorization and cannot relax these gates. No unrelated historical files were deleted or moved to NAS.
