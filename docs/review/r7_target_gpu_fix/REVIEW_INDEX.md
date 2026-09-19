# R7 mixed-device target route: ready for external review

Tested implementation: **07ae5426ee29ab594d5d5bee196c019fdf5d61dc**. The publication commit is separate. User requested autonomous resolution followed by experiment launch. Engineering and numerical-route preparation are complete; the earlier explicitly required external execution review has not been supplied. TARGET_SCREEN remains NOT_RUN, receipt disabled. No self-issued external PASS.

## Resolution

[Localization](diagnostic.json) used one original C visit on each device (16F/2B/2Adam). Initial tensors and all eight inputs were bitwise equal. All seven pre-update predictions met the original tolerance; the final prediction after BN/Adam update had12 violations. This locates the discrepancy after adaptation, without asserting a particular kernel is faulty or loosening numerical tolerances.

The new route keeps unchanged historical C_BASE on CPU, with no visible GPU in that job's child; C0 and all six frozen methods retain the source-qualified CUDA backbone and CPU method/FP64 state. This is an execution-device choice, not an algorithm change. It trades some wall-clock speed for preservation of the original baseline. The prior all-GPU failure remains [FAILED](gpu-9f7.json); it is not reclassified as passed.

[Mixed-route plan](MIXED_ROUTE_PLAN.md) preceded execution. [Qualification](gpu-mixed.json) passed all eight arms, C_BASE8F/1B/1Adam and exact CPU fresh repeat, C01F and repeat, all FULL/STATIC counters and fresh-static equivalence, serialized procedural packages and frozen backbone. Actual79F(32CPU+47GPU),4B/4Adam, noVJP/AdamW,44.152s. [Current30 target CPU tests](target-mixed.json) passed. Previously passed59R7 and34source-boundary checks remain applicable: their production modules, original C and trusted artifact loader have no diff from9f7c597. These are123 relevant checks across explicitly identified SHAs, not123 reruns on the new SHA. No historical166 suite.

[Real six-artifact disk roundtrip](roundtrip-9f7.json) remains valid evidence for the unchanged loader/network/context modules: six fresh segmenters, independent submitted inventory/context hashes, original checkpoint/effective tensors,12deserializations, zero forwards or pixels. The source run, artifacts and old CPU attempt were preserved; no source retraining.

## Shell repairs and binding

TS-COST-1: actual per-attempt counters and durable snapshots preserve partial online/model/prediction/scoring/evidence failures; successful online counts precede scoring. First failures remain distinct from cleanup/evidence failures. Hard-kill tails are lower-bound/unknown, never inferred from nominal counts or prediction-file count.

TS-BACKEND-1: CPU route rejects GPU contexts; GPU source contexts are bound exactly; mixed route explicitly requires CPU baseline. A bounded owned all-eight factory readiness subprocess must succeed before any target job. Every job has its own process/model/optimizer/state/output; label scoring starts only after model release. Finite job/matrix wall/output audits and terminal cleanup remain required. Procedural path/link/label isolation checks passed; real target pixels remain unread.

- [Exact24 jobs](EXACT_24_JOB_MANIFEST.json), [frozen preservation](PRESERVATION.json), [scheduler](SCHEDULER_PLAN.json).
- [Resource proposal](RESOURCE_PROPOSAL.json): one serial worker, GPU6 proposed for seven non-baseline arms; CPU C_BASE,2threads. Job wall cap12hours and matrix cap96hours are finite upper limits, not ETA; output256MiB/job and7GiB total. Three workers are not approved.
- [Evidence binding](MIXED_EVIDENCE_BINDING.json), [inventory](ARTIFACT_INVENTORY.json), [schema](ARTIFACT_INVENTORY.schema.json), [disabled authorization](AUTHORIZATION.disabled.json).
- [Implementation patch](execution-layer.patch), [delivery](DELIVERY.json), [original submitted review](external_review/README.md).

Public paths/physical authorization stay null and enable stays false. Private exact candidate binding is prepared but disabled. Remaining step is the externally reviewed code/artifact/device/resource/registration/output binding required by the original user instructions, including review of the newly supplied roundtrip/mixed-route evidence. A request to run is user authorization, not evidence that an external reviewer issued PASS.

Monitoring remains deleted. No target run, MECHANISM/EXTENSION, retry/resume, scientific winner or final-label tuning. Qualification logs retain failures and sanitize private path prefixes only.
