# R7 target GPU bridge and partial-cost repair

Implementation: **9f7c597fc510b3621f51e995a2dcb269e2fa8cd4**. This is the tested code SHA; the documentation publication commit is separate.

**Real TARGET_SCREEN was not started.** Engineering changes and serialized loader evidence are ready for review, but fixed GPU numerical qualification failed. No external TARGET PASS is issued. The user's request was “请根据文件解决问题立即开启实验”; necessary bounded qualification/loading was executed under that request. The attachment's embedded prompt was treated as review material, not a new user command. Earlier exact external review requirements remain unsatisfied.

## Findings and changes

- TS-COST-1: snapshot actual host/global counters after every attempted visit, including failed calls; retain committed visits, prediction-file count, failure phase and lower-bound/unknown tail. Persist successful online counts before scoring. Cost-write/cleanup errors cannot replace the first online failure. An exited worker receives terminal cleanup/resource audit and last-durable-cost evidence.
- TS-BACKEND-1: reuse the source GPU bridge, bind backend/qualification/code/device, reject GPU contexts on CPU before target reads, and load all eight factories in a bounded owned readiness process before the first target job.
- ART-ROUNDTRIP-1: six original disk artifacts passed existing load_artifact with the independent inventory transcribed in the supplied review, unchanged contexts, original checkpoint and effective tensor identity. Six fresh segmenters, 12 deserializations, zero model calls or image reads. This new evidence still needs external review.

## Measured evidence

[Qualification plan](QUALIFICATION_PLAN.md) was committed before execution. [28 target checks](target-9f7.json), [59 R7 checks](regression-9f7.json) and [34 source boundary checks](source-9f7.json) passed: 121 total, none skipped. Historical 166 suite and source training were not rerun.

[GPU qualification](gpu-9f7.json) **FAILED**, exit 1 after 19.671 seconds: C_BASE CPU/GPU final logits had 12/524288 elements outside predeclared rtol=.003/atol=.0003; greatest absolute difference among mismatches was 0.0004787929. Observed cost: 40 forwards, 5 backwards, 5 Adam calls (32 GPU and 8 CPU forwards). All five C_BASE visits obeyed 8F/1B/1Adam assertions. Subsequent C0 and six method GPU tests were **NOT_RUN** because qualification stopped at its first failure. The arms field is intended coverage, not coverage achieved. No threshold, precision, algorithm or seed was changed to pass.

[Serialized roundtrip](roundtrip-9f7.json) passed in 10.209 seconds on the matching source backend. [Terminal audit](TERMINAL_RESOURCE_AUDIT.json) confirmed both owned processes exited and evidence stayed below output caps. Public logs sanitize private path prefixes; private originals are preserved.

The discrepancy establishes neither a scientific defect nor harmlessness. Historical C includes BN-gradient/Adam updates; earlier source bridge qualification does not qualify that separate adaptive path. CPU historical-C preservation passed. Further diagnostics should compare identical initial tensors/RNG/augmented inputs, pre-update outputs, BN gradients/Adam state and post-update outputs against the unchanged official implementation. Keep this failed criterion; do not select seeds or widen tolerances after observing it. A separately bounded diagnostic plan and subsequent review are needed before claiming a qualified route. No automatic rerun occurred.

## Binding and isolation

- [Exact jobs](EXACT_24_JOB_MANIFEST.json), [scheduler](SCHEDULER_PLAN.json), [preservation](PRESERVATION.json).
- [Six identities](ARTIFACT_INVENTORY.json), [schema](ARTIFACT_INVENTORY.schema.json), [disabled authorization](AUTHORIZATION.disabled.json), [resource proposal](RESOURCE_PROPOSAL.json).
- [Patch](execution-layer.patch), [delivery](DELIVERY.json).
- Supplied review archived byte-for-byte under [external_review](external_review/README.md); its original probes concern the previous candidate, not a PASS for this repair.

The single-worker GPU-6 plan is a proposal, not TARGET resource approval. Public authorization retains null physical IDs/paths, false enable/authorization and NOT_RUN external review. Its qualification digest identifies a FAILED result and cannot pass preflight. An exact private proposal is prepared separately but remains disabled. No source or target pixels were opened. Only registration metadata was read for binding/recurrence arithmetic. No target reader traversed real target paths. CPU path/link/label-capability/output-isolation and failure tests passed on procedural fixtures.

This status supersedes historical preparation descriptions in r7_target_screen/DELIVERY.json for this candidate. Monitoring was deleted at the user's request and not recreated. SOURCE_PREP completion, contexts/checkpoints and failed CPU attempt remain preserved. No MECHANISM/EXTENSION, retries/resume, model selection, scientific winner or final-label tuning.
