# R7-F1 repair and rereview index

**R7_IMPLEMENTATION_READY_FOR_REVIEW** · external_review=AWAITING_REREVIEW · source_binding_status=PENDING.

Tested implementation: `8f179ce0ca848791793a17dab0606f0bde2b3e07`. Branch: `review/r7-inference-context-fix-v1`. This is a Stage I shared identity fix; real source/target execution remains disabled. The later evidence publication commit is not the tested code SHA.

- [Delivery and execution boundaries](DELIVERY.json)
- [R7-F1 resolution and limitations](R7_F1_RESOLUTION.md)
- [Complete context schema and provenance mapping](CONTEXT_SCHEMA.md)
- [Implementation SHA](IMPLEMENTATION_SHA) · [changed-file binding](IMPLEMENTATION_BINDING.json) · [narrow code/test patch](implementation.patch)
- [Four original science hashes and preservation proof](PRESERVATION.json)
- [Actual CPU results, costs and prior failures](CPU_RESULTS.json) · [reproduction instructions](RUN_CPU.md)
- [Final local implementation log](logs/final-local.log) · [final server implementation + reference log](logs/final-server.log) · [independent local reference log](logs/final-math-local.log)
- [Before-fix observation](logs/before-fix.log) · [two failing rejection assertions on old code](logs/before-rejections.log) · [old-code reproducer](reproduce_reviewed_code.py)
- [Development log 1](logs/dev01.log) · [development log 2](logs/dev02.log) · [raw/public log hashes](LOG_MANIFEST.json)
- [Supplied review](input/R7_EXTERNAL_REVIEW.md) · [fix prompt](input/R7_FIX_CODEX_PROMPT.md) · [review record](input/R7_REVIEW_RECORD.json) · [input byte/hash record](INPUT_MANIFEST.json)
- [A group formula/state evidence](../r7/A/REVIEW_INDEX.md) · [B group evidence](../r7/B/REVIEW_INDEX.md) · [C group evidence](../r7/C/REVIEW_INDEX.md)
- [Unchanged original science](../r7/SCIENCE_BINDING.json) · [target metadata matrix](../r7/DRY_RUN.json) · [source budget](../r7/SOURCE_BUDGET.json)

Both environments passed 59 implementation tests (original47 + new12) and22 independent reference checks, zero final failures/errors/skips. Local suite: 41.814s; server: 89.010s. Each suite executed495 procedural backbone forwards:85 random full ResUNet and410 Small fixture, including16 old-C forwards. Actual backward=36, VJP=2, Adam=24, AdamW=6. Each new context test suite contributes31 forwards (6 full,25 Small), no additional backward/optimizer/VJP. Model calls are not zero and are not real source/target experiments.

Measured hash work per suite:407 named-tensor digest calls,978899706 tensor bytes processed (repeated hashing, not model size). Full-context builds=199; wall time local=0.614752s, server=0.996113s; process CPU local=1.118106s, server=1.149247s. Context timings include tensor work; do not add them. Small-matrix operations and per-run development costs remain in CPU_RESULTS.

The real loader and restore reject incompatible head, encoder, projection, policy, group/static/ablation and method assets before forwards. Same-byte distinct instances restore and produce exactly equal next outputs; C0 also binds the environment. Preparation records the initial training environment and refuses drift. Expected contexts come from independently trusted preparation metadata, not from the candidate segmenter.

Local NumPy RuntimeWarnings remain visible:33 lines in the implementation log and15 in the independent reference log. The server log has0 observed RuntimeWarning lines. The warning origin was not diagnosed and is not asserted benign. No formulas/tolerances were changed. Mocked preparation packaging checks are explicitly labeled and do not claim1000 source steps.

Original reports remain unchanged as history, including the original review NEEDS_FIX and prior failure evidence. Group links above retain their original implementation/test SHA; use this index for the current shared-binding patch and new fixed-SHA regressions. No external rereview PASS is asserted. No GPU query or real RGB/mask/checkpoint read occurred. SOURCE_PREP/TARGET_SCREEN/TARGET_MECHANISM/TARGET_EXTENSION=NOT_RUN. Stop here for external rereview.
