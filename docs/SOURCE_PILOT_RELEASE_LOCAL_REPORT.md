# Release local regression report

Baseline 8d417ca; branch experiment/source-pilot-run-v1. E1/E2/E3 and receipt-bound entry are implemented under the current direct user task. The original config SHA remains e1aeb2454899d024e00934e1f012b5a0fe8cb3bb656f5dd20f73468fdd7a3faf. Native host/image-block, Prompt/Memory/AdaBN, model, loss, preprocessing and source selection source files were not edited.

Command: `PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' DPA_CTTA_BASE_ROOT=$REFERENCE $PYTHON -P audit/run_source_pilot_release_checks.py`.

Observed exit **0**; **173 tests**, zero failures/errors/skips, 366.90 seconds; Python 3.12.9, Torch 2.6.0, SciPy 1.16.0. The full suite includes 45-image default-neighbor native equivalence, K=4 full-model gradients, input/clone/label regression and five release tests. The release tests reject 11 bad result cases, preserve N/A plus B prefix on failure, distinguish initialization from optimizer update, reject mismatched receipts, and compare read-only optimizer hooks against native output/state/RNG for both full model families.

The final aggregate count is computed from recomputed rows and checked against 832 visits / 624 updates, rather than reported from a constant alone. This production completion branch remains subject to actual registered execution. The final scripts compile. Four early targeted release tests also passed; no pinned independent old-defect probe was loosened or passed off as a new-release test.

[Machine results](../audit/source_pilot_release_results.json), [full log](../audit/source_pilot_release_cpu.log), [release contract](SOURCE_PILOT_RELEASE_CONTRACT.md), [execution entry](../scripts/run_source_pilot_once.py).

At this local checkpoint, real source/checkpoint registration, GPU smoke and formal pilot are NOT_RUN. The supplied review is not a fabricated independent signoff for this new patch. Later execution receipts bind the actual clean commit after it exists. CI remains NOT_CONFIGURED.
