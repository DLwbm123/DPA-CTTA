# R7 SOURCE_PREP preparation review index

**R7_SOURCE_PREP_IMPLEMENTATION_READY_FOR_REVIEW**. Stage I review=PASS; new source execution-layer review=NOT_RUN; source binding=PENDING; execution_authorized=false.

Tested implementation: `9fa2ce37149909687ef5be2e08aa2b32a5887903`. Evidence-only publication follows on `review/r7-source-prep-v1`; those later documentation bytes are not claimed as a rerun SHA.

- [Delivery and stop conditions](DELIVERY.json) · [implementation SHA](IMPLEMENTATION_SHA) · [source/test file binding](IMPLEMENTATION_BINDING.json)
- [Execution/data/device/budget contract and limits](IMPLEMENTATION.md) · [narrow patch](implementation.patch)
- [Real source evidence and explicit pending fields](SOURCE_BINDING.json) · [frozen111/23/25 split audit](SOURCE_SPLIT_AUDIT.json)
- [Proposed full source costs/resources](PROPOSED_COST_AND_RESOURCES.json) · [default-disabled resource config](../../../configs/r7_source_prep.defaults.json) · [default-disabled authorization template](../../../configs/r7_source_prep.authorization.disabled.json)
- [Actual CPU results, costs and retained development failures](CPU_RESULTS.json) · [reproduce acceptance](RUN_CPU.md) · [log byte/hash records](LOG_MANIFEST.json)
- [Final local17 execution checks](logs/final-local-source.log) · [local59 R7 checks](logs/final-local-r7.log) · [server17 execution checks](logs/final-server-source.log) · [server59 R7 checks](logs/final-server-r7.log)
- [First failed development run](logs/dev01.log) · [second failed run](logs/dev02.log) · [passing development execution checks](logs/dev03.log) · [development R7 regression](logs/dev-r7.log)
- [Science/method/history preservation](PRESERVATION.json)
- [Supplied Stage I rereview](input/R7_F1_REREVIEW.md) · [external rereview record](input/R7_REREVIEW_RECORD.json) · [preparation prompt](input/R7_SOURCE_PREP_PREPARATION_PROMPT.md) · [exact supplied input hashes](INPUT_MANIFEST.json)

Each environment passed17 new execution-layer tests and59 R7 regressions, zero final failures/errors/skips. Local time: new=1.005s, R7=35.021s. Server: new=1.936s, R7=91.334s. Each environment made497 procedural forwards (85 random full ResUNet,412 Small),36 backward calls,2 VJPs,24 Adam and6 AdamW steps. New checks alone made2 Small forwards,0 backward/VJP/optimizer calls,5 synthetic checkpoint loads and2 synthetic image/mask decodes. This is not source training. No old166 rerun or new22-reference run is claimed.

Known real source checkpoint bytes were independently hashed once, matching `88b7d8902d23fb1c15b27668e0bf02599f8298aa45c38ae90279ae5a1d42bdf0` (90388078 bytes), without deserialization/model loading. Existing registered159 source records have no patient/eye linkage; source groups use image-content SHA256 with explicit dependence risk. The unchanged R7 rule yields111/23/25 and no registered target-image hash overlap. Patient overlap and checkpoint pretraining membership remain unknown. Source image/mask bytes have not been decoded or rehashed; future authorized reads verify the exact registered bytes before decode. Private paths and membership rows remain private.

The complete proposed source budget is135328 forwards,9072 source backward +1536 calibration backward,3072 oracle Adam +1536 calibration Adam,6000 AdamW and1024 A-basis VJPs. This includes scaler, constant-R and oracle-query costs for the frozen split; it is not measured execution. The72-hour CPU cap is a proposal, not ETA; exhaustion stops without truncation/retry. GPU device IDs/resources are unbound, GPU qualification NOT_RUN, and the implementation rejects GPU execution.

Original formulas, four science files, FULL/STATIC budgets, query-state separation,24/9/at-most12 target matrix, old C and historical results remain unchanged. NumPy warnings and development failures are retained. Original Stage I PASS is supplied external evidence for its pinned SHA, not a self-issued PASS for this new layer.

All SOURCE_PREP/TARGET scopes remain NOT_RUN. Stop for external execution-layer review and future explicitly bound resources/source authorization.
