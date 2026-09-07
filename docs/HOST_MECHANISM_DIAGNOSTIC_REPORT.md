# H1 host mechanism diagnostic report — BLOCKED_GPU_RESOURCE

Existing-log analysis and the independent diagnostic implementation/local tests are complete. GPU smoke and formal diagnostics were **not started** because the deployment preflight showed insufficient admission headroom on the only authorized GPU. This is **BLOCKED_GPU_RESOURCE**, not H1_DIAGNOSTIC_COMPLETE, and is not a claim that H1 was measured to OOM. No other process was modified, no other GPU was used, and no waiter or automatic continuation exists.

Frozen implementation commit: `daa95d25ad03e9240621c832d4e45d8ab188b1b2`. **GPU execution commit: none.** The commit containing this final report is the later result/blocker delivery. Task A ran before code freeze using the new analysis file plus the original `1f8f1fa` validator; that file is included in the frozen implementation. No GPU smoke evidence or formal execution receipt is fabricated.

[Diagnostic configuration](../configs/host_mechanism_diagnostic_v1.json), [implementation contract](HOST_MECHANISM_DIAGNOSTIC_CONTRACT.md), [old-log aggregates](../results/host_mechanism_diagnostic_v1/old_log_analysis.json), [next-round target-dev draft, not executed](TARGET_DEV_PROTOCOL_DRAFT_H1.md).

## Existing 832-record analysis — completed, no model run

The original private JSONL passed ordered coverage/channel/lifecycle validation. Analysis exited 0 with CUDA uninitialized and no Adam calls. Exact-zero signs, linear quantiles and ascending ceil(10% n) tails were fixed before analysis. Differences below are percentage points. ASSD differences use jointly defined pairs; macro ASSD is not defined. Original logs contain no actual retrieval events, prediction area or confusion matrices; none are inferred.

| Task / channel | Clean A−N mean | Median | p10 | p90 | Positive / zero / negative | Minimum | Worst 10% mean |
| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| fundus / OD | -1.869636 | -1.409808 | -2.053575 | -0.203453 | 2 / 0 / 18 | -15.552919 | -9.458257 |
| fundus / OC | -14.778140 | -15.376423 | -29.663198 | 0.265303 | 3 / 0 / 17 | -41.902986 | -36.439562 |
| fundus / macro | -8.323888 | -8.025009 | -15.377249 | -1.361080 | 0 / 0 / 20 | -22.633290 | -19.172978 |
| polyp / polyp | -0.511427 | -0.249367 | -0.767138 | 0.121092 | 7 / 0 / 25 | -4.534415 | -2.775470 |

| Task / macro or single channel | Clean positions 1–5 | 6–16 | 17 onward |
| --- | ---: | ---: | ---: |
| fundus | -5.584678 | -8.635904 | -10.889856 |
| polyp | -0.253147 | -0.793973 | -0.397889 |

All A−N/B−N/C−N/B−A/C−B task/channel/segment distributions, zero/positive/negative counts, signed lower tails, common-ASSD counts and same-content cross-segment differences are in the linked JSON. Complete group-level four-segment measurements remain private. The position bins are not evidence of retrieval. These source development groups are not new validation data or verified independent patients; the descriptive trends do not establish significance or mechanism.

## Local regression

Full supported-environment regression: **179 tests**, 0 failures, 0 errors, 0 skips; exit 0; 374.47 seconds. Python 3.12.9, Torch 2.6.0. Old tests and assertions are retained. [Machine audit](../audit/host_diagnostic_results.json), [full log](../audit/host_diagnostic_cpu.log). Four earlier targeted tests also passed. Real source images/weights and CUDA were denied during this CPU suite; blocked API attempts were empty.

## GPU preflight, counts and stop

The deployment preflight observed physical GPU **7**, its expected UUID verified and recorded privately, with **2739 MiB free (2.675 GiB)**. The prior source-pilot Fundus full-K=4 B smoke measured **3525201408 allocated bytes (3.283 GiB)**, and its formal B peak was 3.883 GiB. H1 additionally maintains independent reference/watched and diagnostic models. The available memory was below that previously measured full-K=4 step, so admission was stopped conservatively before any GPU model allocation/run. The actual peak of this new H1 implementation is **unmeasured**; the prior peak is evidence for the admission decision, not an asserted exact H1 minimum or observed H1 OOM. No K reduction, GPU switch, process interference or background wait was used.

The existing registration digest still matches the original receipt, and all 122 registered file size/mtime records are unchanged. Assets were not replaced or re-registered; this round did not repeat raw-byte hashing or decode real query images/labels. The copied server payload contains source code only. The existing environment was retained.

| Phase | Status | Actual GPU native steps | GPU counterfactual steps | Exit |
| --- | --- | ---: | ---: | --- |
| Existing 832-record analysis | COMPLETE | 0 | 0 | 0 |
| Full CPU regression | PASS, 179 tests, no skips | 0 | 0 | 0 |
| Deployment/resource preflight | BLOCKED_GPU_RESOURCE | 0 | 0 | deployment command 0; admission blocked |
| Prescribed 80-step GPU smoke | NOT_RUN | 0 | 0 | not invoked / null |
| Prescribed 200-step source clean diagnostic | NOT_RUN | 0 | 0 | not invoked / null |
| Independent new GPU-result recomputation | NOT_RUN, no new GPU records | 0 | 0 | not invoked / null |

Actual H1 GPU model/prompt/proxy forwards and extra gradient calls are all **0**. The 280-call GPU budget was not consumed. CPU procedural regression calls are separate. New GPU peak/time are null, not zero-valued measurements. Old-log analysis took 0.186 s; full local regression took 374.47 s. Private file-byte total after the public export is 783431 bytes (derived from the checked pre-export total and export file size), well below 256 MiB. Directory ownership/mode 0700 and file permissions 0600 were checked. No model, optimizer, image, mask, prediction, feature or gradient-vector artifact was saved.

[Execution/resource audit](../results/host_mechanism_diagnostic_v1/execution_audit.json). No partial real diagnostic JSONL or failed-visit prefix exists because no GPU visit was attempted. Local procedural failure tests preserve the first completed record and stop at the second visit without completion. Input registration/privacy and GPU-resource admission are separate from scientific effectiveness.

## Mechanism, oracle and next-round decision

Real source N/G/I/U and all six counterfactual/oracle results are **NOT_RUN**. Only random-source-state, procedural-input CPU checks establish current code isolation; they do not establish real-checkpoint GPU equivalence or source-query mechanism. The old-log losses alone cannot distinguish AdaBN, memory initialization, update magnitude or proxy/query mismatch.

**Mechanism remains undetermined. Next-round candidate: insufficient evidence.** No H implementation is selected or built. The target-dev draft specifies N, original A, original B and at most one future evidence-supported H, but no target asset was enumerated/read and no target experiment was executed. Nothing in this source analysis makes exceeding N on source a prerequisite for shifted development; no oracle or boundary-supervision global conclusion is drawn.

Stop status: **BLOCKED_GPU_RESOURCE**. This task is stopped; it will not automatically resume or enter another experiment. CI is NOT_CONFIGURED.
