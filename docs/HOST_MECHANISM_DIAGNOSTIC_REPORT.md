# H1 host mechanism diagnostic report — preparation checkpoint

This checkpoint records completed existing-log analysis and the isolated diagnostic implementation. GPU smoke/formal status will be recorded from their actual execution; no GPU completion is claimed here. The current user task authorizes physical GPU 7 only, co-resident use without changing other programs, and a finite 80-step smoke plus 200-step formal diagnosis. Original pilot and source inputs are not rerun/reselected.

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

## GPU diagnostics and research decision

NOT_RUN at preparation checkpoint. N/G/I/U, six-branch counterfactuals, oracle diagnostics, actual GPU call counts and resource observations must come from the frozen execution. No H variant is implemented. A next-round candidate cannot be selected from the existing-log analysis alone.
