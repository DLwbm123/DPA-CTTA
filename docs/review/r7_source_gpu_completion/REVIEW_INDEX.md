# R7 SOURCE_PREP completion — artifact review pending

R7_SOURCE_PREP_COMPLETE_PENDING_ARTIFACT_REVIEW. Executable SHA `ba04ca46a6bae5df4f4ebb272ef4045f5ee400e5`; GPU-transition publication `ea47b81727115d50abe4824d3d2ed1327377f3ab`. This evidence-only publication changes no executable code or science. External GPU review and external artifact review remain NOT_RUN. TARGET_SCREEN, TARGET_MECHANISM and TARGET_EXTENSION remain NOT_RUN; next_scope_authorized=false.

[Results and audit](RESULTS.json) contains all 45 completed phase records, exact counters, six artifact/context identities, effective frozen tensor identity, aggregate validation and terminal evidence. Parent/worker exits are zero; owned PIDs were absent at closeout; no first, after-check, supervisor or completion error records were found. Source after-check is UNCHANGED. Supervisor wall time is 8650.285 seconds (2 h 24 min 10 s). Output is 11,602,322 bytes against 536,870,912; wall cap is 259,200 seconds. The original supervisor audits after exit/cleanup and after pending completion writes before publishing completion.json.

All frozen counters match: 135328 forwards, 9072 source backwards, 1536 calibration backwards, 3072 source Adam, 1536 calibration Adam, 6000 AdamW and 1024 VJP. Each of six models completed 1000 fit and 256 calibration steps. Exact release code exercised the inference loader against independently retained expected context for all six before serialization. Closeout verified each serialized artifact length/hash and context-file/context identities. No additional model execution or separate post-closeout model reload is claimed.

The user-stopped CPU attempt remains separate: 4335.491 worker seconds, 4136 forwards, 2067 backwards and 2067 Adam, after-check UNCHANGED. Its FAILED/KeyboardInterrupt record is preserved as an authorized interruption; its costs are not subtracted from the GPU budget. Qualification costs remain in the prior GPU review index. No retries occurred within this GPU attempt.

## Source validation only

Each row averages 256 query observations from 64 source-validation episodes; these are repeated source queries, not 256 independent patients. Patient/eye dependence and pretrained exposure remain UNKNOWN. Soft Dice is not TARGET_SCREEN hard-mask scoring. No C_BASE/C0 result exists here, so FULL-C cannot be reported.

| Configuration | Soft Dice OD | Soft Dice OC | Segmentation loss |
|---|---:|---:|---:|
| A_FULL | 0.946955 | 0.744653 | 0.115214 |
| A_STATIC | 0.946961 | 0.743842 | 0.115404 |
| B_FULL | 0.947002 | 0.746433 | 0.114719 |
| B_STATIC | 0.946992 | 0.746401 | 0.114735 |
| C_FULL | 0.946960 | 0.743536 | 0.115481 |
| C_STATIC | 0.946959 | 0.743525 | 0.115488 |

FULL minus own STATIC soft-Dice OC differences are A +0.000811, B +0.000032, C +0.000011. These small source-only mean differences establish neither target adaptation benefit nor a mechanism. No scientific winner or model selection is issued. Proxy-MSE values are retained in RESULTS.json; cross-method magnitudes alone are not treated as comparable evidence of quality.

## Next bounded plan

First obtain independent source-artifact review, then bind the exact TARGET implementation, six artifacts/contexts, registration, device/resources and output roots for external execution review. Existing TARGET preparation is CPU-only and predates GPU-specific contexts; compatibility must be reviewed explicitly, not assumed or bypassed. Do not edit the completed source attempt or start TARGET from this publication.

The next discriminating experiment remains the frozen 24-job TARGET_SCREEN (seed 20260907, orders 0/1/4, eight fixed arms, 122913 forwards, 5853 backward/Adam calls). Hypothesis: permitted FULL history improves recurrent-stream behavior over independently trained own STATIC; matched controls are own STATIC, C_BASE and C0. Labels are post-hoc scoring only. Report absolute, FULL-C, FULL-own-STATIC, order/recurrence tails and complete costs/failures. First failure stops; no retry/resume or extra seeds/arms. This plan remains unauthorized pending its existing gates. Further tuning candidates require a separately isolated finite plan using permitted source development data; source validation means here do not justify selecting a candidate or spending an additional training budget.

No private paths, patient/content identifiers, raw per-query diagnostics, checkpoints or trained tensors are published. Those remain in protected runtime storage. No source/target pixels were read by this closeout.
