# R33 state exchange: started

Update: the user authorized a scheduling handoff, and all six registered jobs are now active, two per GPU. The original three compute processes were preserved. See [scheduling amendment](SCHEDULING.md), [authorization receipt](SCHEDULING_AMENDMENT.json) and [six-worker startup check](PARALLEL_STARTUP_CHECK.json). The profile estimate below predates shared-GPU execution and is retained as a historical estimate.

Status: **RUNNING_COUNTERFACTUALS**, scientific results pending. Three real A100 profiles passed (176 Adam steps/1,440 forwards each), after generated-input mechanical checks and a complete synthetic scorer test. Three of six formal jobs are active; the second batch is queued. Process identities and neutral process arguments were verified.

Scientific code: `0c986726bcd8244e8e873e36b5e126a6b5027db7`. Matrix:3 seeds ×2 orders ×2 query domains ×4 matched-age parameter/Adam combinations =48 branches. All states use native C; no history age reset, controller or LR grid. Diagonal SS/CC must exactly reproduce R32 pre/post outputs; Adam exchange must preserve the first pre-update prediction at fixed parameters. No new labels are released until all six online jobs retire successfully.

Measured profile peak reserved memory:0.607GiB. Estimated formal GPU-worker cost:6.345h. Estimated online end with20% wall reserve:2026-10-09T19:25:08.505491+08:00, followed by CPU scoring. These are estimates, not completion receipts; sharing and storage load may change them. Original T0 is2026-10-09 16:45:03 Beijing; no cumulative GPU cap, with registered engineering deadlines. R32's9.275215GPUh remains in its own ledger.

The narrow question is how Adam content conditions subsequent learning at fixed parameters; R32 already demonstrated prediction-relevant parameter-history effects. Off-diagonal losses can reflect incompatibility. Both query domains, both ordered history pairs, OD/OC, all fixed windows and negative cells remain mandatory. All evidence is development-exposed with unknown patient linkage/ROI; this is not a deployable policy or independent clinical validation.

See [frozen protocol](../../docs/protocols/R33_STATE_EXCHANGE.md), [evidence and interpretation](EVIDENCE.md), [profile admission](PROFILE_ADMISSION.json), [mechanical checks](MECHANICAL_TESTS.json), [startup audit](STARTUP_AUDIT.json) and [query registration](QUERY_REGISTRATION.json).
