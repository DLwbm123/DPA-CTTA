# r14 fixed 2x2 closeout

[Chinese report](REPORT_ZH.md) and [aggregate results](REPORT.json). Primary analysis is fixed last4000; best is auxiliary. A/C reuse r12; B/D remove only the credit objective contribution. Both seeds have negative A-B; the preregistered credit gate failed. No r15 or extra experiment was started.

The retained combined preregistration was initially a draft. AUTHORIZATION_ZH.md records the subsequent r14-only user authorization; it does not authorize further experiments.

Source training worker preserves the r12 architecture and sampling code, with no credit term in the training objective. The supervisor preserves the original 3h clock and original r13 batch deadline, stops computation 300s early, and uses a 3GB soft stop below the strict 4GB limit. It refuses duplicate/completed launches. WORK_ROOT/config.private.json is locally supplied and excluded. Required keys inherit the r13 configuration, plus expected_parent_source (the published ../src/train_r12.py), expected_core_source (../src/train_r03.py), batch_origin_unix and batch_deadline_unix. The historical receipt must not be reset for a retry; this completed experiment must not be relaunched.

Licensed source data, backbone code/checkpoint, r03 experts and private r12 checkpoints/split/reference/history are required. Public aggregates alone cannot reproduce exact weights or establish patient/backbone independence. See the inherited data-use limitations in r13. The source uses internal labels FULL/CONTROL for new B/D; these must not be confused with reused A/C.

The analyzer uses private reused/{full,control}_SEED and runs/{full,control}_SEED last/best detailed rows. Only aggregate metrics and distribution summaries are exported. No private row, identifier, host path, configuration, cache or weight is published. REUSE_GATE's original 320 count is retained with a separate actual-640 correction receipt.
