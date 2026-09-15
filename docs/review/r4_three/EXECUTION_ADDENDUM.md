# R4T execution addendum

The user explicitly waived this round's external review and requested immediate execution on 2026-09-15. This is an explicit user waiver, not an external review pass. The authorized finite scope is all 70 trajectories on physical GPUs 5, 6 and 7, with at most three concurrent workers and background execution.

Before any GPU work, startup inspection found that the new entry point omitted the existing neutral-subprocess installer used by R3. The entry now invokes that unchanged installer before dispatch, including in child workers. Scientific code, frozen configuration, data registration, budgets and scheduling are unchanged. No GPU attempt preceded this correction.

The focused CPU execution suite passed 5/5, including an entry-order regression that checks protection is installed before dispatch. Raw output is in `logs/entry-fix-cpu.log`. The earlier 145 local and 38 server test results describe the Stage I implementation, not a rerun after this entry correction.

Stage I SHA and readiness artifacts remain historical. The runtime authorization and private launch receipt bind the new execution commit containing this addendum. The complete review waiver and device authorization are retained privately with the run. GPU launch and completion must be established from live receipts and output, not this document.
