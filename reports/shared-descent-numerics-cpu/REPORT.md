# Shared-descent numerical repair: CPU qualification

**Implemented; CPU qualification PASS. No GPU profile or experiment started.**

A deterministic float32 CPU counterexample reproduces the strict descent check
failure after parameter-write rounding, despite a feasible closed-form update.
The repair retains the formula and original tolerance. It audits the representable
commit; if it violates the check or overflows, it restores the saved actual native
Adam proposal bitwise and records pre/post-repair derivatives and the fallback.
This protects the numerical invariant without promising segmentation improvement.

Forty related tests passed in20.232s; after the final existing-profile fallback
counter integration, sixteen focused tests passed in5.971s. An initial29-test
pass took13.224s. All known test-body CPU cost39.427s; GPU0. Nine new tests cover
rounding, native restoration, both dtypes/scales, inactive and disabled behavior,
overflow-safe JSON, original tolerance, end-to-end Host counts/Adam/snapshot/RNG,
failed-call journal, correct step indexing and serialized worker failure receipts.
Existing exact disabled R46/CW_LSO parity and independent dense KKT tests passed.
Calls remain10F/5B/1Adam; disabled joint8F/1B/1Adam. No extra network pass or step.

Failed descent guards now carry the exact available numerical diagnostics into
the physical journal and worker receipt. The online call index is captured before
Host.step, so a post-increment guard failure is no longer labelled one step late.
NaN/inf diagnostics use explicit strings and cannot break strict JSON writing.
No unavailable R47 failing-step values are fabricated, and no historical R47
source, attempts, protocol or cost ledger was changed.

See docs/SHARED_DESCENT_NUMERICS_CPU.md and CPU_CHECKS.json for full analysis.
This branch is CPU-only feature/shared-descent-numerics-cpu. Restoring native Adam
on unsafe filtered commits may reduce how often the joint module acts; frequency,
GPU memory/time, real-model invariant coverage and Dice efficacy are unmeasured.
The demonstrated rounding class is not proof of the exact two R47 failure causes.
No deployment, GPU work, automatic monitor, retry or next experiment is included.
